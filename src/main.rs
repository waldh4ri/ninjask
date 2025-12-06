//! Ninjask - CSV Explorer TUI
//!
//! Terminal-based CSV viewer for large datasets (millions of rows).
//! Uses virtual scrolling and zero-copy slicing for efficient navigation.
//!
//! # Features
//! - Virtual viewport: only renders visible rows
//! - Zero-copy slicing: fast DataFrame operations

use std::env;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use polars::prelude::*;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Span},
    widgets::{
        Block, Borders, Cell, Clear, List, ListItem, ListState, Paragraph, Row, Scrollbar,
        ScrollbarOrientation, ScrollbarState, Table, TableState,
    },
    Frame, Terminal,
};
use tui_input::backend::crossterm::EventHandler;
use tui_input::Input;

/// Default CSV file path
const DEFAULT_CSV_PATH: &str = "data.csv";

/// Dummy rows to generate if CSV is missing
const DUMMY_ROWS: usize = 100_000;

/// Parse command-line arguments and return CSV file path
fn parse_args() -> String {
    let args: Vec<String> = env::args().collect();
    let mut csv_path = DEFAULT_CSV_PATH.to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-f" | "--file" => {
                if i + 1 < args.len() {
                    csv_path = args[i + 1].clone();
                    i += 2;
                } else {
                    eprintln!("Error: -f requires a file path argument");
                    eprintln!("Usage: ninjask [-f <csv_file>]");
                    std::process::exit(1);
                }
            }
            "-h" | "--help" => {
                println!("Ninjask - High-Performance CSV Explorer TUI");
                println!();
                println!("Usage: ninjask [OPTIONS]");
                println!();
                println!("Options:");
                println!("  -f, --file <PATH>  CSV file to load (default: data.csv)");
                println!("  -h, --help         Show this help message");
                println!();
                println!("Keybindings:");
                println!("  j/k or ↑/↓         Navigate rows");
                println!("  h/l or ←/→         Navigate columns");
                println!("  /                  Search/filter data");
                println!("  s                  Sort by selected column");
                println!("  c                  Column visibility picker");
                println!("  q                  Quit");
                std::process::exit(0);
            }
            arg if arg.starts_with('-') => {
                eprintln!("Unknown option: {}", arg);
                eprintln!("Usage: ninjask [-f <csv_file>]");
                std::process::exit(1);
            }
            // Positional argument - treat as file path
            path => {
                csv_path = path.to_string();
                i += 1;
            }
        }
    }

    csv_path
}

/// Input poll timeout (~60 FPS)
const POLL_TIMEOUT: Duration = Duration::from_millis(16);

/// Truncate string to fit display width (Unicode-safe)
fn truncate_string(s: &str, max_chars: usize) -> String {
    let char_count = s.chars().count();
    if char_count <= max_chars {
        s.to_string()
    } else if max_chars <= 1 {
        "…".to_string()
    } else {
        let truncated: String = s.chars().take(max_chars - 1).collect();
        format!("{}…", truncated)
    }
}

// ================= Application State Types =================

/// Application mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AppMode {
    /// Normal browsing mode - navigate and view data
    Normal,
    /// Search mode - typing in the search bar
    EditingSearch,
    /// Column selection popup is open
    ColumnSelection,
    /// Cell detail popup is open
    CellDetail,
}

/// Sort order for columns
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SortOrder {
    /// Original order
    None,
    /// Ascending
    Ascending,
    /// Descending
    Descending,
}

impl SortOrder {
    /// Cycle sort order
    fn next(self) -> Self {
        match self {
            SortOrder::None => SortOrder::Ascending,
            SortOrder::Ascending => SortOrder::Descending,
            SortOrder::Descending => SortOrder::None,
        }
    }

    /// Display indicator for sort order
    fn indicator(self) -> &'static str {
        match self {
            SortOrder::None => "",
            SortOrder::Ascending => " ▲",
            SortOrder::Descending => " ▼",
        }
    }
}

/// Column state: visibility and width
#[derive(Debug, Clone)]
struct ColumnState {
    name: String,
    visible: bool,
    /// Cached display width (auto-calculated from data)
    display_width: u16,
}

/// Main application state
struct App {
    /// Original DataFrame (Arc for cheap clone)
    original_df: Arc<DataFrame>,
    /// Current DataFrame (filtered/sorted)
    current_df: Arc<DataFrame>,
    /// Column states
    columns: Vec<ColumnState>,
    /// Selected column index
    selected_column: usize,
    /// Sort state: (column_index, sort_order)
    sort_state: Option<(usize, SortOrder)>,
    /// Table state (row selection/scroll)
    table_state: TableState,
    /// Current mode
    mode: AppMode,
    /// Search input state
    search_input: Input,
    /// Search query
    search_query: String,
    /// Column picker state
    column_list_state: ListState,
    /// Index of column being moved
    moving_column: Option<usize>,
    /// Exit flag
    should_quit: bool,
    /// Status message
    status_message: String,
    /// Total row count
    total_rows: usize,
    /// Filtered row count
    filtered_rows: usize,
    /// Viewport height (rows visible)
    viewport_height: usize,
    /// Cell detail content
    cell_detail_content: String,
    /// Cell detail column name
    cell_detail_column: String,
    /// CSV load time (ms)
    load_time_ms: u64,
    /// Last search time (ms)
    search_time_ms: u64,
}

impl App {
    /// Create new App instance
    fn new(df: DataFrame) -> Self {
        let column_names: Vec<String> = df
            .get_column_names()
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        let mut columns: Vec<ColumnState> = column_names
            .iter()
            .map(|name| ColumnState {
                name: name.clone(),
                visible: true,
                display_width: 10, // Will be calculated below
            })
            .collect();

        // Calculate initial column widths by sampling data
        Self::calculate_column_widths(&df, &mut columns, 500);

        let total_rows = df.height();

        let mut table_state = TableState::default();
        table_state.select(Some(0));

        let mut column_list_state = ListState::default();
        column_list_state.select(Some(0));

        let df_arc = Arc::new(df);

        Self {
            original_df: Arc::clone(&df_arc),
            current_df: df_arc,
            columns,
            selected_column: 0,
            sort_state: None,
            table_state,
            mode: AppMode::Normal,
            search_input: Input::default(),
            search_query: String::new(),
            column_list_state,
            moving_column: None,
            should_quit: false,
            status_message: String::from("Ready. Press ? for help."),
            total_rows,
            filtered_rows: total_rows,
            viewport_height: 20, // Default, will be updated on first render
            cell_detail_content: String::new(),
            cell_detail_column: String::new(),
            load_time_ms: 0,
            search_time_ms: 0,
        }
    }

    /// Calculate column widths by sampling data
    fn calculate_column_widths(df: &DataFrame, columns: &mut [ColumnState], sample_size: usize) {
        const MIN_WIDTH: u16 = 8;
        const MAX_WIDTH: u16 = 70;

        let sample_rows = sample_size.min(df.height());

        for col_state in columns.iter_mut() {
            // Start with header width (add 2 for sort indicator)
            let mut max_width = col_state.name.chars().count() + 2;

            // Sample first N rows to find max width
            if let Ok(col) = df.column(&col_state.name) {
                for i in 0..sample_rows {
                    if let Ok(val) = col.get(i) {
                        let formatted = format!("{}", val);
                        let len = formatted.chars().count();
                        max_width = max_width.max(len);
                    }
                }
            }

            // Clamp to reasonable bounds and add padding
            col_state.display_width = ((max_width + 1) as u16).clamp(MIN_WIDTH, MAX_WIDTH);
        }
    }

    /// Recalculate column widths from filtered data
    fn recalculate_column_widths(&mut self) {
        Self::calculate_column_widths(&self.current_df, &mut self.columns, 500);
    }

    /// Get visible column names
    fn visible_columns(&self) -> Vec<&str> {
        self.columns
            .iter()
            .filter(|c| c.visible)
            .map(|c| c.name.as_str())
            .collect()
    }

    /// Get index of visible column in original list
    fn visible_column_original_index(&self, visible_idx: usize) -> Option<usize> {
        self.columns
            .iter()
            .enumerate()
            .filter(|(_, c)| c.visible)
            .nth(visible_idx)
            .map(|(idx, _)| idx)
    }

    /// Apply search filter to DataFrame
    fn apply_search_filter(&mut self) -> Result<()> {
        let start = Instant::now();

        if self.search_query.is_empty() {
            // Reset to original (with sort applied if any)
            self.current_df = Arc::clone(&self.original_df);
        } else {
            // Build a filter that matches the search query across all string columns.
            // Arc::clone is cheap - just increments reference count, then we need to clone the DataFrame for lazy.
            let df = (*self.original_df).clone().lazy();

            // Treat the user input as a full regex. We default to case-insensitive matching by
            // prefixing (?i) unless the user already provided inline flags at the start.
            let pattern = if self.search_query.starts_with("(?") {
                self.search_query.clone()
            } else {
                format!("(?i){}", self.search_query)
            };

            // Validate the regex early to avoid runtime errors in Polars expressions.
            if let Err(err) = regex::Regex::new(&pattern) {
                self.status_message = format!("Invalid regex: {}", err);
                self.search_time_ms = start.elapsed().as_millis() as u64;
                return Ok(());
            }

            let visible_cols = self.visible_columns();
            if visible_cols.is_empty() {
                self.current_df = Arc::clone(&self.original_df);
                self.search_time_ms = start.elapsed().as_millis() as u64;
                return Ok(());
            }

            // Build filter expression: col1.contains(pattern) OR col2.contains(pattern) OR ...
            let mut filter_expr: Option<Expr> = None;

            for col_name in &visible_cols {
                let col_filter = col(*col_name)
                    .cast(DataType::String)
                    .str()
                    .contains(lit(pattern.clone()), false);

                filter_expr = Some(match filter_expr {
                    Some(expr) => expr.or(col_filter),
                    None => col_filter,
                });
            }

            if let Some(expr) = filter_expr {
                self.current_df = Arc::new(df.filter(expr).collect()?);
            }
        }

        // Re-apply sort if active
        self.apply_current_sort()?;

        self.filtered_rows = self.current_df.height();

        // Recalculate column widths for filtered data
        self.recalculate_column_widths();

        // Reset selection if out of bounds
        if self.filtered_rows == 0 {
            self.table_state.select(None);
        } else if let Some(selected) = self.table_state.selected() {
            if selected >= self.filtered_rows {
                self.table_state.select(Some(self.filtered_rows - 1));
            }
        }

        // Record search time
        self.search_time_ms = start.elapsed().as_millis() as u64;

        Ok(())
    }

    /// Apply sort to DataFrame
    fn apply_current_sort(&mut self) -> Result<()> {
        if let Some((col_idx, order)) = self.sort_state {
            if order != SortOrder::None {
                let col_name = &self.columns[col_idx].name;
                let descending = order == SortOrder::Descending;

                // Clone the Arc (cheap) and dereference to clone the DataFrame (necessary for lazy)
                self.current_df = Arc::new(
                    (*self.current_df).clone()
                        .lazy()
                        .sort(
                            [col_name],
                            SortMultipleOptions::new().with_order_descending(descending),
                        )
                        .collect()?
                );
            }
        }
        Ok(())
    }

    /// Cycle sort order on selected column
    fn cycle_sort(&mut self) -> Result<()> {
        // Get the original index of the currently selected visible column
        let original_idx = match self.visible_column_original_index(self.selected_column) {
            Some(idx) => idx,
            None => return Ok(()),
        };

        let new_order = if let Some((col_idx, order)) = self.sort_state {
            if col_idx == original_idx {
                order.next()
            } else {
                SortOrder::Ascending
            }
        } else {
            SortOrder::Ascending
        };

        self.sort_state = if new_order == SortOrder::None {
            None
        } else {
            Some((original_idx, new_order))
        };

        // Reset to filtered data (or original if no filter)
        if self.search_query.is_empty() {
            self.current_df = Arc::clone(&self.original_df);
        } else {
            self.apply_search_filter()?;
            return Ok(());
        }

        self.apply_current_sort()?;

        let col_name = &self.columns[original_idx].name;
        self.status_message = format!(
            "Sort: {} {}",
            col_name,
            match new_order {
                SortOrder::None => "(none)",
                SortOrder::Ascending => "(ascending)",
                SortOrder::Descending => "(descending)",
            }
        );

        Ok(())
    }

    /// Toggle column visibility
    fn toggle_column_visibility(&mut self) {
        if let Some(idx) = self.column_list_state.selected() {
            if idx < self.columns.len() {
                // Don't allow hiding all columns
                let visible_count = self.columns.iter().filter(|c| c.visible).count();
                if visible_count > 1 || !self.columns[idx].visible {
                    self.columns[idx].visible = !self.columns[idx].visible;
                } else {
                    self.status_message = "Cannot hide all columns!".to_string();
                }
            }
        }

        // Reset selected column if out of bounds
        let visible_count = self.columns.iter().filter(|c| c.visible).count();
        if self.selected_column >= visible_count {
            self.selected_column = visible_count.saturating_sub(1);
        }
    }

    /// Move selection (rows/columns)
    fn move_selection(&mut self, delta_row: i32, delta_col: i32) {
        // Handle row movement
        if delta_row != 0 {
            let current = self.table_state.selected().unwrap_or(0);
            let max_row = self.current_df.height().saturating_sub(1);

            let new_row = if delta_row > 0 {
                (current + delta_row as usize).min(max_row)
            } else {
                current.saturating_sub((-delta_row) as usize)
            };

            self.table_state.select(Some(new_row));

            // Adjust scroll offset to keep selection visible
            let offset = self.table_state.offset();
            if new_row < offset {
                // Selection moved above viewport - scroll up
                *self.table_state.offset_mut() = new_row;
            } else if self.viewport_height > 0 && new_row >= offset + self.viewport_height {
                // Selection moved below viewport - scroll down
                *self.table_state.offset_mut() = new_row - self.viewport_height + 1;
            }
        }

        // Handle column movement
        if delta_col != 0 {
            let visible_count = self.columns.iter().filter(|c| c.visible).count();
            if visible_count > 0 {
                let max_col = visible_count - 1;

                let new_col = if delta_col > 0 {
                    (self.selected_column + delta_col as usize).min(max_col)
                } else {
                    self.selected_column.saturating_sub((-delta_col) as usize)
                };

                self.selected_column = new_col;
            }
        }
    }

    /// Page up/down
    fn page_move(&mut self, page_size: usize, down: bool) {
        let current = self.table_state.selected().unwrap_or(0);
        let max_row = self.current_df.height().saturating_sub(1);

        let new_row = if down {
            (current + page_size).min(max_row)
        } else {
            current.saturating_sub(page_size)
        };

        self.table_state.select(Some(new_row));

        // Adjust scroll offset to keep selection visible
        let offset = self.table_state.offset();
        if new_row < offset {
            *self.table_state.offset_mut() = new_row;
        } else if self.viewport_height > 0 && new_row >= offset + self.viewport_height {
            *self.table_state.offset_mut() = new_row - self.viewport_height + 1;
        }
    }

    /// Jump to start/end
    fn jump_to(&mut self, start: bool) {
        if self.current_df.height() > 0 {
            let new_row = if start {
                0
            } else {
                self.current_df.height() - 1
            };
            self.table_state.select(Some(new_row));

            // Adjust scroll offset
            if start {
                *self.table_state.offset_mut() = 0;
            } else if self.viewport_height > 0 {
                let max_offset = self.current_df.height().saturating_sub(self.viewport_height);
                *self.table_state.offset_mut() = max_offset;
            }
        }
    }

    /// Handle input events by mode
    fn handle_event(&mut self, event: Event) -> Result<()> {
        match self.mode {
            AppMode::Normal => self.handle_normal_mode(event)?,
            AppMode::EditingSearch => self.handle_search_mode(event)?,
            AppMode::ColumnSelection => self.handle_column_selection_mode(event)?,
            AppMode::CellDetail => self.handle_cell_detail_mode(event)?,
        }
        Ok(())
    }

    fn handle_normal_mode(&mut self, event: Event) -> Result<()> {
        if let Event::Key(key) = event {
            match key.code {
                // Quit
                KeyCode::Char('q') => self.should_quit = true,
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.should_quit = true
                }

                // Navigation - Vim style
                KeyCode::Char('j') | KeyCode::Down => self.move_selection(1, 0),
                KeyCode::Char('k') | KeyCode::Up => self.move_selection(-1, 0),
                KeyCode::Char('l') | KeyCode::Right => self.move_selection(0, 1),
                KeyCode::Char('h') | KeyCode::Left => self.move_selection(0, -1),

                // Page navigation
                KeyCode::PageDown | KeyCode::Char('d')
                    if key.modifiers.contains(KeyModifiers::CONTROL) =>
                {
                    self.page_move(20, true)
                }
                KeyCode::PageUp | KeyCode::Char('u')
                    if key.modifiers.contains(KeyModifiers::CONTROL) =>
                {
                    self.page_move(20, false)
                }

                // Jump to start/end
                KeyCode::Home | KeyCode::Char('g') => self.jump_to(true),
                KeyCode::End | KeyCode::Char('G') => self.jump_to(false),

                // Search mode
                KeyCode::Char('/') => {
                    self.mode = AppMode::EditingSearch;
                    self.status_message = "Search (regex): Type pattern and press Enter".to_string();
                }

                // Sort
                KeyCode::Char('s') => {
                    self.cycle_sort()?;
                }

                // Column selection
                KeyCode::Char('c') => {
                    self.mode = AppMode::ColumnSelection;
                    self.status_message = "Column Selection: Space to toggle, Enter to confirm".to_string();
                }

                // Clear search
                KeyCode::Esc => {
                    if !self.search_query.is_empty() {
                        self.search_query.clear();
                        self.search_input.reset();
                        self.apply_search_filter()?;
                        self.status_message = "Search cleared".to_string();
                    }
                }

                // Help
                KeyCode::Char('?') => {
                    self.status_message = "Keys: j/k=↑↓ h/l=←→ /=regex search s=sort c=columns Enter=view cell q=quit".to_string();
                }

                // View cell detail
                KeyCode::Enter => {
                    self.open_cell_detail();
                }

                _ => {}
            }
        }
        Ok(())
    }

    /// Open cell detail popup
    fn open_cell_detail(&mut self) {
        // Get visible column names as owned strings to avoid borrow issues
        let visible_cols: Vec<String> = self
            .columns
            .iter()
            .filter(|c| c.visible)
            .map(|c| c.name.clone())
            .collect();

        if visible_cols.is_empty() || self.current_df.height() == 0 {
            self.status_message = "No cell to display".to_string();
            return;
        }

        let row_idx = self.table_state.selected().unwrap_or(0);
        if row_idx >= self.current_df.height() {
            self.status_message = "Invalid row selection".to_string();
            return;
        }

        if self.selected_column >= visible_cols.len() {
            self.status_message = "Invalid column selection".to_string();
            return;
        }

        let col_name = &visible_cols[self.selected_column];
        self.cell_detail_column = col_name.clone();

        // Get the cell value
        if let Ok(col) = self.current_df.column(col_name.as_str()) {
            if let Ok(val) = col.get(row_idx) {
                self.cell_detail_content = format!("{}", val);
            } else {
                self.cell_detail_content = "<error reading value>".to_string();
            }
        } else {
            self.cell_detail_content = "<column not found>".to_string();
        }

        self.mode = AppMode::CellDetail;
        self.status_message = "Press Enter to close".to_string();
    }

    fn handle_cell_detail_mode(&mut self, event: Event) -> Result<()> {
        if let Event::Key(key) = event {
            match key.code {
                KeyCode::Enter | KeyCode::Esc | KeyCode::Char('q') => {
                    self.mode = AppMode::Normal;
                    self.status_message = "Ready. Press ? for help.".to_string();
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn handle_search_mode(&mut self, event: Event) -> Result<()> {
        if let Event::Key(key) = event {
            match key.code {
                KeyCode::Enter => {
                    // Apply search
                    self.search_query = self.search_input.value().to_string();
                    self.apply_search_filter()?;
                    self.mode = AppMode::Normal;
                    // Jump to first row to display found entries
                    if self.filtered_rows > 0 {
                        self.table_state.select(Some(0));
                        *self.table_state.offset_mut() = 0;
                    }
                    self.status_message = if self.search_query.is_empty() {
                        "Search cleared".to_string()
                    } else {
                        format!("Found {} matches for '{}'", self.filtered_rows, self.search_query)
                    };
                }
                KeyCode::Esc => {
                    // Cancel search, keep previous query
                    self.search_input = Input::new(self.search_query.clone());
                    self.mode = AppMode::Normal;
                    self.status_message = "Search cancelled".to_string();
                }
                _ => {
                    // Forward to input widget
                    self.search_input.handle_event(&event);
                }
            }
        }
        Ok(())
    }

    fn handle_column_selection_mode(&mut self, event: Event) -> Result<()> {
        if let Event::Key(key) = event {
            match key.code {
                KeyCode::Esc => {
                    if self.moving_column.is_some() {
                        // Cancel move mode
                        self.moving_column = None;
                        self.status_message = "Move cancelled".to_string();
                    } else {
                        // Exit column selection
                        self.mode = AppMode::Normal;
                        self.status_message = format!(
                            "{} columns visible",
                            self.columns.iter().filter(|c| c.visible).count()
                        );
                    }
                }
                KeyCode::Enter => {
                    if self.moving_column.is_some() {
                        // Confirm move and exit move mode
                        self.moving_column = None;
                        self.status_message = "Column moved".to_string();
                    } else {
                        // Exit column selection
                        self.mode = AppMode::Normal;
                        self.status_message = format!(
                            "{} columns visible",
                            self.columns.iter().filter(|c| c.visible).count()
                        );
                    }
                }
                KeyCode::Char('c') => {
                    if self.moving_column.is_none() {
                        self.mode = AppMode::Normal;
                        self.status_message = format!(
                            "{} columns visible",
                            self.columns.iter().filter(|c| c.visible).count()
                        );
                    }
                }
                KeyCode::Char('j') | KeyCode::Down => {
                    let current = self.column_list_state.selected().unwrap_or(0);
                    let max = self.columns.len().saturating_sub(1);
                    if current < max {
                        if self.moving_column.is_some() {
                            // Move column down
                            self.columns.swap(current, current + 1);
                            // Update sort_state if it references moved columns
                            if let Some((sort_idx, order)) = self.sort_state {
                                if sort_idx == current {
                                    self.sort_state = Some((current + 1, order));
                                } else if sort_idx == current + 1 {
                                    self.sort_state = Some((current, order));
                                }
                            }
                            self.moving_column = Some(current + 1);
                        }
                        self.column_list_state.select(Some(current + 1));
                    }
                }
                KeyCode::Char('k') | KeyCode::Up => {
                    let current = self.column_list_state.selected().unwrap_or(0);
                    if current > 0 {
                        if self.moving_column.is_some() {
                            // Move column up
                            self.columns.swap(current, current - 1);
                            // Update sort_state if it references moved columns
                            if let Some((sort_idx, order)) = self.sort_state {
                                if sort_idx == current {
                                    self.sort_state = Some((current - 1, order));
                                } else if sort_idx == current - 1 {
                                    self.sort_state = Some((current, order));
                                }
                            }
                            self.moving_column = Some(current - 1);
                        }
                        self.column_list_state.select(Some(current - 1));
                    }
                }
                KeyCode::Char('l') | KeyCode::Right => {
                    let current = self.column_list_state.selected().unwrap_or(0);
                    let max = self.columns.len().saturating_sub(1);
                    if current < max && self.moving_column.is_some() {
                        // Move column down (right = down in list context)
                        self.columns.swap(current, current + 1);
                        if let Some((sort_idx, order)) = self.sort_state {
                            if sort_idx == current {
                                self.sort_state = Some((current + 1, order));
                            } else if sort_idx == current + 1 {
                                self.sort_state = Some((current, order));
                            }
                        }
                        self.moving_column = Some(current + 1);
                        self.column_list_state.select(Some(current + 1));
                    }
                }
                KeyCode::Char('h') | KeyCode::Left => {
                    let current = self.column_list_state.selected().unwrap_or(0);
                    if current > 0 && self.moving_column.is_some() {
                        // Move column up (left = up in list context)
                        self.columns.swap(current, current - 1);
                        if let Some((sort_idx, order)) = self.sort_state {
                            if sort_idx == current {
                                self.sort_state = Some((current - 1, order));
                            } else if sort_idx == current - 1 {
                                self.sort_state = Some((current, order));
                            }
                        }
                        self.moving_column = Some(current - 1);
                        self.column_list_state.select(Some(current - 1));
                    }
                }
                KeyCode::Char(' ') => {
                    if self.moving_column.is_none() {
                        self.toggle_column_visibility();
                    }
                }
                // Toggle move mode
                KeyCode::Char('m') => {
                    if let Some(_) = self.moving_column {
                        // Exit move mode
                        self.moving_column = None;
                        self.status_message = "Column placed".to_string();
                    } else {
                        // Enter move mode
                        let current = self.column_list_state.selected().unwrap_or(0);
                        self.moving_column = Some(current);
                        self.status_message = "Move mode: use j/k or arrows to move, Enter/m to confirm, Esc to cancel".to_string();
                    }
                }
                // Select all columns
                KeyCode::Char('a') => {
                    if self.moving_column.is_none() {
                        for col in &mut self.columns {
                            col.visible = true;
                        }
                        self.status_message = "All columns selected".to_string();
                    }
                }
                // Unselect all columns (except keep at least one visible)
                KeyCode::Char('n') => {
                    if self.moving_column.is_none() {
                        // Keep only the first column visible
                        for (idx, col) in self.columns.iter_mut().enumerate() {
                            col.visible = idx == 0;
                        }
                        self.selected_column = 0;
                        self.status_message = "All columns hidden except first".to_string();
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
}

// ============================================================================
// UI Rendering
// ============================================================================

/// Main UI rendering
fn ui(frame: &mut Frame, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header
            Constraint::Min(10),   // Table
            Constraint::Length(3), // Status/Search bar
        ])
        .split(frame.area());

    render_header(frame, app, chunks[0]);
    render_table(frame, app, chunks[1]);
    render_status_bar(frame, app, chunks[2]);

    // Render column selection popup if active
    if app.mode == AppMode::ColumnSelection {
        render_column_popup(frame, app);
    }

    // Render cell detail popup if active
    if app.mode == AppMode::CellDetail {
        render_cell_detail_popup(frame, app);
    }
}

fn render_header(frame: &mut Frame, app: &App, area: Rect) {
    let timing_info = if app.load_time_ms > 0 {
        if app.search_time_ms > 0 {
            format!(
                " │ Load: {}ms │ Search: {}ms",
                app.load_time_ms, app.search_time_ms
            )
        } else {
            format!(" │ Load: {}ms", app.load_time_ms)
        }
    } else {
        String::new()
    };

    let header_text = format!(
        " Ninjask │ Rows: {} / {} │ Cols: {} │ {}{}",
        app.filtered_rows,
        app.total_rows,
        app.columns.iter().filter(|c| c.visible).count(),
        if app.search_query.is_empty() {
            "No filter"
        } else {
            &app.search_query
        },
        timing_info
    );

    let header = Paragraph::new(header_text)
        .style(Style::default().fg(Color::Cyan).bold())
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray))
                .title(" CSV Explorer ")
                .title_alignment(Alignment::Center),
        );

    frame.render_widget(header, area);
}

/// Render main data table with virtual scrolling
fn render_table(frame: &mut Frame, app: &mut App, area: Rect) {
    // Calculate viewport dimensions first and store it
    // Account for: borders (2), header row (1)
    let table_height = area.height.saturating_sub(3) as usize;
    app.viewport_height = table_height;

    let visible_cols = app.visible_columns();

    if visible_cols.is_empty() || app.current_df.height() == 0 {
        let empty = Paragraph::new("No data to display")
            .alignment(Alignment::Center)
            .block(Block::default().borders(Borders::ALL));
        frame.render_widget(empty, area);
        return;
    }

    // Use cached auto-calculated column widths
    let widths: Vec<Constraint> = app
        .columns
        .iter()
        .filter(|c| c.visible)
        .map(|c| Constraint::Length(c.display_width))
        .collect();

    // Build header row with sort indicators
    let header_cells: Vec<Cell> = visible_cols
        .iter()
        .enumerate()
        .map(|(idx, col_name)| {
            let original_idx = app.visible_column_original_index(idx);

            // Check if this column is sorted
            let sort_indicator = if let Some((sort_col, order)) = app.sort_state {
                if original_idx == Some(sort_col) {
                    order.indicator()
                } else {
                    ""
                }
            } else {
                ""
            };

            let style = if idx == app.selected_column {
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD | Modifier::UNDERLINED)
            } else {
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            };

            Cell::from(format!("{}{}", col_name, sort_indicator)).style(style)
        })
        .collect();

    let header = Row::new(header_cells)
        .style(Style::default().bg(Color::DarkGray))
        .height(1);

    // --- Virtual viewport logic: only render visible rows ---

    // Calculate viewport dimensions
    // Account for: borders (2), header row (1)
    let total_rows = app.current_df.height();
    let table_height = app.viewport_height;

    // Get current scroll position
    let selected = app.table_state.selected().unwrap_or(0);

    // Calculate the visible window
    // We want the selected row to be visible, preferably centered
    let offset = app.table_state.offset();
    let viewport_start = offset;
    let viewport_end = (viewport_start + table_height).min(total_rows);

    // Slice the DataFrame - THIS IS THE KEY OPTIMIZATION
    // Polars slice is O(1) - it just adjusts pointers, no data copying
    let visible_row_count = viewport_end.saturating_sub(viewport_start);

    let rows: Vec<Row> = if visible_row_count > 0 {
        // Select only visible columns first
        let col_names: Vec<&str> = visible_cols.clone();
        let df_view = app
            .current_df
            .select(col_names)
            .unwrap_or_else(|_| (*app.current_df).clone());

        // Slice to visible rows only
        let sliced = df_view.slice(viewport_start as i64, visible_row_count);

        // Convert visible rows to UI widgets
        (0..sliced.height())
            .map(|row_idx| {
                let cells: Vec<Cell> = (0..sliced.width())
                    .map(|col_idx| {
                        let value = sliced
                            .get_columns()
                            .get(col_idx)
                            .and_then(|s| s.get(row_idx).ok())
                            .map(|v| format!("{}", v))
                            .unwrap_or_default();

                        // Get the width for this column
                        let col_width = app
                            .columns
                            .iter()
                            .filter(|c| c.visible)
                            .nth(col_idx)
                            .map(|c| c.display_width as usize)
                            .unwrap_or(10);

                        // Truncate long values for display (Unicode-safe)
                        let display_value = truncate_string(&value, col_width.saturating_sub(1));

                        Cell::from(display_value)
                    })
                    .collect();

                let actual_row_idx = viewport_start + row_idx;
                let style = if Some(actual_row_idx) == app.table_state.selected() {
                    Style::default().bg(Color::Blue).fg(Color::White)
                } else if actual_row_idx % 2 == 0 {
                    Style::default().bg(Color::Rgb(30, 30, 40))
                } else {
                    Style::default()
                };

                Row::new(cells).style(style).height(1)
            })
            .collect()
    } else {
        vec![]
    };

    let table = Table::new(rows, &widths)
        .header(header)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray))
                .title(format!(" Data [Row {}/{}] ", selected + 1, total_rows)),
        );

    // Use render_widget instead of render_stateful_widget since we manually
    // handle scrolling and row highlighting in the virtual viewport
    frame.render_widget(table, area);

    // Render scrollbar
    if total_rows > table_height {
        let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
            .begin_symbol(Some("↑"))
            .end_symbol(Some("↓"));

        let mut scrollbar_state =
            ScrollbarState::new(total_rows.saturating_sub(table_height)).position(offset);

        frame.render_stateful_widget(
            scrollbar,
            area.inner(ratatui::layout::Margin {
                vertical: 1,
                horizontal: 0,
            }),
            &mut scrollbar_state,
        );
    }
}

fn render_status_bar(frame: &mut Frame, app: &App, area: Rect) {
    match app.mode {
        AppMode::EditingSearch => {
            // Show search input
            let chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Length(10), Constraint::Min(10)])
                .split(area);

            let label = Paragraph::new(" Search: ")
                .style(Style::default().fg(Color::Yellow).bold());
            frame.render_widget(label, chunks[0]);

            let input = Paragraph::new(app.search_input.value())
                .style(Style::default().fg(Color::White))
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(Color::Yellow)),
                );
            frame.render_widget(input, chunks[1]);

            // Show cursor
            frame.set_cursor_position((
                chunks[1].x + app.search_input.visual_cursor() as u16 + 1,
                chunks[1].y + 1,
            ));
        }
        _ => {
            // Show status message
            let status = Paragraph::new(Line::from(vec![
                Span::styled(" ", Style::default()),
                Span::styled(&app.status_message, Style::default().fg(Color::Gray)),
                Span::styled(
                    " │ ",
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(
                    "/ search  s sort  c columns  q quit",
                    Style::default().fg(Color::DarkGray),
                ),
            ]))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::DarkGray)),
            );
            frame.render_widget(status, area);
        }
    }
}

fn render_column_popup(frame: &mut Frame, app: &mut App) {
    // Create centered popup
    let area = centered_rect(50, 60, frame.area());

    // Clear background
    frame.render_widget(Clear, area);

    // Create list items
    let items: Vec<ListItem> = app
        .columns
        .iter()
        .enumerate()
        .map(|(idx, col)| {
            let checkbox = if col.visible { "[✓]" } else { "[ ]" };
            let is_being_moved = app.moving_column == Some(idx);
            
            let style = if is_being_moved {
                Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)
            } else if col.visible {
                Style::default().fg(Color::Green)
            } else {
                Style::default().fg(Color::DarkGray)
            };
            
            let move_indicator = if is_being_moved { " ↕" } else { "" };
            ListItem::new(format!("{} {}{}", checkbox, col.name, move_indicator)).style(style)
        })
        .collect();

    let title = if app.moving_column.is_some() {
        " MOVING: j/k/arrows=move, Enter/m=place, Esc=cancel "
    } else {
        " Columns: Space=toggle, m=move, a=all, n=none, Enter=done "
    };

    let border_color = if app.moving_column.is_some() {
        Color::Magenta
    } else {
        Color::Yellow
    };

    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(border_color))
                .title(title)
                .title_alignment(Alignment::Center),
        )
        .highlight_style(
            Style::default()
                .bg(if app.moving_column.is_some() { Color::Magenta } else { Color::Yellow })
                .fg(Color::Black)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol(if app.moving_column.is_some() { "≡ " } else { "▶ " });

    frame.render_stateful_widget(list, area, &mut app.column_list_state);
}

fn render_cell_detail_popup(frame: &mut Frame, app: &App) {
    // Create centered popup
    let area = centered_rect(70, 50, frame.area());

    // Clear background
    frame.render_widget(Clear, area);

    // Split the popup into title area and content area
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(3)])
        .split(area);

    // Render column name as header
    let header = Paragraph::new(format!(" Column: {} ", app.cell_detail_column))
        .style(Style::default().fg(Color::Yellow).bold())
        .alignment(Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::TOP | Borders::LEFT | Borders::RIGHT)
                .border_style(Style::default().fg(Color::Yellow))
                .title(" Cell Detail ")
                .title_alignment(Alignment::Center),
        );
    frame.render_widget(header, chunks[0]);

    // Render cell content with word wrapping
    let content = Paragraph::new(app.cell_detail_content.clone())
        .style(Style::default().fg(Color::White))
        .wrap(ratatui::widgets::Wrap { trim: false })
        .block(
            Block::default()
                .borders(Borders::BOTTOM | Borders::LEFT | Borders::RIGHT)
                .border_style(Style::default().fg(Color::Yellow))
                .title(" Press Enter to close ")
                .title_alignment(Alignment::Center)
                .title_position(ratatui::widgets::block::Position::Bottom),
        );
    frame.render_widget(content, chunks[1]);
}

/// Create centered rectangle
fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

// ================= Data Loading =================

/// Load CSV file or generate dummy data if missing
fn load_data(path: &str) -> Result<(DataFrame, u64)> {
    let start = Instant::now();

    if Path::new(path).exists() {
        eprintln!("Loading CSV: {}", path);

        // Load CSV with error handling for malformed lines
        let parse_options = CsvParseOptions::default()
            .with_truncate_ragged_lines(true); // Handle lines with wrong number of fields

        let df = CsvReadOptions::default()
            .with_has_header(true)
            .with_infer_schema_length(Some(1000)) // Infer types from first 1000 rows (Int64, Float64, String, etc.)
            .with_ignore_errors(true) // Skip rows that can't be parsed instead of failing
            .with_parse_options(parse_options)
            .try_into_reader_with_file_path(Some(path.into()))
            .with_context(|| format!("Failed to create CSV reader for '{}'", path))?
            .finish()
            .with_context(|| format!("Failed to parse CSV file '{}'. The file may be corrupted or have an unsupported format.", path))?;

        if df.height() == 0 {
            eprintln!("Warning: CSV file is empty or all rows were skipped due to errors");
        }

        // Log detected schema
        eprintln!(
            "Loaded {} rows x {} columns",
            df.height(),
            df.width()
        );
        for col in df.get_columns() {
            eprintln!("  - {}: {:?}", col.name(), col.dtype());
        }

        let load_time = start.elapsed().as_millis() as u64;
        Ok((df, load_time))
    } else {
        eprintln!("CSV not found, generating {} dummy rows...", DUMMY_ROWS);
        generate_dummy_csv(path)?;
        load_data(path)
    }
}

/// Generate dummy CSV file
fn generate_dummy_csv(path: &str) -> Result<()> {
    let mut file = File::create(path).context("Failed to create dummy CSV")?;

    // Write header
    writeln!(
        file,
        "id,name,email,department,salary,hire_date,status,notes"
    )?;

    let departments = ["Engineering", "Sales", "Marketing", "HR", "Finance", "Operations"];
    let statuses = ["Active", "Inactive", "On Leave", "Terminated"];

    for i in 0..DUMMY_ROWS {
        let dept = departments[i % departments.len()];
        let status = statuses[i % statuses.len()];
        let salary = 50000 + (i * 100) % 100000;

        writeln!(
            file,
            "{},User_{},user_{}@example.com,{},{},2020-{:02}-{:02},{},Notes for user {}",
            i + 1,
            i + 1,
            i + 1,
            dept,
            salary,
            (i % 12) + 1,
            (i % 28) + 1,
            status,
            i + 1
        )?;
    }

    file.flush()?;
    eprintln!("Generated dummy CSV: {}", path);
    Ok(())
}

// ================= Main Application Loop =================

fn main() -> Result<()> {
    // Parse arguments
    let csv_path = parse_args();

    // Load data
    let (df, load_time_ms) = load_data(&csv_path)?;

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app state
    let mut app = App::new(df);
    app.load_time_ms = load_time_ms;

    // Main loop (~60 FPS)
    let result = run_app(&mut terminal, &mut app);

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(err) = result {
        eprintln!("Error: {:?}", err);
        return Err(err);
    }

    Ok(())
}

fn run_app<B: ratatui::backend::Backend>(terminal: &mut Terminal<B>, app: &mut App) -> Result<()> {
    loop {
        // Render UI
        terminal.draw(|f| ui(f, app))?;

        // Poll for events with timeout (enables 60 FPS refresh)
        if event::poll(POLL_TIMEOUT)? {
            let event = event::read()?;
            app.handle_event(event)?;
        }

        // Check quit flag
        if app.should_quit {
            break;
        }
    }

    Ok(())
}
