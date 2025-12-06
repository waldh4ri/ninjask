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
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use chrono::{DateTime, NaiveDateTime, Utc};
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

/// Sanitize string for display - handles binary data, control characters, and invalid UTF-8
fn sanitize_for_display(s: &str, max_chars: usize) -> String {
    // First pass: clean control characters and normalize whitespace
    let cleaned: String = s
        .chars()
        .map(|c| {
            if c.is_control() && c != '\n' && c != '\t' {
                '�' // Unicode replacement character for control chars
            } else if c.is_whitespace() && c != ' ' && c != '\n' && c != '\t' {
                ' ' // Normalize exotic whitespace to regular space
            } else {
                c
            }
        })
        .collect();
    
    // Second pass: truncate to max length
    truncate_string(&cleaned, max_chars)
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
    /// Time filter setup - select column
    TimeFilterSetup,
    /// Time filter configuration - configure filter options
    TimeFilterConfig,
    /// Value filter selection - choose values to show
    ValueFilterSetup,
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

/// Time unit for relative filtering
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TimeUnit {
    Minutes,
    Hours,
    Days,
    Weeks,
    Months,
}

impl TimeUnit {
    fn label(&self) -> &'static str {
        match self {
            TimeUnit::Minutes => "minutes",
            TimeUnit::Hours => "hours",
            TimeUnit::Days => "days",
            TimeUnit::Weeks => "weeks",
            TimeUnit::Months => "months",
        }
    }

    #[allow(dead_code)]
    fn all_units() -> [TimeUnit; 5] {
        [
            TimeUnit::Minutes,
            TimeUnit::Hours,
            TimeUnit::Days,
            TimeUnit::Weeks,
            TimeUnit::Months,
        ]
    }
}

/// Time filter configuration for datetime columns
#[derive(Debug, Clone)]
struct TimeFilter {
    /// Column name to filter on
    column_name: String,
    /// Filter mode
    mode: TimeFilterMode,
}

/// Value filter for categorical columns
#[derive(Debug, Clone)]
struct ValueFilter {
    /// Column name to filter on
    column_name: String,
    /// Selected values to show (empty = show all)
    selected_values: Vec<String>,
}

/// Time filter mode options
#[derive(Debug, Clone, PartialEq)]
enum TimeFilterMode {
    /// No filter
    None,
    /// After a specific date/time (seconds since epoch)
    After(i64),
    /// Before a specific date/time (seconds since epoch)
    Before(i64),
    /// Between two dates/times (seconds since epoch)
    Range(i64, i64),
    /// Relative: last N days/hours/minutes
    LastN(u32, TimeUnit),
}

impl TimeFilterMode {
    fn label(&self) -> String {
        match self {
            TimeFilterMode::None => "None".to_string(),
            TimeFilterMode::After(ts) => {
                let dt = DateTime::<Utc>::from_timestamp(*ts, 0)
                    .map(|dt| dt.format("%Y-%m-%d").to_string())
                    .unwrap_or_default();
                format!("After {}", dt)
            }
            TimeFilterMode::Before(ts) => {
                let dt = DateTime::<Utc>::from_timestamp(*ts, 0)
                    .map(|dt| dt.format("%Y-%m-%d").to_string())
                    .unwrap_or_default();
                format!("Before {}", dt)
            }
            TimeFilterMode::Range(start, end) => {
                let start_dt = DateTime::<Utc>::from_timestamp(*start, 0)
                    .map(|dt| dt.format("%Y-%m-%d").to_string())
                    .unwrap_or_default();
                let end_dt = DateTime::<Utc>::from_timestamp(*end, 0)
                    .map(|dt| dt.format("%Y-%m-%d").to_string())
                    .unwrap_or_default();
                format!("{} to {}", start_dt, end_dt)
            }
            TimeFilterMode::LastN(n, unit) => {
                format!("Last {} {}", n, unit.label())
            }
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
    /// User manually adjusted width (overrides auto-calculated)
    manual_width: Option<u16>,
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
    /// Active time filters (per column)
    time_filters: Vec<TimeFilter>,
    /// Time filter input state during configuration
    time_filter_input: Input,
    /// Selected datetime column for filtering
    selected_time_column: Option<String>,
    /// Datetime columns in the DataFrame
    datetime_columns: Vec<String>,
    /// Column picker state for time filters
    time_filter_list_state: ListState,
    /// Current time filter mode being configured (1-4 for filter type)
    time_filter_mode_choice: Option<u32>,
    /// Last time filter application time (ms)
    time_filter_ms: u64,
    /// Temporary storage for range filter start timestamp
    range_filter_start: Option<i64>,
    /// Value filters (per column)
    value_filters: Vec<ValueFilter>,
    /// Unique values for current column being filtered
    value_filter_options: Vec<String>,
    /// Selected values in value filter popup (indices)
    value_filter_selections: Vec<bool>,
    /// List state for value filter popup
    value_filter_list_state: ListState,
    /// Column being filtered by value
    value_filter_column: Option<String>,
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
                manual_width: None,
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
            time_filters: Vec::new(),
            time_filter_input: Input::default(),
            selected_time_column: None,
            datetime_columns: Vec::new(),
            time_filter_list_state: ListState::default(),
            time_filter_mode_choice: None,
            time_filter_ms: 0,
            range_filter_start: None,
            value_filters: Vec::new(),
            value_filter_options: Vec::new(),
            value_filter_selections: Vec::new(),
            value_filter_list_state: ListState::default(),
            value_filter_column: None,
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

    /// Resize the selected column
    fn resize_selected_column(&mut self, delta: i16) {
        if let Some(original_idx) = self.visible_column_original_index(self.selected_column) {
            let col = &mut self.columns[original_idx];
            let current_width = col.manual_width.unwrap_or(col.display_width);
            let new_width = (current_width as i16 + delta).clamp(5, 200) as u16;
            col.manual_width = Some(new_width);
            
            self.status_message = format!("Column '{}' width: {}", col.name, new_width);
        }
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

    /// Calculate milliseconds ago for relative time filtering
    fn calculate_ms_ago(&self, n: u32, unit: TimeUnit) -> i64 {
        let ms = match unit {
            TimeUnit::Minutes => 60_000,
            TimeUnit::Hours => 3_600_000,
            TimeUnit::Days => 86_400_000,
            TimeUnit::Weeks => 604_800_000,
            TimeUnit::Months => 2_592_000_000, // 30 days
        };
        (n as i64) * ms
    }

    /// Parse datetime string (supports multiple formats)
    fn parse_datetime(&self, input: &str) -> Option<i64> {
        // Try ISO 8601 formats first
        if let Ok(dt) = DateTime::parse_from_rfc3339(input) {
            return Some(dt.timestamp());
        }
        if let Ok(dt) = DateTime::parse_from_rfc2822(input) {
            return Some(dt.timestamp());
        }

        // Try basic formats: YYYY-MM-DD and YYYY-MM-DDTHH:MM:SS
        if let Ok(date) = chrono::NaiveDate::parse_from_str(input, "%Y-%m-%d") {
            let ndt = date.and_hms_opt(0, 0, 0)?;
            let dt = DateTime::<Utc>::from_naive_utc_and_offset(ndt, Utc);
            return Some(dt.timestamp());
        }

        if let Ok(dt) = NaiveDateTime::parse_from_str(input, "%Y-%m-%dT%H:%M:%S") {
            let utc_dt = DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc);
            return Some(utc_dt.timestamp());
        }

        if let Ok(dt) = NaiveDateTime::parse_from_str(input, "%Y-%m-%d %H:%M:%S") {
            let utc_dt = DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc);
            return Some(utc_dt.timestamp());
        }

        None
    }

    /// Apply all filters (search + time filters) to DataFrame
    fn apply_all_filters(&mut self) -> Result<()> {
        let start = Instant::now();

        let mut df = (*self.original_df).clone().lazy();

        // Apply search filter
        if !self.search_query.is_empty() {
            df = self.apply_search_filter_lazy(df)?;
        }

        // Apply time filters
        for time_filter in &self.time_filters {
            df = self.apply_time_filter_lazy(df, time_filter)?;
        }

        // Apply value filters
        for value_filter in &self.value_filters {
            df = self.apply_value_filter_lazy(df, value_filter)?;
        }

        self.current_df = Arc::new(df.collect()?);
        self.filtered_rows = self.current_df.height();
        self.time_filter_ms = start.elapsed().as_millis() as u64;

        // Re-apply sort if active
        self.apply_current_sort()?;

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

        Ok(())
    }

    /// Apply search filter with lazy evaluation
    fn apply_search_filter_lazy(&self, df: LazyFrame) -> Result<LazyFrame> {
        let pattern = if self.search_query.starts_with("(?") {
            self.search_query.clone()
        } else {
            format!("(?i){}", self.search_query)
        };

        // Validate the regex early
        if let Err(err) = regex::Regex::new(&pattern) {
            return Err(anyhow::anyhow!("Invalid regex: {}", err));
        }

        let visible_cols = self.visible_columns();
        if visible_cols.is_empty() {
            return Ok(df);
        }

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
            Ok(df.filter(expr))
        } else {
            Ok(df)
        }
    }

    /// Apply value filter with lazy evaluation
    fn apply_value_filter_lazy(&self, df: LazyFrame, filter: &ValueFilter) -> Result<LazyFrame> {
        if filter.selected_values.is_empty() {
            return Ok(df);
        }

        // Build filter expression: column value is in selected values
        // Convert to string for comparison to match how we extracted unique values
        let mut filter_expr: Option<Expr> = None;
        
        for value in &filter.selected_values {
            let value_filter = col(&filter.column_name)
                .cast(DataType::String)
                .eq(lit(value.clone()));
            
            filter_expr = Some(match filter_expr {
                Some(expr) => expr.or(value_filter),
                None => value_filter,
            });
        }

        if let Some(expr) = filter_expr {
            Ok(df.filter(expr))
        } else {
            Ok(df)
        }
    }

    /// Apply time filter with lazy evaluation
    fn apply_time_filter_lazy(&self, df: LazyFrame, filter: &TimeFilter) -> Result<LazyFrame> {
        match &filter.mode {
            TimeFilterMode::None => Ok(df),
            TimeFilterMode::After(ts) => {
                // Convert seconds to nanoseconds for comparison with Polars datetime
                let ts_ns = *ts as i64 * 1_000_000_000;
                Ok(df.filter(col(&filter.column_name).gt(lit(ts_ns))))
            }
            TimeFilterMode::Before(ts) => {
                let ts_ns = *ts as i64 * 1_000_000_000;
                Ok(df.filter(col(&filter.column_name).lt(lit(ts_ns))))
            }
            TimeFilterMode::Range(start, end) => {
                // Convert seconds to nanoseconds
                let start_ns = *start as i64 * 1_000_000_000;
                let end_ns = *end as i64 * 1_000_000_000;
                // Range: column >= start AND column <= end (inclusive)
                let start_filter = col(&filter.column_name).gt_eq(lit(start_ns));
                let end_filter = col(&filter.column_name).lt_eq(lit(end_ns));
                Ok(df.filter(start_filter.and(end_filter)))
            }
            TimeFilterMode::LastN(n, unit) => {
                let ms_ago = self.calculate_ms_ago(*n, *unit);
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as i64;
                let cutoff_ms = now - ms_ago;
                // Convert milliseconds to nanoseconds for Polars datetime comparison
                let cutoff_ns = cutoff_ms * 1_000_000;
                Ok(df.filter(col(&filter.column_name).gt(lit(cutoff_ns))))
            }
        }
    }

    /// Apply search filter to DataFrame
    fn apply_search_filter(&mut self) -> Result<()> {
        let start = Instant::now();

        if self.search_query.is_empty() && self.time_filters.is_empty() && self.value_filters.is_empty() {
            // Reset to original (with sort applied if any)
            self.current_df = Arc::clone(&self.original_df);
        } else {
            // Use combined filter function
            return self.apply_all_filters();
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

        // Reset to filtered data (or original if no filters)
        if self.search_query.is_empty() && self.time_filters.is_empty() && self.value_filters.is_empty() {
            self.current_df = Arc::clone(&self.original_df);
        } else {
            self.apply_all_filters()?;
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
            AppMode::TimeFilterSetup => self.handle_time_filter_setup(event)?,
            AppMode::TimeFilterConfig => self.handle_time_filter_config(event)?,
            AppMode::ValueFilterSetup => self.handle_value_filter_setup(event)?,
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
                
                // Column resize with Alt+Arrow keys (must come before regular arrow keys)
                KeyCode::Left if key.modifiers.contains(KeyModifiers::ALT) => {
                    self.resize_selected_column(-2);
                }
                KeyCode::Right if key.modifiers.contains(KeyModifiers::ALT) => {
                    self.resize_selected_column(2);
                }
                
                // Regular horizontal navigation
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

                // Time filter
                KeyCode::Char('t') => {
                    if !self.datetime_columns.is_empty() {
                        self.mode = AppMode::TimeFilterSetup;
                        self.time_filter_list_state.select(Some(0));
                        self.status_message = "Select datetime column (j/k navigate, Enter confirm, Esc cancel)".to_string();
                    } else {
                        self.status_message = "No datetime columns in data".to_string();
                    }
                }

                // Value filter
                KeyCode::Char('v') => {
                    self.start_value_filter()?;
                }

                // Clear all filters
                KeyCode::Esc => {
                    let had_filters = !self.search_query.is_empty() 
                        || !self.time_filters.is_empty() 
                        || !self.value_filters.is_empty();
                    
                    if had_filters {
                        self.search_query.clear();
                        self.search_input = Input::default();
                        self.time_filters.clear();
                        self.value_filters.clear();
                        self.apply_all_filters()?;
                        self.status_message = "All filters cleared".to_string();
                    }
                }

                // Help
                KeyCode::Char('?') => {
                    self.status_message = "Keys: j/k=↑↓ h/l=←→ /=search s=sort c=columns t=time v=value filter Alt+←/→=resize q=quit".to_string();
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
                let raw = format!("{}", val);
                // Sanitize but allow larger display for detail view
                self.cell_detail_content = sanitize_for_display(&raw, 50000);
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

    fn handle_time_filter_setup(&mut self, event: Event) -> Result<()> {
        if let Event::Key(key) = event {
            match key.code {
                KeyCode::Enter => {
                    if let Some(selected_idx) = self.time_filter_list_state.selected() {
                        if selected_idx < self.datetime_columns.len() {
                            self.selected_time_column = Some(self.datetime_columns[selected_idx].clone());
                            self.mode = AppMode::TimeFilterConfig;
                            self.time_filter_mode_choice = None;
                            self.time_filter_input = Input::default();
                            self.status_message = "Select filter type: 1=After, 2=Before, 3=Range, 4=Last N, Esc=Cancel".to_string();
                        }
                    }
                }
                KeyCode::Char('j') | KeyCode::Down => {
                    let current = self.time_filter_list_state.selected().unwrap_or(0);
                    let max = self.datetime_columns.len().saturating_sub(1);
                    if current < max {
                        self.time_filter_list_state.select(Some(current + 1));
                    }
                }
                KeyCode::Char('k') | KeyCode::Up => {
                    let current = self.time_filter_list_state.selected().unwrap_or(0);
                    if current > 0 {
                        self.time_filter_list_state.select(Some(current - 1));
                    }
                }
                KeyCode::Esc => {
                    self.mode = AppMode::Normal;
                    self.status_message = "Time filter cancelled".to_string();
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn handle_time_filter_config(&mut self, event: Event) -> Result<()> {
        if let Event::Key(key) = event {
            match key.code {
                KeyCode::Char('1') if self.time_filter_mode_choice.is_none() => {
                    self.time_filter_mode_choice = Some(1);
                    self.time_filter_input = Input::default();
                    self.status_message = "After timestamp (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)".to_string();
                }
                KeyCode::Char('2') if self.time_filter_mode_choice.is_none() => {
                    self.time_filter_mode_choice = Some(2);
                    self.time_filter_input = Input::default();
                    self.status_message = "Before timestamp (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)".to_string();
                }
                KeyCode::Char('3') if self.time_filter_mode_choice.is_none() => {
                    self.time_filter_mode_choice = Some(3);
                    self.time_filter_input = Input::default();
                    self.status_message = "Start timestamp (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)".to_string();
                }
                KeyCode::Char('4') if self.time_filter_mode_choice.is_none() => {
                    self.time_filter_mode_choice = Some(4);
                    self.time_filter_input = Input::default();
                    self.status_message = "Last N units (e.g., '7 days', '24 hours', '30 minutes')".to_string();
                }
                KeyCode::Enter if self.time_filter_mode_choice.is_some() => {
                    let col_name = self.selected_time_column.clone().unwrap_or_default();
                    let input_str = self.time_filter_input.value().to_string();

                    if input_str.is_empty() {
                        self.status_message = "Input cannot be empty".to_string();
                        return Ok(());
                    }

                    let mode = match self.time_filter_mode_choice {
                        Some(1) => {
                            // After
                            if let Some(ts) = self.parse_datetime(&input_str) {
                                TimeFilterMode::After(ts)
                            } else {
                                self.status_message = "Invalid date format".to_string();
                                return Ok(());
                            }
                        }
                        Some(2) => {
                            // Before
                            if let Some(ts) = self.parse_datetime(&input_str) {
                                TimeFilterMode::Before(ts)
                            } else {
                                self.status_message = "Invalid date format".to_string();
                                return Ok(());
                            }
                        }
                        Some(3) => {
                            // Range - two-step process
                            if self.range_filter_start.is_none() {
                                // First step: get start date
                                if let Some(ts) = self.parse_datetime(&input_str) {
                                    self.range_filter_start = Some(ts);
                                    self.time_filter_input = Input::default();
                                    self.status_message = "End date (format: YYYY-MM-DD)".to_string();
                                    return Ok(()); // Don't apply yet, wait for end date
                                } else {
                                    self.status_message = "Invalid date format".to_string();
                                    return Ok(());
                                }
                            } else {
                                // Second step: get end date
                                if let Some(end_ts) = self.parse_datetime(&input_str) {
                                    let start_ts = self.range_filter_start.unwrap();
                                    if end_ts < start_ts {
                                        self.status_message = "End date must be after start date".to_string();
                                        self.range_filter_start = None;
                                        return Ok(());
                                    }
                                    TimeFilterMode::Range(start_ts, end_ts)
                                } else {
                                    self.status_message = "Invalid date format".to_string();
                                    self.range_filter_start = None;
                                    return Ok(());
                                }
                            }
                        }
                        Some(4) => {
                            // Last N with unit selection: parse "7 days" or "24 hours" etc.
                            let parts: Vec<&str> = input_str.trim().split_whitespace().collect();
                            
                            if parts.is_empty() {
                                self.status_message = "Format: <number> <unit> (e.g., '7 days', '24 hours')".to_string();
                                return Ok(());
                            }
                            
                            // Parse number from first part
                            let n = if let Ok(num) = parts[0].parse::<u32>() {
                                num
                            } else {
                                self.status_message = "Invalid number".to_string();
                                return Ok(());
                            };
                            
                            // Parse unit from second part, or default to days
                            let unit = if parts.len() > 1 {
                                let unit_str = parts[1].to_lowercase();
                                if unit_str.starts_with("minute") || unit_str == "m" {
                                    TimeUnit::Minutes
                                } else if unit_str.starts_with("hour") || unit_str == "h" {
                                    TimeUnit::Hours
                                } else if unit_str.starts_with("day") || unit_str == "d" {
                                    TimeUnit::Days
                                } else if unit_str.starts_with("week") || unit_str == "w" {
                                    TimeUnit::Weeks
                                } else if unit_str.starts_with("month") || unit_str == "mo" {
                                    TimeUnit::Months
                                } else {
                                    self.status_message = "Unknown unit. Use: minutes, hours, days, weeks, months".to_string();
                                    return Ok(());
                                }
                            } else {
                                TimeUnit::Days // Default to days if no unit specified
                            };
                            
                            TimeFilterMode::LastN(n, unit)
                        }
                        _ => TimeFilterMode::None,
                    };

                    // Add or update the filter
                    if let Some(col_name_str) = &self.selected_time_column {
                        // Remove existing filter for this column if any
                        self.time_filters.retain(|f| f.column_name != *col_name_str);
                        // Add new filter
                        self.time_filters.push(TimeFilter {
                            column_name: col_name_str.clone(),
                            mode: mode.clone(),
                        });

                        // Apply filters
                        self.apply_all_filters()?;

                        self.mode = AppMode::Normal;
                        self.range_filter_start = None; // Reset range state
                        self.status_message = format!(
                            "Time filter applied: {} {}",
                            col_name,
                            mode.label()
                        );
                    }
                }
                KeyCode::Esc => {
                    if self.time_filter_mode_choice.is_some() {
                        self.time_filter_mode_choice = None;
                        self.time_filter_input = Input::default();
                        self.range_filter_start = None; // Reset range state
                        self.status_message = "Select filter type: 1=After, 2=Before, 3=Range, 4=Last N, Esc=Cancel".to_string();
                    } else {
                        self.mode = AppMode::Normal;
                        self.range_filter_start = None; // Reset range state
                        self.status_message = "Time filter cancelled".to_string();
                    }
                }
                _ => {
                    if self.time_filter_mode_choice.is_some() {
                        self.time_filter_input.handle_event(&event);
                    }
                }
            }
        }
        Ok(())
    }

    /// Start value filter for selected column
    fn start_value_filter(&mut self) -> Result<()> {
        const MAX_UNIQUE_VALUES: usize = 100;

        // Get the selected column
        let visible_cols: Vec<String> = self
            .columns
            .iter()
            .filter(|c| c.visible)
            .map(|c| c.name.clone())
            .collect();

        if visible_cols.is_empty() {
            self.status_message = "No columns available".to_string();
            return Ok(());
        }

        if self.selected_column >= visible_cols.len() {
            self.status_message = "Invalid column selection".to_string();
            return Ok(());
        }

        let col_name = visible_cols[self.selected_column].clone();

        // Get unique values from the ORIGINAL dataframe (not filtered)
        // This ensures all possible values are available for selection
        if let Ok(col) = self.original_df.column(&col_name) {
            let unique_result = col.unique();
            
            match unique_result {
                Ok(unique_series) => {
                    let n_unique = unique_series.len();

                    if n_unique > MAX_UNIQUE_VALUES {
                        self.status_message = format!(
                            "Column '{}' has {} unique values (max {}). Use search (/) instead.",
                            col_name, n_unique, MAX_UNIQUE_VALUES
                        );
                        return Ok(());
                    }

                    if n_unique == 0 {
                        self.status_message = format!("Column '{}' has no values", col_name);
                        return Ok(());
                    }

                    // Convert unique values to strings and sort
                    // Extract actual string representation that matches the data
                    let mut values: Vec<String> = (0..unique_series.len())
                        .filter_map(|i| {
                            unique_series.get(i).ok().and_then(|v| {
                                // Use get_str() for string types, format for others
                                match v {
                                    polars::prelude::AnyValue::String(s) => Some(s.to_string()),
                                    polars::prelude::AnyValue::StringOwned(s) => Some(s.to_string()),
                                    _ => Some(format!("{}", v)),
                                }
                            })
                        })
                        .collect();
                    values.sort();

                    // Initialize selections - check if there's an existing filter
                    let existing_filter = self.value_filters
                        .iter()
                        .find(|f| f.column_name == col_name);

                    let selections = if let Some(filter) = existing_filter {
                        // Pre-select values that are in the existing filter
                        values
                            .iter()
                            .map(|v| filter.selected_values.contains(v))
                            .collect()
                    } else {
                        // All selected by default
                        vec![true; values.len()]
                    };

                    self.value_filter_options = values;
                    self.value_filter_selections = selections;
                    self.value_filter_column = Some(col_name.clone());
                    self.value_filter_list_state.select(Some(0));
                    self.mode = AppMode::ValueFilterSetup;
                    self.status_message = format!(
                        "Column '{}': {} unique values. Space=toggle, a=all, n=none, Enter=apply",
                        col_name, n_unique
                    );
                }
                Err(e) => {
                    self.status_message = format!("Error getting unique values: {}", e);
                }
            }
        } else {
            self.status_message = format!("Column '{}' not found", col_name);
        }

        Ok(())
    }

    fn handle_value_filter_setup(&mut self, event: Event) -> Result<()> {
        if let Event::Key(key) = event {
            match key.code {
                KeyCode::Enter => {
                    // Apply the filter
                    if let Some(col_name) = &self.value_filter_column.clone() {
                        let selected_values: Vec<String> = self
                            .value_filter_options
                            .iter()
                            .enumerate()
                            .filter_map(|(i, v)| {
                                if self.value_filter_selections.get(i) == Some(&true) {
                                    Some(v.clone())
                                } else {
                                    None
                                }
                            })
                            .collect();

                        // Remove existing filter for this column
                        self.value_filters.retain(|f| f.column_name != *col_name);

                        // Add new filter only if not all values are selected
                        let all_selected = self.value_filter_selections.iter().all(|&x| x);
                        if !all_selected && !selected_values.is_empty() {
                            self.value_filters.push(ValueFilter {
                                column_name: col_name.clone(),
                                selected_values: selected_values.clone(),
                            });
                        }

                        // Apply all filters
                        self.apply_all_filters()?;

                        let filter_msg = if all_selected {
                            format!("Value filter cleared for '{}'", col_name)
                        } else {
                            format!(
                                "Value filter applied to '{}': {} values selected",
                                col_name,
                                selected_values.len()
                            )
                        };
                        self.status_message = filter_msg;
                    }

                    self.mode = AppMode::Normal;
                }
                KeyCode::Esc => {
                    self.mode = AppMode::Normal;
                    self.status_message = "Value filter cancelled".to_string();
                }
                KeyCode::Char(' ') => {
                    // Toggle selected value
                    if let Some(idx) = self.value_filter_list_state.selected() {
                        if idx < self.value_filter_selections.len() {
                            self.value_filter_selections[idx] = !self.value_filter_selections[idx];
                        }
                    }
                }
                KeyCode::Char('a') => {
                    // Select all
                    self.value_filter_selections = vec![true; self.value_filter_options.len()];
                    self.status_message = "All values selected".to_string();
                }
                KeyCode::Char('n') => {
                    // Select none - but keep at least one selected
                    let current_idx = self.value_filter_list_state.selected().unwrap_or(0);
                    self.value_filter_selections = vec![false; self.value_filter_options.len()];
                    if current_idx < self.value_filter_selections.len() {
                        self.value_filter_selections[current_idx] = true;
                    }
                    self.status_message = "All values deselected except current".to_string();
                }
                KeyCode::Char('j') | KeyCode::Down => {
                    let current = self.value_filter_list_state.selected().unwrap_or(0);
                    let max = self.value_filter_options.len().saturating_sub(1);
                    if current < max {
                        self.value_filter_list_state.select(Some(current + 1));
                    }
                }
                KeyCode::Char('k') | KeyCode::Up => {
                    let current = self.value_filter_list_state.selected().unwrap_or(0);
                    if current > 0 {
                        self.value_filter_list_state.select(Some(current - 1));
                    }
                }
                _ => {}
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

    // Render time filter popups if active
    if app.mode == AppMode::TimeFilterSetup {
        render_time_filter_setup_popup(frame, app);
    }

    if app.mode == AppMode::TimeFilterConfig {
        render_time_filter_config_popup(frame, app);
    }

    // Render value filter popup if active
    if app.mode == AppMode::ValueFilterSetup {
        render_value_filter_popup(frame, app);
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

    // Build filter info string
    let filter_info = if app.search_query.is_empty() && app.time_filters.is_empty() && app.value_filters.is_empty() {
        "No filter".to_string()
    } else {
        let mut parts = Vec::new();
        if !app.search_query.is_empty() {
            parts.push(format!("Search: {}", app.search_query));
        }
        if !app.time_filters.is_empty() {
            for tf in &app.time_filters {
                parts.push(format!("{}:{}", tf.column_name, tf.mode.label()));
            }
        }
        if !app.value_filters.is_empty() {
            for vf in &app.value_filters {
                parts.push(format!("{}:{} vals", vf.column_name, vf.selected_values.len()));
            }
        }
        parts.join(" | ")
    };

    let header_text = format!(
        " 🪲 Ninjask {} │ Rows: {} / {} │ Cols: {} │ {}{}",
        env!("CARGO_PKG_VERSION"),
        app.filtered_rows,
        app.total_rows,
        app.columns.iter().filter(|c| c.visible).count(),
        filter_info,
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

    // Use cached auto-calculated column widths (or manual override)
    let widths: Vec<Constraint> = app
        .columns
        .iter()
        .filter(|c| c.visible)
        .map(|c| Constraint::Length(c.manual_width.unwrap_or(c.display_width)))
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
                        // Get the width for this column (manual or auto)
                        let col_width = app
                            .columns
                            .iter()
                            .filter(|c| c.visible)
                            .nth(col_idx)
                            .map(|c| c.manual_width.unwrap_or(c.display_width) as usize)
                            .unwrap_or(10);

                        // Safe value extraction with sanitization
                        let display_value = match sliced
                            .get_columns()
                            .get(col_idx)
                            .and_then(|s| s.get(row_idx).ok())
                        {
                            Some(val) => {
                                let mut raw = format!("{}", val);
                                
                                // Strip quotes from string values
                                if raw.starts_with('"') && raw.ends_with('"') && raw.len() >= 2 {
                                    raw = raw[1..raw.len()-1].to_string();
                                }
                                
                                // Sanitize and truncate for display
                                sanitize_for_display(&raw, col_width.saturating_sub(1))
                            }
                            None => "<error>".to_string(),
                        };

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
                .style(Style::default().fg(Color::Yellow).bold())
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(Color::Yellow)),
                );
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
                    "/ search  s sort  c columns  t time-filter  v value-filter  q quit",
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
            ListItem::new(format!("  {} {}{}  ", checkbox, col.name, move_indicator)).style(style)
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
                .title_position(ratatui::widgets::block::Position::Bottom)
                .padding(ratatui::widgets::Padding::new(2, 2, 1, 1)),
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

fn render_time_filter_setup_popup(frame: &mut Frame, app: &mut App) {
    // Create centered popup
    let area = centered_rect(60, 60, frame.area());
    frame.render_widget(Clear, area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(5),
            Constraint::Length(2),
        ])
        .split(area);

    // Header
    let header = Paragraph::new(" Select Datetime Column to Filter ")
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::Cyan).bold())
        .block(
            Block::default()
                .borders(Borders::TOP | Borders::LEFT | Borders::RIGHT)
                .border_style(Style::default().fg(Color::Cyan))
        );
    frame.render_widget(header, chunks[0]);

    // List of datetime columns
    let items: Vec<ListItem> = app
        .datetime_columns
        .iter()
        .map(|col| ListItem::new(format!("    {}  ", col)))
        .collect();

    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::LEFT | Borders::RIGHT)
                .border_style(Style::default().fg(Color::Cyan))
        )
        .highlight_style(
            Style::default()
                .bg(Color::Cyan)
                .fg(Color::Black)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▶ ");

    frame.render_stateful_widget(list, chunks[1], &mut app.time_filter_list_state.clone());

    // Footer
    let footer = Paragraph::new("Enter=confirm, Esc=cancel")
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::Gray))
        .block(
            Block::default()
                .borders(Borders::BOTTOM | Borders::LEFT | Borders::RIGHT)
                .border_style(Style::default().fg(Color::Cyan))
        );
    frame.render_widget(footer, chunks[2]);
}

fn render_value_filter_popup(frame: &mut Frame, app: &mut App) {
    // Create centered popup
    let area = centered_rect(60, 70, frame.area());
    frame.render_widget(Clear, area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(5),
            Constraint::Length(3),
        ])
        .split(area);

    // Header with column name and count
    let default_col = "Unknown".to_string();
    let col_name = app.value_filter_column.as_ref().unwrap_or(&default_col);
    let selected_count = app.value_filter_selections.iter().filter(|&&x| x).count();
    let header = Paragraph::new(format!(
        " Filter '{}' ({}/{} selected) ",
        col_name,
        selected_count,
        app.value_filter_options.len()
    ))
    .alignment(Alignment::Center)
    .style(Style::default().fg(Color::Cyan).bold())
    .block(
        Block::default()
            .borders(Borders::TOP | Borders::LEFT | Borders::RIGHT)
            .border_style(Style::default().fg(Color::Cyan)),
    );
    frame.render_widget(header, chunks[0]);

    // List of values with checkboxes
    let items: Vec<ListItem> = app
        .value_filter_options
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            let checkbox = if app.value_filter_selections.get(idx) == Some(&true) {
                "[✓]"
            } else {
                "[ ]"
            };
            let style = if app.value_filter_selections.get(idx) == Some(&true) {
                Style::default().fg(Color::Green)
            } else {
                Style::default().fg(Color::DarkGray)
            };
            // Truncate long values
            let display_value = truncate_string(value, 50);
            ListItem::new(format!("   {} {}  ", checkbox, display_value)).style(style)
        })
        .collect();

    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::LEFT | Borders::RIGHT)
                .border_style(Style::default().fg(Color::Cyan)),
        )
        .highlight_style(
            Style::default()
                .bg(Color::Cyan)
                .fg(Color::Black)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▶ ");

    frame.render_stateful_widget(list, chunks[1], &mut app.value_filter_list_state.clone());

    // Footer with instructions
    let footer = Paragraph::new("Space=toggle, a=all, n=none, Enter=apply, Esc=cancel")
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::Gray))
        .block(
            Block::default()
                .borders(Borders::BOTTOM | Borders::LEFT | Borders::RIGHT)
                .border_style(Style::default().fg(Color::Cyan)),
        );
    frame.render_widget(footer, chunks[2]);
}

fn render_time_filter_config_popup(frame: &mut Frame, app: &mut App) {
    // Create centered popup
    let area = centered_rect(70, 60, frame.area());
    frame.render_widget(Clear, area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(5),
            Constraint::Length(3),
        ])
        .split(area);

    // Header with column name
    let unknown_col = "Unknown".to_string();
    let col_name = app.selected_time_column.as_ref().unwrap_or(&unknown_col);
    let header = Paragraph::new(format!(" Filter: {} ", col_name))
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::Cyan).bold())
        .block(
            Block::default()
                .borders(Borders::TOP | Borders::LEFT | Borders::RIGHT)
                .border_style(Style::default().fg(Color::Cyan))
        );
    frame.render_widget(header, chunks[0]);

    // Content area
    if app.time_filter_mode_choice.is_none() {
        // Show filter type selection
        let options = vec![
            "  1: After  - Show rows after a specific date/time",
            "  2: Before - Show rows before a specific date/time",
            "  3: Range  - Show rows between two dates/times",
            "  4: Last N - Show rows from last N minutes/hours/days/weeks",
        ];
        let text = format!("\n{}\n", options.join("\n"));
        let content = Paragraph::new(text)
            .style(Style::default().fg(Color::White))
            .wrap(ratatui::widgets::Wrap { trim: false })
            .block(
                Block::default()
                    .borders(Borders::LEFT | Borders::RIGHT)
                    .border_style(Style::default().fg(Color::Cyan))
                    .padding(ratatui::widgets::Padding::new(2, 2, 0, 0)),
            );
        frame.render_widget(content, chunks[1]);

        let footer = Paragraph::new("Press 1-4 to select, Esc=cancel")
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::Gray))
            .block(
                Block::default()
                    .borders(Borders::BOTTOM | Borders::LEFT | Borders::RIGHT)
                    .border_style(Style::default().fg(Color::Cyan))
            );
        frame.render_widget(footer, chunks[2]);
    } else {
        // Show input field for the selected filter type
        let input_label = match app.time_filter_mode_choice {
            Some(1) => "After timestamp (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS):",
            Some(2) => "Before timestamp (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS):",
            Some(3) => {
                if app.range_filter_start.is_none() {
                    "Start timestamp (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS):"
                } else {
                    "End timestamp (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS):"
                }
            }
            Some(4) => "Last N (e.g., '7 days', '24 hours', '30 minutes'):",
            _ => "Input:",
        };

        let content = Paragraph::new(format!("\n{}\n\n{}", input_label, app.time_filter_input.value()))
            .style(Style::default().fg(Color::White))
            .block(
                Block::default()
                    .borders(Borders::LEFT | Borders::RIGHT)
                    .border_style(Style::default().fg(Color::Yellow))
                    .padding(ratatui::widgets::Padding::new(2, 2, 0, 0))
            );
        frame.render_widget(content, chunks[1]);

        // Show cursor (adjusted for padding: left=2, top=0 + 1 newline before label = 1)
        let input_line = 4; // 1 (top padding newline) + 1 (label line) + 2 (blank lines after label)
        frame.set_cursor_position((
            chunks[1].x + app.time_filter_input.visual_cursor() as u16 + 3, // +1 border +2 left padding
            chunks[1].y + input_line as u16,
        ));

        let footer = Paragraph::new("Enter=apply, Esc=back")
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::Gray))
            .block(
                Block::default()
                    .borders(Borders::BOTTOM | Borders::LEFT | Borders::RIGHT)
                    .border_style(Style::default().fg(Color::Yellow))
            );
        frame.render_widget(footer, chunks[2]);
    }
}

// ================= Data Loading =================

/// Load CSV file or generate dummy data if missing
fn load_data(path: &str) -> Result<(DataFrame, u64, Vec<String>)> {
    let start = Instant::now();

    if Path::new(path).exists() {
        eprintln!("Loading CSV: {}", path);

        // Load CSV with error handling for malformed lines
        let parse_options = CsvParseOptions::default()
            .with_truncate_ragged_lines(true) // Handle lines with wrong number of fields
            .with_encoding(CsvEncoding::LossyUtf8); // Handle invalid UTF-8 gracefully

        let mut df = CsvReadOptions::default()
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

        // Parse datetime columns from strings
        df = parse_datetime_columns(df)?;

        // Detect datetime columns (after parsing)
        let datetime_columns = detect_datetime_columns(&df);

        // Log detected schema
        eprintln!(
            "Loaded {} rows x {} columns",
            df.height(),
            df.width()
        );
        for col in df.get_columns() {
            eprintln!("  - {}: {:?}", col.name(), col.dtype());
        }

        if !datetime_columns.is_empty() {
            eprintln!("Datetime columns: {}", datetime_columns.join(", "));
        }

        let load_time = start.elapsed().as_millis() as u64;
        Ok((df, load_time, datetime_columns))
    } else {
        eprintln!("CSV not found, generating {} dummy rows...", DUMMY_ROWS);
        generate_dummy_csv(path)?;
        load_data(path)
    }
}

/// Parse string columns that look like datetime into proper Datetime type
fn parse_datetime_columns(df: DataFrame) -> Result<DataFrame> {
    // First, scan the dataframe to identify datetime columns
    let mut datetime_cols = Vec::new();
    let mut date_cols = Vec::new();
    
    for col_name in df.get_column_names() {
        if let Ok(column) = df.column(col_name) {
            if matches!(column.dtype(), DataType::String) {
                // Sample first non-null value to check if it looks like a datetime
                if let Some(sample) = column.str()
                    .ok()
                    .and_then(|s| s.into_iter().find_map(|v| v)) 
                {
                    if is_datetime_string(sample) {
                        datetime_cols.push(col_name.to_string());
                    } else if is_date_string(sample) {
                        date_cols.push(col_name.to_string());
                    }
                }
            }
        }
    }
    
    // Now convert to lazy and apply transformations
    let mut lazy_df = df.lazy();
    
    for col_name in datetime_cols {
        eprintln!("  Parsing '{}' as datetime column", col_name);
        lazy_df = lazy_df.with_column(
            col(&col_name)
                .str()
                .to_datetime(
                    Some(polars::prelude::TimeUnit::Milliseconds),
                    None,
                    StrptimeOptions::default(),
                    lit("raise"),
                )
                .cast(DataType::Datetime(polars::prelude::TimeUnit::Nanoseconds, None))
                .alias(&col_name)
        );
    }
    
    for col_name in date_cols {
        eprintln!("  Parsing '{}' as date column", col_name);
        lazy_df = lazy_df.with_column(
            col(&col_name)
                .str()
                .to_date(StrptimeOptions::default())
                .alias(&col_name)
        );
    }
    
    Ok(lazy_df.collect()?)
}

/// Check if a string looks like a datetime (YYYY-MM-DD with optional time)
fn is_datetime_string(s: &str) -> bool {
    // Pattern: YYYY-MM-DDTHH:MM:SS or YYYY-MM-DD HH:MM:SS
    let has_time_separator = s.contains('T') || (s.contains(' ') && s.contains(':'));
    let has_date_pattern = s.len() >= 10 && s.chars().nth(4) == Some('-') && s.chars().nth(7) == Some('-');
    has_date_pattern && has_time_separator
}

/// Check if a string looks like a date (YYYY-MM-DD only)
fn is_date_string(s: &str) -> bool {
    // Pattern: YYYY-MM-DD (exactly 10 chars)
    s.len() == 10 && s.chars().nth(4) == Some('-') && s.chars().nth(7) == Some('-')
}

/// Detect datetime columns in a DataFrame
fn detect_datetime_columns(df: &DataFrame) -> Vec<String> {
    df.get_columns()
        .iter()
        .filter_map(|col| {
            match col.dtype() {
                DataType::Date | DataType::Datetime(_, _) => Some(col.name().to_string()),
                _ => None,
            }
        })
        .collect()
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
    let (df, load_time_ms, datetime_columns) = load_data(&csv_path)?;

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app state
    let mut app = App::new(df);
    app.load_time_ms = load_time_ms;
    app.datetime_columns = datetime_columns;

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
