
# <img src="https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/291.png" height="35" align="top"/> Ninjask

**WIP: Terminal CSV Explorer for Large Files**

Ninjask is a work-in-progress terminal-based CSV viewer designed for efficiently exploring large CSV datasets (millions of rows) using [Polars](https://pola.rs/) for DataFrame operations. 

- Fast navigation with virtual scrolling (only visible rows rendered)
- Zero-copy slicing for instant DataFrame operations
- In-terminal UI (TUI) with column selection, sorting, and regex search
- Handles malformed CSVs gracefully

Ninjask currently loads the entire DataFrame in memory using Polars' eager API. This can be memory exhausting for very large CSVs (buy more ram lol). Operations like filtering and searching are done through the Lazy API to avoid unnecessary intermediate steps with multiples filters and searches enabled.

## Compilation Guide

1. Ensure you have [Rust](https://www.rust-lang.org/tools/install) installed (nightly not required).
2. Clone the repository and enter the project directory:
   ```sh
   git clone git@github.com:waldh4ri/ninjask.git
   ```
   ```sh
   cd ninjask
   ```
3. Build in release mode for best performance:
   ```sh
   cargo build --release
   ```

## Usage

```sh
./target/release/ninjask -f data.csv
```
You can generate a dummy 100k lines data.csv like this:
```sh
./target/release/ninjask --test
```
### Options
- `-f, --file <PATH>`: CSV file to load (default: data.csv)
- `-d, --delimiter <SEP>`: Delimiter character (auto-detected if not specified)
  - Examples: `,`, `;`, `tab`, `|`
- `--test`: Generate dummy data.csv file if it doesn't exist
- `--low-memory`: Use low-memory mode for large CSV files (streaming approach)
- `--no-header`: Treat first row as data (no header row)
- `-h, --help`: Show this help message

### Keybindings
- `j/k` or `↑/↓`: Navigate rows
- `h/l` or `←/→`: Navigate columns
- `/`: Search data (regex)
- `s`: Sort by selected column
- `c`: Column visibility picker
- `t`: Time filter
- `v`: Value filter
- `q`: Quit

## Status

This tool is experimental and under active development. Performance and features may change.

## Goals
- Provide a responsive, interactive terminal UI to quickly get a glance at a CSV
- Explore Polars and Rust for high-performance data handling

## License
MIT
