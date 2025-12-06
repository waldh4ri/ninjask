# Ninjask

**WIP: Terminal CSV Explorer for Large Files**

Ninjask is a work-in-progress terminal-based CSV viewer designed for efficiently exploring large CSV datasets (millions of rows) using SIMD-optimized DataFrame operations (via Polars).

- Fast navigation with virtual scrolling (only visible rows rendered)
- Zero-copy slicing for instant DataFrame operations
- In-terminal UI (TUI) with column selection, sorting, and regex search
- Handles malformed CSVs gracefully

## Compilation Guide

1. Ensure you have [Rust](https://www.rust-lang.org/tools/install) installed (nightly not required).
2. Clone the repository and enter the project directory:
   ```sh
   git clone git@github.com:waldh4ri/ninjask.git
   cd ninjask
   ```
3. Build in release mode for best performance:
   ```sh
   cargo build --release
   ```
4. Run the tool:
   ```sh
   ./target/release/ninjask [OPTIONS]
   ```

## Usage

```sh
./target/release/ninjask [OPTIONS]
```

Options:
- `-f <file.csv>`: Specify CSV file to load (default: data.csv)
- `-h`: Show help

## Status

This tool is experimental and under active development. Performance and features may change.

## Goals
- Toy with SIMD-optimized DataFrame for large CSVs
- Provide a responsive, interactive terminal UI
- Explore Polars and Rust for high-performance data handling

## License
MIT
