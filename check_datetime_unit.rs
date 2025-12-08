use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .with_parse_options(
            CsvParseOptions::default()
                .with_try_parse_dates(true)
                .with_eol_char(b'\n')
        )
        .try_into_reader_with_file_path(Some("test_check_datetime.csv".into()))?
        .finish()?;

    println!("DataFrame schema:");
    for col in df.get_columns() {
        println!("  {}: {:?}", col.name(), col.dtype());
    }

    println!("\nActual values:");
    println!("{:?}", df);

    // Get the timestamp column
    if let Ok(ts_col) = df.column("timestamp") {
        if let Ok(ts_series) = ts_col.datetime() {
            println!("\nTimestamp values (raw):");
            for i in 0..ts_series.len() {
                if let Some(val) = ts_series.get(i) {
                    println!("  Row {}: {}", i, val);
                }
            }
        }
    }

    Ok(())
}
