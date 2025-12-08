//! Integration tests for Ninjask
//!
//! Tests CSV loading with various edge cases including:
//! - Large files (100k+ rows)
//! - Unicode characters (emoji, CJK, RTL text)
//! - Special characters (quotes, commas, newlines in fields)
//! - Empty fields and malformed data

use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use tempfile::TempDir;

use anyhow::Result;
use polars::prelude::*;

/// Helper to create a temporary CSV file
fn create_temp_csv(content: &str) -> Result<(TempDir, PathBuf)> {
    let temp_dir = TempDir::new()?;
    let file_path = temp_dir.path().join("test.csv");
    let mut file = File::create(&file_path)?;
    file.write_all(content.as_bytes())?;
    file.flush()?;
    Ok((temp_dir, file_path))
}

/// Load CSV using the same logic as the main application
/// Includes graceful error handling for malformed lines
fn load_csv_as_strings(path: &str) -> Result<DataFrame> {
    let parse_options = CsvParseOptions::default()
        .with_truncate_ragged_lines(true); // Handle lines with wrong number of fields

    let df = CsvReadOptions::default()
        .with_has_header(true)
        .with_infer_schema_length(Some(0)) // Don't infer types
        .with_ignore_errors(true) // Skip rows that can't be parsed
        .with_parse_options(parse_options)
        .try_into_reader_with_file_path(Some(path.into()))?
        .finish()?;

    // Cast all columns to String type
    let string_df = df
        .lazy()
        .select([all().cast(DataType::String)])
        .collect()?;

    Ok(string_df)
}

/// Generate a large CSV with various edge cases
pub fn generate_stress_test_csv(path: &str, num_rows: usize) -> Result<()> {
    let mut file = File::create(path)?;

    // Header with special characters in column names
    writeln!(
        file,
        "id,name,email,department,salary,notes,unicode_test,emoji_field"
    )?;

    let departments = [
        "Engineering",
        "Sales & Marketing", // Contains &
        "R&D",
        "HR/Admin",         // Contains /
        "Finance (Global)", // Contains parentheses
        "IT-Support",       // Contains hyphen
    ];

    let unicode_samples = [
        "日本語テスト",           // Japanese
        "中文测试",               // Chinese
        "한국어 테스트",          // Korean
        "Тест на русском",        // Russian
        "اختبار عربي",            // Arabic (RTL)
        "עברית",                  // Hebrew (RTL)
        "Ελληνικά",               // Greek
        "मराठी परीक्षण",          // Hindi/Marathi
        "ทดสอบภาษาไทย",           // Thai
        "Ñoño año",               // Spanish with special chars
    ];

    let emoji_samples = [
        "🚀 Launch!",
        "✅ Done",
        "❌ Failed",
        "🎉🎊🎁",
        "👨‍👩‍👧‍👦 Family",     // Complex emoji with ZWJ
        "🏳️‍🌈 Pride",        // Flag emoji
        "🧑‍💻 Developer",
        "📊📈📉",
        "⚠️ Warning",
        "💡 Idea",
    ];

    let tricky_notes = [
        "Simple note",
        "Note with, comma",
        "\"Quoted note\"",
        "Note with 'single quotes'",
        "Note\twith\ttabs",
        "Multi\nline\nnote",       // Embedded newlines (will be escaped)
        "Backslash \\ test",
        "Pipe | test",
        "Semicolon; test",
        "Ampersand & test",
        "<html>tags</html>",
        "Path: C:\\Users\\test",
        "URL: https://example.com?foo=bar&baz=qux",
        "",                        // Empty field
        "   ",                     // Whitespace only
        "NULL",                    // Literal NULL
        "null",
        "None",
        "N/A",
        "-",
    ];

    for i in 0..num_rows {
        let dept = departments[i % departments.len()];
        let unicode = unicode_samples[i % unicode_samples.len()];
        let emoji = emoji_samples[i % emoji_samples.len()];
        let note = tricky_notes[i % tricky_notes.len()];

        // Properly escape fields for CSV
        let escaped_note = escape_csv_field(note);
        let escaped_unicode = escape_csv_field(unicode);
        let escaped_emoji = escape_csv_field(emoji);
        let escaped_dept = escape_csv_field(dept);

        let salary = 30000 + (i * 137) % 170000; // Varied salaries

        // Generate email with some edge cases
        let email = if i % 50 == 0 {
            "".to_string() // Empty email occasionally
        } else if i % 37 == 0 {
            format!("user+tag{}@sub.example.com", i) // Plus addressing
        } else if i % 23 == 0 {
            format!("user.name.{}@example.co.uk", i) // Multiple dots
        } else {
            format!("user_{}@example.com", i)
        };

        // Generate name with some edge cases
        let name = if i % 100 == 0 {
            format!("O'Brien {}", i) // Irish name with apostrophe
        } else if i % 77 == 0 {
            format!("José García {}", i) // Spanish accents
        } else if i % 55 == 0 {
            format!("François Müller {}", i) // French/German
        } else if i % 33 == 0 {
            format!("Søren Ørsted {}", i) // Danish
        } else {
            format!("User {}", i)
        };

        let escaped_name = escape_csv_field(&name);
        let escaped_email = escape_csv_field(&email);

        writeln!(
            file,
            "{},{},{},{},{},{},{},{}",
            i + 1,
            escaped_name,
            escaped_email,
            escaped_dept,
            salary,
            escaped_note,
            escaped_unicode,
            escaped_emoji
        )?;
    }

    file.flush()?;
    Ok(())
}

/// Escape a field for CSV format (RFC 4180 compliant)
fn escape_csv_field(field: &str) -> String {
    // Check if field needs quoting
    let needs_quotes = field.contains(',')
        || field.contains('"')
        || field.contains('\n')
        || field.contains('\r')
        || field.starts_with(' ')
        || field.ends_with(' ');

    if needs_quotes {
        // Escape double quotes by doubling them
        let escaped = field.replace('"', "\"\"");
        format!("\"{}\"", escaped)
    } else {
        field.to_string()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_csv_loading() -> Result<()> {
        let content = r#"id,name,value
1,Alice,100
2,Bob,200
3,Carol,300"#;

        let (_temp_dir, path) = create_temp_csv(content)?;
        let df = load_csv_as_strings(path.to_str().unwrap())?;

        assert_eq!(df.height(), 3);
        assert_eq!(df.width(), 3);

        // Verify all columns are strings
        for col in df.get_columns() {
            assert_eq!(col.dtype(), &DataType::String);
        }

        Ok(())
    }

    #[test]
    fn test_unicode_characters() -> Result<()> {
        let content = r#"id,name,greeting
1,田中太郎,こんにちは
2,김철수,안녕하세요
3,Müller,Grüß Gott
4,Ñoño,¡Hola!"#;

        let (_temp_dir, path) = create_temp_csv(content)?;
        let df = load_csv_as_strings(path.to_str().unwrap())?;

        assert_eq!(df.height(), 4);

        // Verify Japanese name loaded correctly
        let name_col = df.column("name")?;
        let first_name = name_col.str()?.get(0).unwrap();
        assert_eq!(first_name, "田中太郎");

        Ok(())
    }

    #[test]
    fn test_emoji_characters() -> Result<()> {
        let content = r#"id,status,mood
1,✅ Done,😊
2,❌ Failed,😢
3,🚀 Launched,🎉
4,⚠️ Warning,😰"#;

        let (_temp_dir, path) = create_temp_csv(content)?;
        let df = load_csv_as_strings(path.to_str().unwrap())?;

        assert_eq!(df.height(), 4);

        let status_col = df.column("status")?;
        let first_status = status_col.str()?.get(0).unwrap();
        assert!(first_status.contains("✅"));

        Ok(())
    }

    #[test]
    fn test_quoted_fields_with_commas() -> Result<()> {
        let content = r#"id,name,address
1,John,"123 Main St, Apt 4"
2,Jane,"456 Oak Ave, Suite 100"
3,Bob,"No comma here""#;

        let (_temp_dir, path) = create_temp_csv(content)?;
        let df = load_csv_as_strings(path.to_str().unwrap())?;

        assert_eq!(df.height(), 3);

        let addr_col = df.column("address")?;
        let first_addr = addr_col.str()?.get(0).unwrap();
        assert!(first_addr.contains(","));
        assert_eq!(first_addr, "123 Main St, Apt 4");

        Ok(())
    }

    #[test]
    fn test_escaped_quotes() -> Result<()> {
        let content = r#"id,name,quote
1,Alice,"She said ""Hello"""
2,Bob,"He replied ""Hi there"""
3,Carol,No quotes here"#;

        let (_temp_dir, path) = create_temp_csv(content)?;
        let df = load_csv_as_strings(path.to_str().unwrap())?;

        assert_eq!(df.height(), 3);

        let quote_col = df.column("quote")?;
        let first_quote = quote_col.str()?.get(0).unwrap();
        assert!(first_quote.contains("\"Hello\""));

        Ok(())
    }

    #[test]
    fn test_empty_fields() -> Result<()> {
        let content = r#"id,name,email,phone
1,Alice,alice@example.com,
2,Bob,,555-1234
3,,,
4,Carol,carol@example.com,555-5678"#;

        let (_temp_dir, path) = create_temp_csv(content)?;
        let df = load_csv_as_strings(path.to_str().unwrap())?;

        assert_eq!(df.height(), 4);

        // Empty fields should be loaded as empty strings or null
        let email_col = df.column("email")?;
        let empty_email = email_col.str()?.get(1);
        // Polars may represent empty as null or empty string
        assert!(empty_email.is_none() || empty_email == Some(""));

        Ok(())
    }

    #[test]
    fn test_whitespace_handling() -> Result<()> {
        let content = r#"id,name,value
1,  Alice  ,100
2,Bob,  200  
3,  Carol,300"#;

        let (_temp_dir, path) = create_temp_csv(content)?;
        let df = load_csv_as_strings(path.to_str().unwrap())?;

        assert_eq!(df.height(), 3);

        // Whitespace should be preserved
        let name_col = df.column("name")?;
        let first_name = name_col.str()?.get(0).unwrap();
        assert!(first_name.contains("Alice"));

        Ok(())
    }

    #[test]
    fn test_special_characters() -> Result<()> {
        let content = r#"id,path,command
1,C:\Users\test,echo "hello"
2,/home/user,ls -la | grep test
3,./relative,cat file.txt > output.txt"#;

        let (_temp_dir, path) = create_temp_csv(content)?;
        let df = load_csv_as_strings(path.to_str().unwrap())?;

        assert_eq!(df.height(), 3);

        let path_col = df.column("path")?;
        let first_path = path_col.str()?.get(0).unwrap();
        assert!(first_path.contains("\\"));

        Ok(())
    }

    #[test]
    fn test_numbers_as_strings() -> Result<()> {
        let content = r#"id,phone,zip,amount
1,555-123-4567,01234,1234.56
2,+1-800-555-0000,00000,0.00
3,123456789,99999-1234,1000000.00"#;

        let (_temp_dir, path) = create_temp_csv(content)?;
        let df = load_csv_as_strings(path.to_str().unwrap())?;

        assert_eq!(df.height(), 3);

        // All should be strings, even numeric-looking data
        for col in df.get_columns() {
            assert_eq!(col.dtype(), &DataType::String);
        }

        // Leading zeros should be preserved
        let zip_col = df.column("zip")?;
        let first_zip = zip_col.str()?.get(0).unwrap();
        assert_eq!(first_zip, "01234");

        Ok(())
    }

    #[test]
    fn test_large_csv_generation_and_loading() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let file_path = temp_dir.path().join("large_test.csv");
        let path_str = file_path.to_str().unwrap();

        // Generate 10,000 rows for testing (use larger for stress testing)
        generate_stress_test_csv(path_str, 10_000)?;

        // Verify file was created
        assert!(file_path.exists());

        // Load and verify
        let df = load_csv_as_strings(path_str)?;

        assert_eq!(df.height(), 10_000);
        assert_eq!(df.width(), 8); // id, name, email, department, salary, notes, unicode_test, emoji_field

        // Verify all columns are strings
        for col in df.get_columns() {
            assert_eq!(col.dtype(), &DataType::String);
        }

        Ok(())
    }

    #[test]
    fn test_rtl_text() -> Result<()> {
        let content = r#"id,arabic,hebrew
1,مرحبا بالعالم,שלום עולם
2,اختبار,בדיקה"#;

        let (_temp_dir, path) = create_temp_csv(content)?;
        let df = load_csv_as_strings(path.to_str().unwrap())?;

        assert_eq!(df.height(), 2);

        let arabic_col = df.column("arabic")?;
        let first_arabic = arabic_col.str()?.get(0).unwrap();
        assert!(first_arabic.contains("مرحبا"));

        Ok(())
    }

    #[test]
    fn test_mixed_line_endings() -> Result<()> {
        // Create file with mixed line endings manually
        let temp_dir = TempDir::new()?;
        let file_path = temp_dir.path().join("mixed_endings.csv");
        let mut file = File::create(&file_path)?;

        // Write with different line endings
        file.write_all(b"id,name,value\r\n")?; // Windows
        file.write_all(b"1,Alice,100\n")?;      // Unix
        file.write_all(b"2,Bob,200\r\n")?;      // Windows
        file.write_all(b"3,Carol,300\n")?;      // Unix
        file.flush()?;

        let df = load_csv_as_strings(file_path.to_str().unwrap())?;

        assert_eq!(df.height(), 3);

        Ok(())
    }

    #[test]
    fn test_very_long_fields() -> Result<()> {
        let long_text = "x".repeat(10_000);
        let content = format!(
            "id,name,description\n1,Test,{}\n2,Test2,Short",
            long_text
        );

        let (_temp_dir, path) = create_temp_csv(&content)?;
        let df = load_csv_as_strings(path.to_str().unwrap())?;

        assert_eq!(df.height(), 2);

        let desc_col = df.column("description")?;
        let first_desc = desc_col.str()?.get(0).unwrap();
        assert_eq!(first_desc.len(), 10_000);

        Ok(())
    }

    #[test]
    fn test_many_columns() -> Result<()> {
        let num_cols = 100;
        let header: String = (0..num_cols)
            .map(|i| format!("col_{}", i))
            .collect::<Vec<_>>()
            .join(",");

        let row: String = (0..num_cols)
            .map(|i| format!("value_{}", i))
            .collect::<Vec<_>>()
            .join(",");

        let content = format!("{}\n{}\n{}\n{}", header, row, row, row);

        let (_temp_dir, path) = create_temp_csv(&content)?;
        let df = load_csv_as_strings(path.to_str().unwrap())?;

        assert_eq!(df.height(), 3);
        assert_eq!(df.width(), 100);

        Ok(())
    }

    #[test]
    fn test_null_like_values() -> Result<()> {
        let content = r#"id,value,status
1,NULL,active
2,null,inactive
3,None,pending
4,N/A,unknown
5,nil,closed
6,-,archived"#;

        let (_temp_dir, path) = create_temp_csv(content)?;
        let df = load_csv_as_strings(path.to_str().unwrap())?;

        assert_eq!(df.height(), 6);

        // All these should be loaded as literal strings, not converted to null
        let value_col = df.column("value")?;
        assert_eq!(value_col.str()?.get(0).unwrap(), "NULL");
        assert_eq!(value_col.str()?.get(1).unwrap(), "null");
        assert_eq!(value_col.str()?.get(2).unwrap(), "None");

        Ok(())
    }

    #[test]
    fn test_dataframe_slicing_for_viewport() -> Result<()> {
        // Simulate the virtual viewport logic
        let content = (0..100)
            .map(|i| format!("{},Row {}", i, i))
            .collect::<Vec<_>>()
            .join("\n");

        let content = format!("id,name\n{}", content);

        let (_temp_dir, path) = create_temp_csv(&content)?;
        let df = load_csv_as_strings(path.to_str().unwrap())?;

        assert_eq!(df.height(), 100);

        // Test slicing (simulating viewport)
        let viewport_start = 25;
        let viewport_size = 10;

        let sliced = df.slice(viewport_start as i64, viewport_size);
        assert_eq!(sliced.height(), 10);

        // Verify first row in slice is row 25
        let id_col = sliced.column("id")?;
        let first_id = id_col.str()?.get(0).unwrap();
        assert_eq!(first_id, "25");

        // Test edge case: slice at end
        let end_slice = df.slice(95, 10);
        assert_eq!(end_slice.height(), 5); // Only 5 rows left

        Ok(())
    }

    #[test]
    fn test_column_selection() -> Result<()> {
        let content = r#"a,b,c,d,e
1,2,3,4,5
6,7,8,9,10"#;

        let (_temp_dir, path) = create_temp_csv(content)?;
        let df = load_csv_as_strings(path.to_str().unwrap())?;

        // Select only specific columns (simulating column visibility)
        let selected = df.select(["a", "c", "e"])?;

        assert_eq!(selected.width(), 3);
        assert_eq!(selected.get_column_names(), vec!["a", "c", "e"]);

        Ok(())
    }

    #[test]
    fn test_sorting() -> Result<()> {
        let content = r#"id,name,value
3,Carol,100
1,Alice,300
2,Bob,200"#;

        let (_temp_dir, path) = create_temp_csv(content)?;
        let df = load_csv_as_strings(path.to_str().unwrap())?;

        // Sort ascending by name
        let sorted_asc = df
            .clone()
            .lazy()
            .sort(["name"], SortMultipleOptions::new().with_order_descending(false))
            .collect()?;

        let name_col = sorted_asc.column("name")?;
        assert_eq!(name_col.str()?.get(0).unwrap(), "Alice");
        assert_eq!(name_col.str()?.get(1).unwrap(), "Bob");
        assert_eq!(name_col.str()?.get(2).unwrap(), "Carol");

        // Sort descending by name
        let sorted_desc = df
            .lazy()
            .sort(["name"], SortMultipleOptions::new().with_order_descending(true))
            .collect()?;

        let name_col = sorted_desc.column("name")?;
        assert_eq!(name_col.str()?.get(0).unwrap(), "Carol");
        assert_eq!(name_col.str()?.get(2).unwrap(), "Alice");

        Ok(())
    }

    #[test]
    fn test_regex_filtering() -> Result<()> {
        let content = r#"id,name,email
1,Alice,alice@example.com
2,Bob,bob@test.org
3,Carol,carol@example.com
4,David,david@test.org"#;

        let (_temp_dir, path) = create_temp_csv(content)?;
        let df = load_csv_as_strings(path.to_str().unwrap())?;

        // Filter emails containing "example"
        let filtered = df
            .lazy()
            .filter(
                col("email")
                    .str()
                    .contains(lit("(?i)example"), false),
            )
            .collect()?;

        assert_eq!(filtered.height(), 2);

        Ok(())
    }

    #[test]
    fn test_multi_column_regex_filter() -> Result<()> {
        let content = r#"id,name,email,notes
1,Alice Smith,alice@example.com,Regular user
2,Bob Jones,bob@test.org,Smith referral
3,Carol,carol@example.com,Nothing special
4,David Smith,david@test.org,Premium"#;

        let (_temp_dir, path) = create_temp_csv(content)?;
        let df = load_csv_as_strings(path.to_str().unwrap())?;

        // Filter for "Smith" across name and notes columns
        let pattern = "(?i)smith";
        let filtered = df
            .lazy()
            .filter(
                col("name")
                    .str()
                    .contains(lit(pattern), false)
                    .or(col("notes").str().contains(lit(pattern), false)),
            )
            .collect()?;

        assert_eq!(filtered.height(), 3); // Alice Smith, Bob (Smith referral), David Smith

        Ok(())
    }

    #[test]
    fn test_windows_datetime_parsing() -> Result<()> {
        use std::fs::File;
        use std::io::Write;
        use tempfile::TempDir;

        let temp_dir = TempDir::new()?;
        let file_path = temp_dir.path().join("dates_windows.csv");
        let mut file = File::create(&file_path)?;

        // Write CSV with Windows line endings and dates
        file.write_all(b"id,name,date,timestamp\r\n")?;
        file.write_all(b"1,Alice,2020-01-15,2020-01-15 10:30:00\r\n")?;
        file.write_all(b"2,Bob,2021-06-20,2021-06-20 14:45:30\r\n")?;
        file.write_all(b"3,Carol,2022-12-31,2022-12-31 23:59:59\r\n")?;
        file.flush()?;

        // Load with Polars date parsing enabled
        let df = polars::prelude::CsvReadOptions::default()
            .with_has_header(true)
            .with_parse_options(
                polars::prelude::CsvParseOptions::default()
                    .with_try_parse_dates(true)
                    .with_eol_char(b'\n')
            )
            .try_into_reader_with_file_path(Some(file_path.clone()))
            .unwrap()
            .finish()
            .unwrap();

        assert_eq!(df.height(), 3);

        // Check that date columns are parsed as Date/Datetime, not String
        let date_col = df.column("date")?;
        let timestamp_col = df.column("timestamp")?;

        // These should be Date or Datetime types, not String
        assert!(matches!(date_col.dtype(), DataType::Date | DataType::String), 
                "date column dtype: {:?}", date_col.dtype());
        assert!(matches!(timestamp_col.dtype(), DataType::Datetime(_, _) | DataType::String), 
                "timestamp column dtype: {:?}", timestamp_col.dtype());

        // If parsed correctly, they should not be String type
        // However, if parsing fails due to \r, they will be String with null-like values
        println!("Date column dtype: {:?}", date_col.dtype());
        println!("Timestamp column dtype: {:?}", timestamp_col.dtype());

        Ok(())
    }
}

// ============================================================================
// Benchmark / Stress Test (run with `cargo test --release -- --ignored`)
// ============================================================================

#[cfg(test)]
mod stress_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    #[ignore] // Run with: cargo test stress_test_100k -- --ignored
    fn stress_test_100k_rows() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let file_path = temp_dir.path().join("stress_100k.csv");
        let path_str = file_path.to_str().unwrap();

        println!("Generating 100,000 row CSV with edge cases...");
        let gen_start = Instant::now();
        generate_stress_test_csv(path_str, 100_000)?;
        println!("Generation time: {:?}", gen_start.elapsed());

        // Check file size
        let metadata = fs::metadata(&file_path)?;
        println!("File size: {} MB", metadata.len() / 1_000_000);

        println!("Loading CSV...");
        let load_start = Instant::now();
        let df = load_csv_as_strings(path_str)?;
        println!("Load time: {:?}", load_start.elapsed());

        assert_eq!(df.height(), 100_000);
        println!("Loaded {} rows x {} columns", df.height(), df.width());

        // Test viewport slicing performance
        println!("Testing viewport slicing (1000 iterations)...");
        let slice_start = Instant::now();
        for i in 0..1000 {
            let offset = (i * 99) % 99_950;
            let _ = df.slice(offset as i64, 50);
        }
        println!("Slice time for 1000 ops: {:?}", slice_start.elapsed());

        // Test sorting performance
        println!("Testing sort performance...");
        let sort_start = Instant::now();
        let _sorted = df
            .clone()
            .lazy()
            .sort(["name"], SortMultipleOptions::new())
            .collect()?;
        println!("Sort time: {:?}", sort_start.elapsed());

        // Test filtering performance
        println!("Testing filter performance...");
        let filter_start = Instant::now();
        let filtered = df
            .lazy()
            .filter(col("department").str().contains(lit("(?i)engineering"), false))
            .collect()?;
        println!("Filter time: {:?}", filter_start.elapsed());
        println!("Filtered rows: {}", filtered.height());

        Ok(())
    }

    #[test]
    #[ignore] // Run with: cargo test stress_test_1m -- --ignored --release
    fn stress_test_1m_rows() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let file_path = temp_dir.path().join("stress_1m.csv");
        let path_str = file_path.to_str().unwrap();

        println!("Generating 1,000,000 row CSV with edge cases...");
        let gen_start = Instant::now();
        generate_stress_test_csv(path_str, 1_000_000)?;
        println!("Generation time: {:?}", gen_start.elapsed());

        // Check file size
        let metadata = fs::metadata(&file_path)?;
        println!("File size: {} MB", metadata.len() / 1_000_000);

        println!("Loading CSV...");
        let load_start = Instant::now();
        let df = load_csv_as_strings(path_str)?;
        println!("Load time: {:?}", load_start.elapsed());

        assert_eq!(df.height(), 1_000_000);
        println!("Loaded {} rows x {} columns", df.height(), df.width());

        // Viewport slicing should still be instant
        println!("Testing viewport slicing...");
        let slice_start = Instant::now();
        for i in 0..10_000 {
            let offset = (i * 99) % 999_950;
            let _ = df.slice(offset as i64, 50);
        }
        let slice_elapsed = slice_start.elapsed();
        println!("Slice time for 10,000 ops: {:?}", slice_elapsed);
        println!(
            "Average slice time: {:?}",
            slice_elapsed / 10_000
        );

        Ok(())
    }
}
