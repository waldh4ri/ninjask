use std::fs::File;
use std::io::{BufWriter, Write};

/// Generates a 2GB CSV file for testing large dataset handling.
/// 
/// The file contains columns: id, name, email, age, city, country, salary, date
/// It generates approximately 2GB of data by writing rows with incrementing IDs.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output_path = "large_2gb.csv";
    let target_size_bytes = 2 * 1024 * 1024 * 1024u64; // 2GB

    println!("Generating 2GB CSV file: {}", output_path);
    println!("Target size: {} bytes ({:.2} GB)", target_size_bytes, target_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0));

    let file = File::create(output_path)?;
    let mut writer = BufWriter::with_capacity(1024 * 1024, file); // 1MB buffer

    // Write header
    writeln!(writer, "id,name,email,age,city,country,salary,date")?;

    let mut current_size = 0u64;
    let mut row_count = 0u64;

    // Sample data pools for variety
    let names = vec!["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"];
    let cities = vec!["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego"];
    let countries = vec!["USA", "Canada", "UK", "Germany", "France", "Spain", "Italy", "Netherlands"];

    // Generate rows until we reach ~2GB
    while current_size < target_size_bytes {
        let name = names[row_count as usize % names.len()];
        let city = cities[row_count as usize % cities.len()];
        let country = countries[row_count as usize % countries.len()];
        let email = format!("user{}@example.com", row_count);
        let age = 20 + (row_count % 50);
        let salary = 50000 + (row_count % 100000);
        let date = format!("2024-{:02}-{:02}", (row_count % 12) + 1, (row_count % 28) + 1);

        let row = format!(
            "{},{},{},{},{},{},{},{}\n",
            row_count, name, email, age, city, country, salary, date
        );

        current_size += row.len() as u64;
        writer.write_all(row.as_bytes())?;
        row_count += 1;

        // Print progress every 100k rows
        if row_count % 100_000 == 0 {
            let progress_gb = current_size as f64 / (1024.0 * 1024.0 * 1024.0);
            println!("Progress: {} rows, {:.2} GB written", row_count, progress_gb);
        }
    }

    writer.flush()?;
    println!("\nCompleted!");
    println!("Total rows: {}", row_count);
    println!("Final size: {} bytes ({:.2} GB)", current_size, current_size as f64 / (1024.0 * 1024.0 * 1024.0));

    Ok(())
}
