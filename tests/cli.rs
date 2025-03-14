use anyhow::Result;
use assert_cmd::Command;
use predicates::prelude::*;
use std::fs::File;
use std::io::Write;
use tempfile::tempdir;

#[test]
fn dies_no_args() -> Result<()> {
    let mut cmd = Command::cargo_bin("haplmate")?;
    let output = cmd.output()?;
    println!(
        "Command stdout:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
    println!(
        "Command stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("Failed to open sample file"))
        .stderr(predicate::str::contains("is not aligned"));
    Ok(())
}

#[test]
fn test_proposed_haplotypes() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempdir()?;

    // Create sample1.fa with two identical reads
    let sample1_path = temp_dir.path().join("sample1.fa");
    let mut sample1 = File::create(&sample1_path)?;
    writeln!(sample1, ">read1\nACGT\n>read2\nACGT")?;

    // Create sample2.fa with two identical reads
    let sample2_path = temp_dir.path().join("sample2.fa");
    let mut sample2 = File::create(&sample2_path)?;
    writeln!(sample2, ">read1\nTGCA\n>read2\nTGCA")?;

    // Run the command with deterministic parameters
    let mut cmd = Command::cargo_bin("haplmate")?;
    cmd.arg(sample1_path)
        .arg(sample2_path)
        .arg("--sa-reruns=1")
        .arg("--sa-iterations=1")
        .arg("--sa-max-temperature=10.0")
        .arg("--lambda1=0.0")
        .arg("--lambda2=0.0")
        .arg("--error-rate=0.04");

    let output = cmd.output()?;
    println!(
        "Command stdout:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
    println!(
        "Command stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Check the output
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("sequence,"))
        .stdout(predicate::str::contains("ACGT,1,0"))
        .stdout(predicate::str::contains("TGCA,0,1"));

    Ok(())
}

#[test]
fn test_basic_haplotype_estimation() -> Result<()> {
    let temp_dir = tempdir()?;

    // Create sample1.fa with two identical reads
    let sample1_path = temp_dir.path().join("sample1.fa");
    let mut sample1 = File::create(&sample1_path)?;
    writeln!(sample1, ">read1\nACGT\n>read2\nACGT")?;

    // Create sample2.fa with two identical reads
    let sample2_path = temp_dir.path().join("sample2.fa");
    let mut sample2 = File::create(&sample2_path)?;
    writeln!(sample2, ">read1\nTGCA\n>read2\nTGCA")?;

    // Run the command with deterministic parameters
    let mut cmd = Command::cargo_bin("haplmate")?;
    cmd.arg(sample1_path)
        .arg(sample2_path)
        .arg("--sa-reruns=1")
        .arg("--sa-iterations=1")
        .arg("--sa-max-temperature=10.0")
        .arg("--lambda1=0.0")
        .arg("--lambda2=0.0")
        .arg("--error-rate=0.04");

    let output = cmd.output()?;
    println!(
        "Command stdout:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
    println!(
        "Command stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Check the output
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("sequence,"))
        .stdout(predicate::str::contains("ACGT,1,0"))
        .stdout(predicate::str::contains("TGCA,0,1"));

    Ok(())
}

#[test]
fn test_haplotype_estimation_with_gaps() -> Result<()> {
    let temp_dir = tempdir()?;

    let sample_path = temp_dir.path().join("sample.fa");
    let mut sample = File::create(&sample_path)?;
    writeln!(sample, ">read1\nA-CT\n>read2\nA-CT")?;

    let mut cmd = Command::cargo_bin("haplmate")?;
    cmd.arg(sample_path)
        .arg("--sa-reruns=5")
        .arg("--sa-iterations=100")
        .arg("--sa-max-temperature=10.0")
        .arg("--sa-min-temperature=0.0")
        .arg("--sa-schedule=0.1")
        .arg("--em-interval=10")
        .arg("--em-cdelta=0.5")
        .arg("--lambda1=0.0")
        .arg("--lambda2=0.0")
        .arg("--error-rate=0.04");

    let output = cmd.output()?;
    println!(
        "Command stdout:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
    println!(
        "Command stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("sequence,"))
        .stdout(predicate::str::contains("CT"));

    Ok(())
}

#[test]
fn test_unaligned_sequences() -> Result<()> {
    let temp_dir = tempdir()?;

    // Create sample with unaligned sequences
    let sample_path = temp_dir.path().join("unaligned.fa");
    let mut sample = File::create(&sample_path)?;
    writeln!(sample, ">read1\nACGT\n>read2\nACG")?;

    // Run command
    let mut cmd = Command::cargo_bin("haplmate")?;
    cmd.arg(sample_path);

    let output = cmd.output()?;
    println!(
        "Command stdout:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
    println!(
        "Command stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Should fail with error message about unaligned sequences
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("is not aligned"));

    Ok(())
}

#[test]
fn test_error_rate_handling() -> Result<()> {
    let temp_dir = tempdir()?;

    let sample_path = temp_dir.path().join("sample.fa");
    let mut sample = File::create(&sample_path)?;
    writeln!(sample, ">read1\nACGT\n>read2\nACTT")?;

    // Run with high error rate
    let mut cmd = Command::cargo_bin("haplmate")?;
    cmd.arg(sample_path.clone())
        .arg("--sa-reruns=5")
        .arg("--sa-iterations=100")
        .arg("--sa-max-temperature=10.0")
        .arg("--sa-min-temperature=0.0")
        .arg("--sa-schedule=0.1")
        .arg("--em-interval=10")
        .arg("--em-cdelta=0.5")
        .arg("--error-rate=0.5")
        .arg("--lambda1=0.0")
        .arg("--lambda2=0.0");

    // Check that output contains at least one sequence
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("sequence,"));

    // Run with low error rate
    let mut cmd = Command::cargo_bin("haplmate")?;
    cmd.arg(sample_path)
        .arg("--sa-reruns=5")
        .arg("--sa-iterations=100")
        .arg("--sa-max-temperature=10.0")
        .arg("--sa-min-temperature=0.0")
        .arg("--sa-schedule=0.1")
        .arg("--em-interval=10")
        .arg("--em-cdelta=0.5")
        .arg("--error-rate=0.04")
        .arg("--lambda1=0.0")
        .arg("--lambda2=0.0");

    let output = cmd.output()?;
    println!(
        "Command stdout:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
    println!(
        "Command stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Check that output contains at least two different sequences
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("sequence,"))
        .stdout(predicate::str::contains("ACGT"))
        .stdout(predicate::str::contains("ACTT"));

    Ok(())
}

#[test]
fn test_invalid_file() -> Result<()> {
    // Try to run with non-existent file
    let mut cmd = Command::cargo_bin("haplmate")?;
    cmd.arg("nonexistent.fa");

    let output = cmd.output()?;
    println!(
        "Command stdout:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
    println!(
        "Command stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Should fail with error about file not found
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("Failed to open sample file"));

    Ok(())
}

#[test]
fn test_multiple_samples() -> Result<()> {
    let temp_dir = tempdir()?;

    // Create three samples with different sequences
    let sample1_path = temp_dir.path().join("sample1.fa");
    let mut sample1 = File::create(&sample1_path)?;
    writeln!(sample1, ">read1\nACGT\n>read2\nACGT")?;

    let sample2_path = temp_dir.path().join("sample2.fa");
    let mut sample2 = File::create(&sample2_path)?;
    writeln!(sample2, ">read1\nTGCA\n>read2\nTGCA")?;

    let sample3_path = temp_dir.path().join("sample3.fa");
    let mut sample3 = File::create(&sample3_path)?;
    writeln!(sample3, ">read1\nGTAC\n>read2\nGTAC")?;

    // Run with deterministic parameters
    let mut cmd = Command::cargo_bin("haplmate")?;
    cmd.arg(sample1_path)
        .arg(sample2_path)
        .arg(sample3_path)
        .arg("--sa-reruns=1")
        .arg("--sa-iterations=1")
        .arg("--sa-max-temperature=10.0")
        .arg("--lambda1=0.0")
        .arg("--lambda2=0.0")
        .arg("--error-rate=0.04");

    // Run the command and capture output
    let output = cmd.output()?;
    println!(
        "Command stdout:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
    println!(
        "Command stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Check that output contains all three sequences with correct sample assignments
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("sequence,"))
        .stdout(predicate::str::contains("ACGT,1,0,0"))
        .stdout(predicate::str::contains("TGCA,0,1,0"))
        .stdout(predicate::str::contains("GTAC,0,0,1"));

    Ok(())
}
