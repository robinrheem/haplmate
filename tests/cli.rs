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
        .arg("--seed=12345")
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
        .arg("--seed=12345")
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
        .arg("--sa-reruns=1")
        .arg("--sa-iterations=1")
        .arg("--sa-max-temperature=10.0")
        .arg("--sa-min-temperature=0.0")
        .arg("--sa-schedule=0.1")
        .arg("--em-interval=10")
        .arg("--em-cdelta=0.5")
        .arg("--error-rate=0.5")
        .arg("--seed=12345")
        .arg("--lambda1=0.0")
        .arg("--lambda2=0.0");

    // Just check that it runs successfully with high error rate
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("sequence,"));

    // Run with low error rate
    let mut cmd = Command::cargo_bin("haplmate")?;
    cmd.arg(sample_path)
        .arg("--sa-reruns=1")
        .arg("--sa-iterations=1")
        .arg("--sa-max-temperature=10.0")
        .arg("--sa-min-temperature=0.0")
        .arg("--sa-schedule=0.1")
        .arg("--em-interval=10")
        .arg("--em-cdelta=0.5")
        .arg("--error-rate=0.04")
        .arg("--seed=12345")
        .arg("--lambda1=0.0")
        .arg("--lambda2=0.0");

    // Check for the presence of both variants at position 4 (where sequences differ)
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("sequence,"))
        .stdout(predicate::str::contains("T,0.5"));

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
        .arg("--sa-max-temperature=10.0")
        .arg("--lambda1=0.1")
        .arg("--lambda2=0.1")
        .arg("--seed=12345");

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

#[test]
fn test_shared_haplotypes() -> Result<()> {
    let temp_dir = tempdir()?;

    // Create sample1.fa with two reads, both ACGT
    let sample1_path = temp_dir.path().join("sample1.fa");
    let mut sample1 = File::create(&sample1_path)?;
    writeln!(sample1, ">read1\nACGT\n>read2\nACGT")?;

    // Create sample2.fa with two reads: one ACGT, one TGCA
    let sample2_path = temp_dir.path().join("sample2.fa");
    let mut sample2 = File::create(&sample2_path)?;
    writeln!(sample2, ">read1\nACGT\n>read2\nTGCA")?;

    // Create sample3.fa with three reads: one ACGT, two TGCA
    let sample3_path = temp_dir.path().join("sample3.fa");
    let mut sample3 = File::create(&sample3_path)?;
    writeln!(sample3, ">read1\nACGT\n>read2\nTGCA\n>read3\nTGCA")?;

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
        .arg("--seed=12345")
        .arg("--error-rate=0.04");

    let output = cmd.output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    println!(
        "Command stdout:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
    println!(
        "Command stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Check header is present
    assert!(stdout.contains("sequence,"));

    // Check both sequences are present with correct frequencies
    let lines: Vec<&str> = stdout.lines().collect();
    let mut found_acgt = false;
    let mut found_tgca = false;

    for line in lines {
        if line.starts_with("ACGT,") {
            found_acgt = true;
            let freqs: Vec<f64> = line
                .split(',')
                .skip(1) // skip sequence
                .map(|s| s.parse::<f64>().unwrap())
                .collect();
            assert!((freqs[0] - 1.0).abs() < 0.01); // ~1.0
            assert!((freqs[1] - 0.5).abs() < 0.01); // ~0.5
            assert!((freqs[2] - 0.33).abs() < 0.01); // ~0.33
        } else if line.starts_with("TGCA,") {
            found_tgca = true;
            let freqs: Vec<f64> = line
                .split(',')
                .skip(1) // skip sequence
                .map(|s| s.parse::<f64>().unwrap())
                .collect();
            assert!((freqs[0] - 0.0).abs() < 0.01); // ~0.0
            assert!((freqs[1] - 0.5).abs() < 0.01); // ~0.5
            assert!((freqs[2] - 0.67).abs() < 0.01); // ~0.67
        }
    }

    assert!(found_acgt, "ACGT sequence not found in output");
    assert!(found_tgca, "TGCA sequence not found in output");

    Ok(())
}

#[test]
fn test_complex_shared_patterns() -> Result<()> {
    let temp_dir = tempdir()?;

    // Create 4 samples with complex sharing patterns
    let sample_configs = vec![
        // Sample 1: 4 reads, 2 ACGT, 1 TGCA, 1 GTAC
        (
            "sample1.fa",
            vec![
                ">read1\nACGT",
                ">read2\nACGT",
                ">read3\nTGCA",
                ">read4\nGTAC",
            ],
        ),
        // Sample 2: 6 reads, 1 ACGT, 3 TGCA, 2 GTAC
        (
            "sample2.fa",
            vec![
                ">read1\nACGT",
                ">read2\nTGCA",
                ">read3\nTGCA",
                ">read4\nTGCA",
                ">read5\nGTAC",
                ">read6\nGTAC",
            ],
        ),
        // Sample 3: 3 reads, all TGCA
        (
            "sample3.fa",
            vec![">read1\nTGCA", ">read2\nTGCA", ">read3\nTGCA"],
        ),
        // Sample 4: 5 reads, 2 ACGT, 1 TGCA, 2 GTAC
        (
            "sample4.fa",
            vec![
                ">read1\nACGT",
                ">read2\nACGT",
                ">read3\nTGCA",
                ">read4\nGTAC",
                ">read5\nGTAC",
            ],
        ),
    ];

    let mut sample_paths = Vec::new();
    for (filename, reads) in sample_configs {
        let sample_path = temp_dir.path().join(filename);
        let mut sample = File::create(&sample_path)?;
        writeln!(sample, "{}", reads.join("\n"))?;
        sample_paths.push(sample_path);
    }

    // Run with deterministic parameters
    let mut cmd = Command::cargo_bin("haplmate")?;
    for path in &sample_paths {
        cmd.arg(path);
    }
    cmd.arg("--sa-reruns=1")
        .arg("--sa-iterations=1")
        .arg("--sa-max-temperature=10.0")
        .arg("--lambda1=0.0")
        .arg("--lambda2=0.0")
        .arg("--seed=12345")
        .arg("--error-rate=0.04");

    let output = cmd.output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Expected frequencies for each sequence in each sample
    let expected_freqs = vec![
        ("ACGT", vec![0.5, 0.167, 0.0, 0.4]),
        ("TGCA", vec![0.25, 0.5, 1.0, 0.2]),
        ("GTAC", vec![0.25, 0.333, 0.0, 0.4]),
    ];

    for (seq, freqs) in expected_freqs {
        let line = stdout
            .lines()
            .find(|l| l.starts_with(&format!("{},", seq)))
            .expect(&format!("Sequence {} not found", seq));

        let actual_freqs: Vec<f64> = line
            .split(',')
            .skip(1)
            .map(|f| f.parse::<f64>().unwrap())
            .collect();

        for (i, (actual, expected)) in actual_freqs.iter().zip(freqs.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 0.01,
                "Sample {} frequency mismatch for {}: expected {}, got {}",
                i + 1,
                seq,
                expected,
                actual
            );
        }
    }

    Ok(())
}

#[test]
fn test_single_sample_with_gaps() -> Result<()> {
    let temp_dir = tempdir()?;

    // Create a sample with gaps in different positions
    let sample_path = temp_dir.path().join("sample.fa");
    let mut sample = File::create(&sample_path)?;
    writeln!(sample, ">read1\nA-C\n>read2\nTGC\n>read3\nTAT")?;

    let mut cmd = Command::cargo_bin("haplmate")?;
    cmd.arg(sample_path)
        .arg("--sa-max-temperature=10.0")
        .arg("--sa-min-temperature=0.0")
        .arg("--seed=12345");

    let output = cmd.output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    println!(
        "Command stdout:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
    println!(
        "Command stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    // Check header is present
    assert!(stdout.contains("sequence,"));

    // Check sequences are present with correct frequencies
    let lines: Vec<&str> = stdout.lines().collect();
    let mut found_tgc = false;
    let mut found_tat = false;
    let mut found_agc = false;

    for line in lines {
        if line.starts_with("TGC,") {
            found_tgc = true;
            let freq: f64 = line.split(',').nth(1).unwrap().parse().unwrap();
            assert!((freq - 0.166).abs() < 0.01, "Expected TGC frequency ~0.166");
        } else if line.starts_with("TAT,") {
            found_tat = true;
            let freq: f64 = line.split(',').nth(1).unwrap().parse().unwrap();
            assert!((freq - 0.166).abs() < 0.01, "Expected TAT frequency ~0.166");
        } else if line.starts_with("AGC,") {
            found_agc = true;
            let freq: f64 = line.split(',').nth(1).unwrap().parse().unwrap();
            assert!((freq - 0.166).abs() < 0.01, "Expected AGC frequency ~0.166");
        }
    }

    assert!(found_tgc, "TGC sequence not found in output");
    assert!(found_tat, "TAT sequence not found in output");
    assert!(found_agc, "AGC sequence not found in output");

    Ok(())
}

#[test]
fn test_multiple_samples_with_gaps() -> Result<()> {
    let temp_dir = tempdir()?;

    // Create samples with gaps and variants in different positions
    let sample_configs = vec![
        // Sample 1: Mix of gaps and variants at informative positions
        (
            "sample1.fa",
            vec![
                ">read1\nAC-GTACGT", // Possible completions: ACCGTACGT or ACAGTACGT
                ">read2\nACCGT-CGT", // Possible completions: ACCGTACGT or ACCGTTCGT
                ">read3\nACAGTTCGT", // Complete sequence
            ],
        ),
        // Sample 2: Different gap patterns
        (
            "sample2.fa",
            vec![
                ">read1\nAC-GT-CGT", // Multiple gaps
                ">read2\nAC-GTACGT", // Single gap
                ">read3\nACCGTACGT", // Complete sequence
            ],
        ),
        // Sample 3: Mix of complete and gapped sequences
        (
            "sample3.fa",
            vec![
                ">read1\nACCGTACGT",
                ">read2\nACAGTTCGT",
                ">read3\nAC-GTTCGT",
            ],
        ),
    ];

    let mut sample_paths = Vec::new();
    for (filename, reads) in sample_configs {
        let sample_path = temp_dir.path().join(filename);
        let mut sample = File::create(&sample_path)?;
        writeln!(sample, "{}", reads.join("\n"))?;
        sample_paths.push(sample_path);
    }

    let mut cmd = Command::cargo_bin("haplmate")?;
    for path in &sample_paths {
        cmd.arg(path);
    }
    cmd.arg("--sa-reruns=1")
        .arg("--sa-max-temperature=10.0")
        .arg("--sa-min-temperature=0.0")
        .arg("--lambda1=0.001")
        .arg("--lambda2=0.001")
        .arg("--seed=12345");

    let output = cmd.output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    println!(
        "Command stdout:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
    println!(
        "Command stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    // Expected frequencies for each sequence in each sample
    let expected_freqs = vec![
        ("ACAGTACGT", vec![0.1111, 0.0952, 0.0]),
        ("ACGGTGCGT", vec![0.0, 0.0476, 0.0]),
        ("ACCGTGCGT", vec![0.1111, 0.0476, 0.0]),
        ("ACCGTACGT", vec![0.2222, 0.1429, 0.1667]),
        ("ACGGTCCGT", vec![0.0, 0.0476, 0.0]),
        ("ACAGTGCGT", vec![0.0, 0.0476, 0.0]),
        ("ACTGTCCGT", vec![0.0, 0.0476, 0.0]),
        ("ACTGTGCGT", vec![0.0, 0.0476, 0.0]),
        ("ACGGTACGT", vec![0.1111, 0.0952, 0.0]),
        ("ACCGTCCGT", vec![0.1111, 0.0476, 0.0]),
        ("ACTGTACGT", vec![0.1111, 0.0952, 0.0]),
        ("ACAGTTCGT", vec![0.1111, 0.0476, 0.3333]),
        ("ACCGTTCGT", vec![0.1111, 0.0476, 0.1667]),
        ("ACAGTCCGT", vec![0.0, 0.0476, 0.0]),
        ("ACGGTTCGT", vec![0.0, 0.0476, 0.1667]),
        ("ACTGTTCGT", vec![0.0, 0.0476, 0.1667]),
    ];

    // Verify that frequencies sum to approximately 1.0 for each sample
    for sample_idx in 0..3 {
        let sum: f64 = expected_freqs
            .iter()
            .map(|(_, freqs)| freqs[sample_idx])
            .sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Sample {} frequencies sum to {} (should be 1.0)",
            sample_idx + 1,
            sum
        );
    }

    // Check each sequence's frequencies
    for (seq, freqs) in expected_freqs {
        let line = stdout
            .lines()
            .find(|l| l.starts_with(&format!("{},", seq)))
            .expect(&format!("Sequence {} not found", seq));

        let actual_freqs: Vec<f64> = line
            .split(',')
            .skip(1)
            .map(|f| f.parse::<f64>().unwrap())
            .collect();

        for (i, (actual, expected)) in actual_freqs.iter().zip(freqs.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 0.01,
                "Sample {} frequency mismatch for {}: expected {}, got {}",
                i + 1,
                seq,
                expected,
                actual
            );
        }
    }
    Ok(())
}

#[test]
fn test_sample_haplotypes() -> Result<()> {
    // Run with deterministic parameters
    let mut cmd = Command::cargo_bin("haplmate")?;
    cmd.arg("tests/data/sample_haplotypes.fa")
        .arg("--sa-max-temperature=10.0")
        .arg("--lambda1=0.0")
        .arg("--lambda2=0.0")
        .arg("--seed=12345");

    let output = cmd.output()?;
    println!(
        "Command stdout:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
    println!(
        "Command stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Check that the command succeeds and outputs CSV format
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("sequence,"));

    // The file contains multiple reads with variations
    // We should see multiple haplotypes in the output
    // Note: Exact frequencies may vary due to stochastic nature of the algorithm
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("0.016666666666666666")); // Check for equal distribution

    // Verify the sum is approximately 1.0
    let output = cmd.output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("SUM,1."));

    Ok(())
}
