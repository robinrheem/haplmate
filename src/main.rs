use anyhow::Result;
use seq_io::fasta::{Reader, Record};
use std::process::exit;

use clap::Parser;

/// Estimating haplotypes with Simulated Annealing and Expectation-Maximization
#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Args {
    /// Input FASTA file(s)
    #[arg(value_name = "FILE", default_value = "-")]
    files: Vec<String>,
    /// Maximum allowed mismatch between haplotypes and reads
    #[arg(short = 'm', long, default_value = "15")]
    mismatches: usize,
    /// Maximum number of EM iterations during intermediate steps
    #[arg(short, long, default_value = "20")]
    em_iterations: usize,
    /// Strength of annealing
    #[arg(long, default_value = "0.1")]
    sa_schedule: f64,
    /// Lambda1 value (for testing purposes only)
    #[arg(long, default_value = "0.1")]
    lambda1: f64,
    /// Lambda2 value (for testing purposes only)
    #[arg(long, default_value = "0.1")]
    lambda2: f64,
    /// Sequencing error
    #[arg(short = 'd', long = "seqerr", default_value = "0.04")]
    error_rate: f64,
    /// Minimum temp to reach in simulated annealing
    #[arg(long, default_value = "0.0")]
    sa_min_temperature: f64,
    /// Starting maximum temp in simulated annealing
    #[arg(long, default_value = "10.0")]
    sa_max_temperature: f64,
    /// Number of iterations in simulated annealing
    #[arg(long, default_value = "500")]
    sa_iterations: usize,
    /// Interval between calling EM
    #[arg(long, default_value = "10")]
    em_interval: usize,
    /// Number of reruns of optimization algorithm
    #[arg(long, default_value = "5")]
    sa_reruns: usize,
    /// Delta to determine intermediate EM convergence steps
    #[arg(long, default_value = "0.5")]
    em_cdelta: f64,
}

/// Check whether all reads in samples are aligned
///
/// # Arguments
///
/// * `samples` - A list of sample filenames
///
/// # Returns
///
/// A vector of unaligned sample filenames, or an error if file I/O or parsing fails.
fn unaligned_samples<'a>(samples: &'a [String]) -> Result<Vec<&'a str>> {
    let mut unaligned: Vec<&'a str> = Vec::new();
    let mut aligned_length: Option<usize> = None;
    for sample in samples {
        let mut reader = Reader::from_path(sample)?;
        while let Some(record) = reader.next() {
            let record = record?;
            let sequence_length = record.seq().len();
            if aligned_length.is_none() {
                aligned_length = Some(sequence_length);
            }
            if Some(sequence_length) != aligned_length {
                unaligned.push(sample.as_str());
                break;
            }
        }
    }
    Ok(unaligned)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let unaligned = unaligned_samples(&args.files)?;
    if !unaligned.is_empty() {
        for sample in &unaligned {
            eprintln!("Sample {} is not aligned", sample);
        }
        exit(1);
    }
    dbg!(args);
    Ok(())
}
