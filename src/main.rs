use ahash::{AHashMap, AHashSet};
use anyhow::{Context, Result};
use argmin::core::{CostFunction, Executor};
use argmin::solver::simulatedannealing::{Anneal, SATempFunc, SimulatedAnnealing};
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use seq_io::fasta::{Reader, Record};
use std::collections::HashSet;
use std::process::exit;
use std::time::Instant;
use tracing::{debug, info, trace};
use tracing_subscriber;

use clap::{Parser, Subcommand};

/// Estimating haplotypes with Simulated Annealing and Expectation-Maximization
#[derive(Debug, Parser)]
#[command(author, version, about)]
#[command(args_conflicts_with_subcommands = true)]
struct Args {
    /// Number of threads to use for parallelization (defaults to number of CPUs)
    #[arg(short = 't', long, global = true)]
    threads: Option<usize>,

    #[command(subcommand)]
    command: Option<Command>,

    #[command(flatten)]
    estimate: EstimateArgs,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Estimate haplotypes from FASTA files (default behavior)
    Estimate(EstimateArgs),
    /// Calculate cost for given haplotypes and reads
    Cost(CostArgs),
    /// Run EM on a given haplotype set and report which haplotypes are removed
    Em(EmArgs),
}

#[derive(Debug, Parser, Default)]
struct EstimateArgs {
    /// Input FASTA file(s)
    #[arg(value_name = "FILE", default_value = "-")]
    files: Vec<String>,
    /// Output file
    #[arg(short, long, default_value = "estimated_haplotypes.csv")]
    output: Option<String>,
    /// Maximum allowed mismatch between haplotypes and reads
    #[arg(short = 'm', long, default_value = "15")]
    mismatches: usize,
    /// Maximum number of EM iterations during intermediate steps
    #[arg(short, long, default_value = "20000")]
    em_iterations: usize,
    /// Lambda1 value (for testing purposes only)
    #[arg(long, default_value = "0.0001")]
    lambda1: f64,
    /// Lambda2 value (for testing purposes only)
    #[arg(long, default_value = "0.0001")]
    lambda2: f64,
    /// Sequencing error
    #[arg(short = 'd', long, default_value = "0.00001")]
    error_rate: f64,
    /// Starting maximum temp in simulated annealing
    #[arg(long, default_value = "10.0")]
    sa_max_temperature: f64,
    /// Number of iterations in simulated annealing
    #[arg(long, default_value = "2000")]
    sa_iterations: usize,
    /// Number of reruns of optimization algorithm
    #[arg(long, default_value = "5")]
    sa_reruns: usize,
    /// Delta to determine intermediate EM convergence steps
    #[arg(long, default_value = "0.1")]
    em_cdelta: f64,
    /// Random seed for deterministic output(testing purposes only)
    #[arg(long)]
    seed: Option<u64>,
    /// SA reannealing after n accepted steps
    #[arg(long, default_value_t = u64::MAX)]
    sa_reannealing_accepted: u64,
    /// SA reannealing after n best cost improvements
    #[arg(long, default_value_t = u64::MAX)]
    sa_reannealing_best: u64,
    /// SA reannealing after n fixed iterations
    #[arg(long, default_value_t = u64::MAX)]
    sa_reannealing_fixed: u64,
    /// SA stall detection after n accepted steps without improvement
    #[arg(long, default_value_t = u64::MAX)]
    sa_stall_accepted: u64,
    /// SA stall detection after n iterations without best cost improvement
    #[arg(long, default_value_t = u64::MAX)]
    sa_stall_best: u64,
    /// Number of crossover points for recombination (more points = more diversity)
    #[arg(long, default_value = "2")]
    recombination_points: usize,
    /// Number of recombine/mutate/read-guided operations per annealing step
    #[arg(long, default_value = "3")]
    operations_per_step: usize,
    /// Path to CSV file containing true haplotypes for debugging/evaluation
    #[arg(long)]
    true_haplotypes_csv: Option<String>,
}

#[derive(Debug, Parser)]
struct CostArgs {
    /// Input FASTA file(s) containing reads
    #[arg(value_name = "FILE")]
    files: Vec<String>,
    /// CSV file containing haplotypes (same format as output)
    #[arg(short = 'c', long, required = true)]
    haplotypes_csv: String,
    /// Maximum allowed mismatch between haplotypes and reads
    #[arg(short = 'm', long, default_value = "15")]
    mismatches: usize,
    /// Lambda1 value (recombination penalty)
    #[arg(long, default_value = "0.0001")]
    lambda1: f64,
    /// Lambda2 value (haplotype count penalty)
    #[arg(long, default_value = "0.0001")]
    lambda2: f64,
    /// Sequencing error rate
    #[arg(short = 'd', long, default_value = "0.00001")]
    error_rate: f64,
}

#[derive(Debug, Parser)]
struct EmArgs {
    /// Input FASTA file(s) containing reads
    #[arg(value_name = "FILE")]
    files: Vec<String>,
    /// CSV file containing haplotypes to run EM on (same format as output)
    #[arg(short = 'c', long, required = true)]
    haplotypes_csv: String,
    /// Write post-EM haplotype frequencies CSV (same format as estimate output)
    #[arg(short, long, default_value = "em_haplotypes.csv")]
    output: Option<String>,
    /// Write per-sample EM log-likelihood trace CSV
    #[arg(long, default_value = "em_log_likelihood.csv")]
    likelihood_output: Option<String>,
    /// Write per-sample, per-iteration true-haplotype frequency trace CSV.
    /// Long format: `sample,iteration,haplotype,frequency`, where `haplotype` is
    /// `true_hap_<i>` (matching the row order in --true-haplotypes-csv) plus a
    /// special `sum` row that is the sum of all true-haplotype frequencies at that
    /// iteration. Only populated when --true-haplotypes-csv is provided; otherwise
    /// the file is still written with just the header so it's obvious where the
    /// output went.
    #[arg(
        long,
        alias = "true-haplotype-trace-output",
        default_value = "em_true_haplotypes_trace.csv"
    )]
    true_haplotypes_trace_output: Option<String>,
    /// Optional CSV file containing true/ground-truth haplotypes for validation
    #[arg(long)]
    true_haplotypes_csv: Option<String>,
    /// Maximum allowed mismatch between haplotypes and reads
    #[arg(short = 'm', long, default_value = "15")]
    mismatches: usize,
    /// Sequencing error rate
    #[arg(short = 'd', long, default_value = "0.00001")]
    error_rate: f64,
    /// Maximum number of EM iterations
    #[arg(short, long, default_value = "20000")]
    em_iterations: usize,
    /// Delta to determine EM convergence
    #[arg(long, default_value = "0.1")]
    em_cdelta: f64,
}

#[derive(Debug, Clone)]
struct Read {
    sequence: Vec<u8>,
    sample: String,
    /// Pre-computed mask: 1 where the read has a non-gap nucleotide, 0 at gap positions.
    /// This avoids re-checking `r != b'-'` for every haplotype comparison.
    non_gap_mask: Vec<u8>,
}

#[derive(Debug, Clone)]
struct Haplotype {
    sequence: Vec<u8>,
    frequencies: Vec<f64>,
}

/// Per-sample diagnostic traces emitted by the EM routines.
#[derive(Debug, Default)]
struct EmTraceResult {
    /// `(sample_name, log_likelihood_per_iteration)` — index 0 holds the initial likelihood,
    /// followed by one entry per completed outer EM iteration.
    likelihoods: Vec<(String, Vec<f64>)>,
    /// `(sample_name, frequencies_per_iteration)` for the haplotypes the caller asked to track.
    /// Each inner `Vec<f64>` has the same length and ordering as `track_haplotype_indices`.
    /// Empty per-sample traces when the caller did not request tracking.
    frequencies: Vec<(String, Vec<Vec<f64>>)>,
}

#[derive(Debug)]
struct OptimizationParameters {
    samples: Vec<String>,
    max_mismatches: usize,
    em_iterations: usize,
    lambda1: f64,
    lambda2: f64,
    error_rate: f64,
    sa_max_temperature: f64,
    sa_iterations: usize,
    sa_reruns: usize,
    em_cdelta: f64,
    original_read_length: usize,
    seed: Option<u64>,
    sa_reannealing_accepted: u64,
    sa_reannealing_best: u64,
    sa_reannealing_fixed: u64,
    sa_stall_accepted: u64,
    sa_stall_best: u64,
    /// Per-position nucleotide frequencies [A, C, G, T] for MAF-based mutation
    nucleotide_frequencies: Vec<[f64; 4]>,
    /// Number of crossover points for recombination
    recombination_points: usize,
    /// Number of recombine/mutate operations per annealing step
    operations_per_step: usize,
    /// True haplotype sequences for debugging (variant positions only)
    true_haplotype_sequences: Option<Vec<Vec<u8>>>,
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
    let mut aligned_length: Option<usize> = None;
    Ok(samples
        .iter()
        .filter(|sample| {
            let reader = Reader::from_path(sample);
            if reader.is_err() {
                eprintln!("Failed to open sample file: {sample}");
                return true; // Mark file as unaligned due to error
            }
            let mut reader = reader.unwrap();
            reader
                .records()
                .filter_map(|result| result.ok())
                .any(|record| {
                    let sequence_length = record.seq().len();
                    if aligned_length.is_none() {
                        aligned_length = Some(sequence_length);
                    }
                    Some(sequence_length) != aligned_length
                })
        })
        .map(|sample| sample.as_str())
        .collect())
}

/// Parse only sequences from a haplotypes CSV file (for true haplotype comparison)
///
/// # Arguments
///
/// * `csv_path` - Path to the CSV file
///
/// # Returns
///
/// A vector of sequences (as Vec<u8>) parsed from the first column of the CSV
fn parse_sequences_from_csv(csv_path: &str) -> Result<Vec<Vec<u8>>> {
    let content = std::fs::read_to_string(csv_path)?;
    let mut lines = content.lines();
    // Skip header
    lines.next();
    let mut sequences = Vec::new();
    for line in lines {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.is_empty() {
            continue;
        }
        if fields[0] == "SUM" {
            continue;
        }
        let mut sequence = fields[0].as_bytes().to_vec();
        // Normalize to uppercase
        for nucleotide in &mut sequence {
            match nucleotide {
                b'a' => *nucleotide = b'A',
                b'c' => *nucleotide = b'C',
                b'g' => *nucleotide = b'G',
                b't' => *nucleotide = b'T',
                _ => {}
            }
        }
        sequences.push(sequence);
    }
    Ok(sequences)
}

/// Parse haplotypes from a CSV file
///
/// # Arguments
///
/// * `csv_path` - Path to the CSV file
/// * `samples` - List of sample names to match against CSV columns
///
/// # Returns
///
/// A vector of Haplotype structs parsed from the CSV
fn parse_haplotypes_csv(csv_path: &str, samples: &[String]) -> Result<Vec<Haplotype>> {
    let content = std::fs::read_to_string(csv_path)?;
    let mut lines = content.lines();
    let header = lines
        .next()
        .ok_or_else(|| anyhow::anyhow!("CSV file is empty"))?;
    let columns: Vec<&str> = header.split(',').collect();
    let sample_indices: Vec<usize> = samples
        .iter()
        .map(|sample| {
            columns
                .iter()
                .position(|col| col == sample)
                .unwrap_or_else(|| panic!("Sample '{}' not found in CSV header", sample))
        })
        .collect();
    let mut haplotypes = Vec::new();
    for line in lines {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.is_empty() {
            continue;
        }
        if fields[0] == "SUM" {
            continue;
        }
        let sequence = fields[0].as_bytes().to_vec();
        let frequencies: Vec<f64> = sample_indices
            .iter()
            .map(|&idx| {
                fields
                    .get(idx)
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0)
            })
            .collect();
        haplotypes.push(Haplotype {
            sequence,
            frequencies,
        });
    }
    Ok(haplotypes)
}

/// Remove all invariants from all reads and track their positions
///
/// # Arguments
///
/// * `reads` - A list of reads
///
/// # Returns
///
/// A tuple containing:
/// * The list of reads with invariants removed
/// * A vector of (position, nucleotide) pairs for the invariant positions
fn remove_invariants(reads: &Vec<Read>) -> (Vec<Read>, Vec<(usize, u8)>) {
    let mut filtered_sequences = vec![Vec::new(); reads.len()];
    let mut invariant_positions = Vec::new();

    for i in 0..reads.first().unwrap().sequence.len() {
        let column: Vec<u8> = reads.iter().map(|read| read.sequence[i]).collect();
        let unique_nucleotides: HashSet<u8> =
            column.iter().filter(|&&c| c != b'-').copied().collect();

        // If only one type of nucleotide (or all gaps), it's invariant
        if unique_nucleotides.len() <= 1 {
            // For all-gap positions, use '-' as the nucleotide
            let nucleotide = unique_nucleotides.iter().next().copied().unwrap_or(b'-');
            invariant_positions.push((i, nucleotide));
            continue;
        }
        for (j, c) in column.into_iter().enumerate() {
            filtered_sequences[j].push(c);
        }
    }
    let filtered_reads = reads
        .iter()
        .enumerate()
        .map(|(i, read)| {
            let seq = filtered_sequences[i].clone();
            let non_gap_mask = seq.iter().map(|&b| u8::from(b != b'-')).collect();
            Read {
                sequence: seq,
                sample: read.sample.clone(),
                non_gap_mask,
            }
        })
        .collect();

    (filtered_reads, invariant_positions)
}

/// Read all reads from the sample
///
/// # Arguments
///
/// * `samples` - A list of sample filenames
///
/// # Returns
///
/// List of reads with sample information
fn extract_reads<'a>(samples: &'a [String]) -> Vec<Read> {
    let mut reads = Vec::new();
    samples.iter().for_each(|sample| {
        let reader = Reader::from_path(sample);
        reader
            .unwrap()
            .records()
            .filter_map(|result| result.ok())
            .for_each(|record| {
                let mut sequence = record.seq().to_vec();
                // Normalize to uppercase for consistent haplotype comparison
                // This treats ACGT, acgt, AcGt, etc. as identical sequences
                for nucleotide in &mut sequence {
                    match nucleotide {
                        b'a' => *nucleotide = b'A',
                        b'c' => *nucleotide = b'C',
                        b'g' => *nucleotide = b'G',
                        b't' => *nucleotide = b'T',
                        _ => {} // Leave other characters unchanged (gaps, N, etc.)
                    }
                }
                let non_gap_mask = sequence.iter().map(|&b| u8::from(b != b'-')).collect();
                reads.push(Read {
                    sequence,
                    sample: sample.to_string(),
                    non_gap_mask,
                });
            });
    });
    reads
}

/// Propose initial haplotype set following C code logic
/// First creates MAF (Major Allele Frequency) haplotype, then uses greedy algorithm
/// to create additional haplotypes from reads that don't match MAF perfectly
///
/// # Arguments
///
/// * `reads` - A list of reads from samples
///
/// # Returns
///
/// List of haplotypes(full sequences) with initial frequencies
fn init_haplotypes(reads: &Vec<Read>, samples: &Vec<String>) -> Vec<Haplotype> {
    if reads.is_empty() {
        debug!("No reads provided, returning empty haplotype set");
        return Vec::new();
    }
    info!("Initializing haplotypes from {} reads", reads.len());
    let sequence_length = reads[0].sequence.len();
    let samples_in_reads: AHashSet<String> = reads.iter().map(|r| r.sample.clone()).collect();
    debug!("Found {} unique samples", samples_in_reads.len());
    // Step 1: Create MAF (Major Allele Frequency) haplotype
    // This follows haplotype_Set_InitializeMAF logic
    let mut maf_sequence = vec![0u8; sequence_length];
    for pos in 0..sequence_length {
        let mut counts = [0; 4]; // A, C, G, T
        for read in reads {
            match read.sequence[pos] {
                b'A' => counts[0] += 1,
                b'C' => counts[1] += 1,
                b'G' => counts[2] += 1,
                b'T' => counts[3] += 1,
                _ => {} // Skip gaps and other characters
            }
        }
        // Follow C code logic exactly for MAF selection, including tie-breaking
        let (a, c, g, t) = (counts[0], counts[1], counts[2], counts[3]);
        maf_sequence[pos] = if a > c && a > g && a > t {
            b'A'
        } else if c > g && c > t {
            b'C'
        } else if g > t {
            b'G'
        } else {
            b'T'
        };
    }
    debug!(
        "Created MAF sequence: {}",
        String::from_utf8_lossy(&maf_sequence)
    );
    // Step 2: Greedy algorithm to create additional haplotypes
    // This follows haplotype_Set_InitializeRest logic
    let num_repeats = 10;
    let all_iteration_results: Vec<Vec<Vec<u8>>> = (0..num_repeats)
        .into_par_iter()
        .map(|repeat| {
            debug!("Running greedy iteration {}", repeat + 1);
            // Each thread gets its own RNG seeded differently
            let mut rng = thread_rng();
            // Create random order of reads (Fisher-Yates shuffle)
            let mut read_indices: Vec<usize> = (0..reads.len()).collect();
            read_indices.shuffle(&mut rng);
            // Track haplotypes found in this iteration
            let mut iteration_haplotypes: Vec<Vec<u8>> = Vec::new();
            for &read_idx in &read_indices {
                let read = &reads[read_idx];
                // Check if read matches MAF perfectly (ignore gaps in read)
                // Optimized: avoid creating iterator chains, use direct indexing
                let mut mismatches_with_maf = 0;
                for pos in 0..sequence_length {
                    let r = read.sequence[pos];
                    if r != maf_sequence[pos] && r != b'-' {
                        mismatches_with_maf += 1;
                    }
                }
                if mismatches_with_maf == 0 {
                    continue; // Skip reads that perfectly match MAF (ignoring their gaps)
                }
                // Try to match against existing haplotypes in this iteration
                let mut matched_haplotype_idx = None;
                for (idx, existing_hap) in iteration_haplotypes.iter().enumerate() {
                    // In C, calc_mismatches_Initialization treats 'N' in haplotypes as a wildcard
                    // and ignores gaps in reads.
                    // Optimized: use direct indexing instead of iterator chains
                    let mut mismatches = 0;
                    for pos in 0..sequence_length {
                        let r = read.sequence[pos];
                        let h = existing_hap[pos];
                        if r != h && r != b'-' && h != b'N' {
                            mismatches += 1;
                        }
                    }
                    if mismatches == 0 {
                        matched_haplotype_idx = Some(idx);
                        break;
                    }
                }
                if let Some(idx) = matched_haplotype_idx {
                    // Perfect match - extend this haplotype with read information
                    let hap = &mut iteration_haplotypes[idx];
                    for pos in 0..sequence_length {
                        let nucleotide = read.sequence[pos];
                        if matches!(nucleotide, b'A' | b'C' | b'G' | b'T') {
                            hap[pos] = nucleotide;
                        }
                    }
                } else {
                    // Create new haplotype from this read
                    let mut new_haplotype = vec![b'-'; sequence_length];
                    for pos in 0..sequence_length {
                        let nucleotide = read.sequence[pos];
                        if matches!(nucleotide, b'A' | b'C' | b'G' | b'T') {
                            new_haplotype[pos] = nucleotide;
                        }
                    }
                    iteration_haplotypes.push(new_haplotype);
                }
            }
            debug!(
                "Iteration {} found {} haplotypes",
                repeat + 1,
                iteration_haplotypes.len()
            );
            iteration_haplotypes
        })
        .collect();
    // Combine all results from parallel iterations using AHashSet for faster hashing
    let mut all_discovered_sequences: AHashSet<Vec<u8>> = AHashSet::new();
    for iteration_result in all_iteration_results {
        for hap in iteration_result {
            all_discovered_sequences.insert(hap);
        }
    }
    info!(
        "Total unique sequences discovered: {}",
        all_discovered_sequences.len()
    );
    // Combine MAF with all discovered haplotypes and remove duplicates
    let mut final_sequences: AHashSet<Vec<u8>> = AHashSet::new();
    final_sequences.insert(maf_sequence.clone());
    for mut seq in all_discovered_sequences {
        // Fill gaps with MAF sequence
        for pos in 0..sequence_length {
            if seq[pos] == b'-' {
                seq[pos] = maf_sequence[pos];
            }
        }
        final_sequences.insert(seq);
    }
    info!(
        "Total unique sequences after filling gaps: {}",
        final_sequences.len()
    );
    // Convert to Haplotype structs and initialize frequencies randomly,
    // replicating allele_Frequencies_Initialize from C code.
    let mut haplotypes: Vec<Haplotype> = final_sequences
        .into_iter()
        .map(|sequence| Haplotype {
            sequence,
            frequencies: vec![0.0; samples.len()],
        })
        .collect();
    let mut rng = thread_rng();
    for (s_idx, _sample) in samples.iter().enumerate() {
        let mut freqs: Vec<f64> = (0..haplotypes.len()).map(|_| rng.gen::<f64>()).collect();
        let sum: f64 = freqs.iter().sum();
        if sum > 0.0 {
            for freq in &mut freqs {
                *freq /= sum;
            }
        }
        for (i, haplotype) in haplotypes.iter_mut().enumerate() {
            haplotype.frequencies[s_idx] = freqs[i];
        }
    }
    for (i, haplotype) in haplotypes.iter().enumerate() {
        let sequence_str = String::from_utf8_lossy(&haplotype.sequence);
        let mut freq_info = String::new();
        for (s_idx, sample) in samples.iter().enumerate() {
            let freq = haplotype.frequencies.get(s_idx).unwrap_or(&0.0);
            if !freq_info.is_empty() {
                freq_info.push_str(", ");
            }
            freq_info.push_str(&format!("{}: {:.6}", sample, freq));
        }
        debug!("Haplotype {}: {} [{}]", i + 1, sequence_str, freq_info);
    }
    info!(
        "Created {} haplotypes with frequency distributions",
        haplotypes.len()
    );
    haplotypes
}

/// Compute per-position nucleotide frequencies from reads
///
/// # Arguments
///
/// * `reads` - A list of reads
///
/// # Returns
///
/// A vector where each element is [A_freq, C_freq, G_freq, T_freq] for that position
fn compute_nucleotide_frequencies(reads: &[Read]) -> Vec<[f64; 4]> {
    if reads.is_empty() {
        return Vec::new();
    }
    let sequence_length = reads[0].sequence.len();
    let mut frequencies = vec![[0.0; 4]; sequence_length];

    for pos in 0..sequence_length {
        let mut counts = [0usize; 4]; // A, C, G, T
        for read in reads {
            match read.sequence[pos] {
                b'A' => counts[0] += 1,
                b'C' => counts[1] += 1,
                b'G' => counts[2] += 1,
                b'T' => counts[3] += 1,
                _ => {} // Skip gaps and other characters
            }
        }
        let total: usize = counts.iter().sum();
        if total > 0 {
            for i in 0..4 {
                frequencies[pos][i] = counts[i] as f64 / total as f64;
            }
        } else {
            // If all gaps, use uniform distribution
            frequencies[pos] = [0.25, 0.25, 0.25, 0.25];
        }
    }
    frequencies
}

/// Restore invariant positions to a sequence
///
/// # Arguments
///
/// * `sequence` - The sequence without invariant positions
/// * `invariant_positions` - Vector of (position, nucleotide) pairs for invariant positions
///
/// # Returns
///
/// The sequence with invariant positions restored
fn restore_invariants(sequence: &[u8], invariant_positions: &[(usize, u8)]) -> Vec<u8> {
    let full_length = sequence.len() + invariant_positions.len();
    let mut restored = vec![0u8; full_length];
    let mut seq_pos = 0;
    let mut curr_pos = 0;

    // Sort positions to ensure correct order
    let mut sorted_positions = invariant_positions.to_vec();
    sorted_positions.sort_by_key(|&(pos, _)| pos);

    // Fill in the sequence
    for (pos, nucleotide) in sorted_positions {
        // Copy sequence up to this position
        while curr_pos < pos && seq_pos < sequence.len() {
            restored[curr_pos] = sequence[seq_pos];
            curr_pos += 1;
            seq_pos += 1;
        }
        // Insert invariant nucleotide
        restored[curr_pos] = nucleotide;
        curr_pos += 1;
    }

    // Copy any remaining sequence
    while seq_pos < sequence.len() {
        restored[curr_pos] = sequence[seq_pos];
        curr_pos += 1;
        seq_pos += 1;
    }

    restored
}

#[derive(Clone)]
struct HaplotypeEstimationProblem {
    samples: Vec<String>,
    reads: Vec<Read>,
    lambda1: f64,
    lambda2: f64,
    em_max_mismatches: usize,
    em_iterations: usize,
    em_convergence_delta: f64,
    sa_max_temperature: f64,
    seed: Option<u64>,
    // Pre-indexed reads by sample: reads_by_sample[sample_idx] = vec of read indices
    reads_by_sample: Vec<Vec<usize>>,
    // Pre-computed positional likelihood table: mismatch_prob_table[k] = e^k * (1-e)^(n-k)
    mismatch_prob_table: Vec<f64>,
    // Per-position nucleotide frequencies [A, C, G, T] for MAF-based mutation
    nucleotide_frequencies: Vec<[f64; 4]>,
    // Number of crossover points for recombination
    recombination_points: usize,
    // Number of recombine/mutate operations per annealing step
    operations_per_step: usize,
    // True haplotype sequences for debugging (variant positions only)
    true_haplotype_sequences: Option<Vec<Vec<u8>>>,
}

#[inline]
fn nuc_to_idx(nuc: u8) -> Option<usize> {
    match nuc {
        b'A' => Some(0),
        b'C' => Some(1),
        b'G' => Some(2),
        b'T' => Some(3),
        _ => None,
    }
}

/// Tracks haplotype set membership and per-position nucleotide counts
/// so that `mutate` and `recombine` can avoid O(h × L) rebuilds each call.
struct AnnealingState {
    existing_sequences: AHashSet<Vec<u8>>,
    per_pos_hap_counts: Vec<[u32; 4]>,
    num_haplotypes: u32,
}

impl AnnealingState {
    fn from_haplotypes(haplotypes: &[Haplotype]) -> Self {
        let seq_len = haplotypes.first().map_or(0, |h| h.sequence.len());
        let mut per_pos_hap_counts = vec![[0u32; 4]; seq_len];
        let mut existing_sequences = AHashSet::with_capacity(haplotypes.len() * 2);

        for hap in haplotypes {
            existing_sequences.insert(hap.sequence.clone());
            for (pos, &nuc) in hap.sequence.iter().enumerate() {
                if let Some(idx) = nuc_to_idx(nuc) {
                    per_pos_hap_counts[pos][idx] += 1;
                }
            }
        }

        Self {
            existing_sequences,
            per_pos_hap_counts,
            num_haplotypes: haplotypes.len() as u32,
        }
    }

    fn add_haplotype(&mut self, sequence: &[u8]) {
        self.existing_sequences.insert(sequence.to_vec());
        for (pos, &nuc) in sequence.iter().enumerate() {
            if let Some(idx) = nuc_to_idx(nuc) {
                self.per_pos_hap_counts[pos][idx] += 1;
            }
        }
        self.num_haplotypes += 1;
    }

    #[inline]
    fn contains(&self, sequence: &[u8]) -> bool {
        self.existing_sequences.contains(sequence)
    }

    #[inline]
    fn hap_counts_at(&self, pos: usize) -> [f64; 4] {
        let c = &self.per_pos_hap_counts[pos];
        [c[0] as f64, c[1] as f64, c[2] as f64, c[3] as f64]
    }

    fn compute_position_weights(&self, nucleotide_frequencies: &[[f64; 4]]) -> Vec<f64> {
        let n = self.num_haplotypes as f64;
        self.per_pos_hap_counts
            .iter()
            .zip(nucleotide_frequencies.iter())
            .map(|(counts, reads_freqs)| {
                (0..4)
                    .map(|j| {
                        let hap_freq = counts[j] as f64 / n;
                        (hap_freq - reads_freqs[j]).powi(2)
                    })
                    .sum::<f64>()
            })
            .collect()
    }
}

impl HaplotypeEstimationProblem {
    fn new(
        samples: Vec<String>,
        reads: Vec<Read>,
        error_rate: f64,
        lambda1: f64,
        lambda2: f64,
        em_max_mismatches: usize,
        em_iterations: usize,
        em_convergence_delta: f64,
        sa_max_temperature: f64,
        original_read_length: usize,
        seed: Option<u64>,
        reads_by_sample: Vec<Vec<usize>>,
        nucleotide_frequencies: Vec<[f64; 4]>,
        recombination_points: usize,
        operations_per_step: usize,
        true_haplotype_sequences: Option<Vec<Vec<u8>>>,
    ) -> Self {
        // Pre-compute positional likelihood table: e^k * (1-e)^(n-k) for k mismatches
        let mismatch_prob_table: Vec<f64> = (0..=em_max_mismatches)
            .map(|m| {
                if m >= original_read_length {
                    0.0
                } else {
                    error_rate.powi(m as i32)
                        * (1.0 - error_rate).powi((original_read_length - m) as i32)
                }
            })
            .collect();
        Self {
            samples,
            reads,
            lambda1,
            lambda2,
            em_max_mismatches,
            em_iterations,
            em_convergence_delta,
            sa_max_temperature,
            seed,
            reads_by_sample,
            mismatch_prob_table,
            nucleotide_frequencies,
            recombination_points,
            operations_per_step,
            true_haplotype_sequences,
        }
    }

    /// Check if a sequence matches any true haplotype and return matching indices
    fn check_true_haplotype_match(&self, sequence: &[u8]) -> Vec<usize> {
        if let Some(ref true_seqs) = self.true_haplotype_sequences {
            true_seqs
                .iter()
                .enumerate()
                .filter(|(_, true_seq)| *true_seq == sequence)
                .map(|(idx, _)| idx)
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Calculate mismatch probability between a read and haplotype.
    ///
    /// Uses a branchless inner loop that the compiler can auto-vectorize with SIMD.
    /// The gap check is eliminated by using the pre-computed `non_gap_mask` on each Read
    /// (1 at non-gap positions, 0 at gaps). This avoids redundant work when comparing
    /// the same read against many haplotypes.
    #[inline]
    fn compute_mismatch_probability(&self, read: &Read, haplotype: &Haplotype) -> f64 {
        let len = read.sequence.len();
        let rs = &read.sequence[..len];
        let hs = &haplotype.sequence[..len];
        let mask = &read.non_gap_mask[..len];

        // Branchless mismatch count — no early exit so the compiler can emit SIMD.
        // Each iteration: (rs[i] != hs[i]) as u32  &  mask[i] as u32
        // With -C target-cpu=native this compiles to vectorized pcmpeqb + pand.
        let mut mismatches: u32 = 0;
        for i in 0..len {
            mismatches += (rs[i] != hs[i]) as u32 & mask[i] as u32;
        }

        if mismatches as usize > self.em_max_mismatches {
            0.0
        } else {
            self.mismatch_prob_table[mismatches as usize]
        }
    }

    /// Pre-compute mismatch probability matrix for all reads against all haplotypes.
    /// Returns a flat Vec in row-major order: matrix[read_idx * num_haps + hap_idx] = probability.
    /// Flat layout gives contiguous memory for cache-friendly access in EM inner loops.
    fn compute_mismatch_matrix(&self, haplotypes: &[Haplotype]) -> Vec<f64> {
        let num_haps = haplotypes.len();
        let mut matrix = vec![0.0; self.reads.len() * num_haps];
        matrix
            .par_chunks_mut(num_haps)
            .zip(self.reads.par_iter())
            .for_each(|(row, read)| {
                for (j, hap) in haplotypes.iter().enumerate() {
                    row[j] = self.compute_mismatch_probability(read, hap);
                }
            });
        matrix
    }

    /// Performs the Square Expectation-Maximization algorithm to estimate haplotype frequencies.
    ///
    /// This is a variant of the standard EM algorithm that uses "squared" updates to help avoid
    /// local optima and improve convergence. The algorithm iteratively:
    ///
    /// 1. Calculates the probability of each read being generated by each haplotype (E-step)
    /// 2. Updates haplotype frequencies based on these probabilities (M-step)
    /// 3. Squares the updates to accelerate convergence
    /// 4. Checks if the likelihood has improved
    /// 5. If likelihood decreases, reverts to previous state
    ///
    /// # Arguments
    ///
    /// * `haplotypes` - Vector of haplotypes to estimate frequencies for. Each haplotype contains:
    ///   - sequence: The nucleotide sequence
    ///   - frequencies: Vec<f64> aligned to `self.samples` order
    ///
    /// # Returns
    ///
    /// * `Ok(())` if frequencies were successfully estimated
    /// * `Err` if an error occurred during estimation
    ///
    /// # Implementation Details
    ///
    /// For each sample:
    ///
    /// - Filters reads belonging to that sample
    /// - Initializes frequencies uniformly across haplotypes
    /// - Calculates initial mismatch probabilities between reads and haplotypes
    /// - Iteratively updates frequencies using squared EM until:
    ///   - Likelihood converges (change < convergence_delta)
    ///   - Maximum iterations reached
    /// - Stores final frequencies > 0.5% in haplotype.frequencies
    ///
    /// The squared updates help escape local optima by taking larger steps in frequency space
    /// while maintaining the convergence guarantees of standard EM.
    ///
    /// # Numerical Considerations
    ///
    /// - Uses log-likelihood to avoid underflow
    /// - Reverts updates that decrease likelihood
    /// - Filters out frequencies < 0.5% to reduce noise
    /// - Handles gaps ('-') in sequences by ignoring them in mismatch calculations
    fn square_expectation_maximization(
        &self,
        haplotypes: &mut Vec<Haplotype>,
        convergence_delta: f64,
        track_haplotype_indices: Option<&[usize]>,
    ) -> Result<EmTraceResult, anyhow::Error> {
        let num_haps = haplotypes.len();
        let mismatch_matrix = self.compute_mismatch_matrix(haplotypes);
        let track = track_haplotype_indices;
        // Process all samples in parallel - each sample is completely independent
        // Collect frequency results, likelihood traces, and (optional) tracked-frequency traces
        // per sample
        let sample_results: Vec<(Vec<f64>, Vec<f64>, Vec<Vec<f64>>)> = self
            .samples
            .par_iter()
            .enumerate()
            .map(|(sample_idx, _sample)| {
                let sample_read_indices = &self.reads_by_sample[sample_idx];
                let num_reads = sample_read_indices.len();
                if num_haps == 1 {
                    let theta_trace = if let Some(idxs) = track {
                        vec![idxs.iter().map(|_| 1.0).collect()]
                    } else {
                        Vec::new()
                    };
                    return (vec![1.0], vec![], theta_trace);
                }
                // Initialize theta from current haplotype frequencies (read-only)
                let mut theta_new: Vec<f64> = haplotypes
                    .iter()
                    .map(|hap| {
                        let f = *hap.frequencies.get(sample_idx).unwrap_or(&0.0);
                        f64::max(1e-10, f)
                    })
                    .collect();
                // Normalize initial frequencies to sum to 1.0
                let sum: f64 = theta_new.iter().sum();
                if sum > 0.0 {
                    for val in theta_new.iter_mut() {
                        *val /= sum;
                    }
                }
                let mismatches: Vec<&[f64]> = sample_read_indices
                    .iter()
                    .map(|&read_idx| {
                        &mismatch_matrix[read_idx * num_haps..(read_idx + 1) * num_haps]
                    })
                    .collect();
                // Initialize mismatch_fp_new = mismatches * theta (flat row-major)
                let mut mismatch_fp_new: Vec<f64> = mismatches
                    .iter()
                    .flat_map(|row| row.iter().zip(&theta_new).map(|(&m, &t)| m * t))
                    .collect();
                // SQUAREM algorithm parameters, matching C code
                let mut step_min = 1.0;
                let mut step_max = 1.0;
                let step_max_d = 1.0;
                let mstep = 4.0;
                let tol = convergence_delta * 0.1; // Same scaling as in C
                let lik_increase = 0.0; // original: 1.0

                // Create intermediate vectors for SQUAREM
                let mut theta_1 = vec![0.0; num_haps];
                let mut theta_2 = vec![0.0; num_haps];
                let mut r = vec![0.0; num_haps];
                let mut v = vec![0.0; num_haps];

                // Create intermediate flat matrices for SQUAREM (row-major)
                let mut mismatch_fp_1 = vec![0.0; num_reads * num_haps];
                let mut mismatch_fp_2 = vec![0.0; num_reads * num_haps];

                // Hoist memberships allocation to avoid re-allocation in every EM update
                let mut memberships = vec![0.0; num_reads * num_haps];
                // EM update closure - equivalent to sqEMUpdate in C
                let mut em_update =
                    |mismatch_fp_in: &[f64], theta_out: &mut [f64], mismatch_fp_out: &mut [f64]| {
                        // E-step: Calculate memberships (normalized probabilities) - row-major access
                        for i in 0..num_reads {
                            let row = i * num_haps;
                            let denom: f64 = mismatch_fp_in[row..row + num_haps].iter().sum();
                            if denom > 0.0 {
                                for j in 0..num_haps {
                                    memberships[row + j] = mismatch_fp_in[row + j] / denom;
                                }
                            } else {
                                // clear row if denom is 0 to avoid using stale values
                                memberships[row..row + num_haps].fill(0.0);
                            }
                        }
                        // M-step: Accumulate memberships row-by-row for cache-friendly access
                        theta_out.fill(0.0);
                        for i in 0..num_reads {
                            let row = i * num_haps;
                            for j in 0..num_haps {
                                theta_out[j] += memberships[row + j];
                            }
                        }
                        for j in 0..num_haps {
                            theta_out[j] /= num_reads as f64;
                        }
                        // Update mismatch_fp with new frequencies - row-major access
                        for i in 0..num_reads {
                            let row = i * num_haps;
                            for j in 0..num_haps {
                                mismatch_fp_out[row + j] = mismatches[i][j] * theta_out[j];
                            }
                        }
                    };
                // Calculate likelihood closure - equivalent to EM_likelihood_sq in C
                let calculate_likelihood = |mismatch_fp: &[f64]| -> f64 {
                    let mut likelihood = 0.0;
                    for i in 0..num_reads {
                        let row = i * num_haps;
                        let row_sum: f64 = mismatch_fp[row..row + num_haps].iter().sum();
                        if row_sum > 0.0 {
                            likelihood += row_sum.ln();
                        }
                    }
                    likelihood
                };
                // Initial likelihood calculation
                let mut likelihood_old = calculate_likelihood(&mismatch_fp_new);
                let mut likelihood_new = likelihood_old;
                let mut likelihood_trace = vec![likelihood_old];
                let mut theta_trace: Vec<Vec<f64>> = Vec::new();
                if let Some(idxs) = track {
                    theta_trace.push(idxs.iter().map(|&i| theta_new[i]).collect());
                }
                let mut iters = 0;
                // Main EM loop
                while iters < self.em_iterations {
                    // Update likelihood_old if new one is valid
                    if !likelihood_new.is_infinite() && !likelihood_new.is_nan() {
                        likelihood_old = likelihood_new;
                    }
                    // First EM update: theta_0 -> theta_1
                    em_update(&mismatch_fp_new, &mut theta_1, &mut mismatch_fp_1);
                    // Second EM update: theta_1 -> theta_2
                    em_update(&mismatch_fp_1, &mut theta_2, &mut mismatch_fp_2);
                    iters += 2;
                    // Calculate step vectors and norms
                    let mut rsq = 0.0;
                    let mut vsq = 0.0;
                    let mut v2sq = 0.0;
                    for j in 0..num_haps {
                        r[j] = theta_1[j] - theta_new[j];
                        rsq += r[j] * r[j];
                        v[j] = (theta_2[j] - theta_1[j]) - r[j];
                        vsq += v[j] * v[j];
                        v2sq += (theta_2[j] - theta_1[j]) * (theta_2[j] - theta_1[j]);
                    }
                    // Early convergence check based on tolerance
                    if rsq.sqrt() < tol {
                        // theta_0 and theta_1 tolerance - use theta_1 results
                        theta_new.copy_from_slice(&theta_1);
                        break;
                    } else if v2sq.sqrt() < tol {
                        // theta_1 and theta_2 tolerance - use theta_2 results
                        theta_new.copy_from_slice(&theta_2);
                        break;
                    }
                    // SQUAREM acceleration step - carefully follow C implementation
                    let mut alpha = f64::max(step_min, f64::min(step_max, (rsq / vsq).sqrt()));
                    if alpha.is_nan() || alpha.is_infinite() {
                        alpha = 1.0; // Fallback to regular EM step
                    }
                    // Compute accelerated parameter estimates - following C logic
                    // Directly update frequencies first without updating mismatch_fp
                    for j in 0..num_haps {
                        theta_new[j] = theta_new[j] - 2.0 * alpha * r[j] + alpha * alpha * v[j];
                        theta_new[j] = f64::max(1e-10, theta_new[j]);
                    }
                    // Renormalize to the simplex after projection to keep a valid mixture
                    let sum_theta: f64 = theta_new.iter().sum();
                    if sum_theta > 0.0 {
                        for val in theta_new.iter_mut() {
                            *val /= sum_theta;
                        }
                    }
                    // Recompute mismatch_fp_new using the normalized theta - row-major access
                    for i in 0..num_reads {
                        let row = i * num_haps;
                        for j in 0..num_haps {
                            mismatch_fp_new[row + j] = mismatches[i][j] * theta_new[j];
                        }
                    }
                    // Stabilization step if alpha far from 1.0
                    if (alpha - 1.0).abs() > 0.01 {
                        // Instead of cloning a large matrix, we use one of our pre-allocated
                        // matrices as a buffer and then swap.
                        em_update(&mismatch_fp_new, &mut theta_new, &mut mismatch_fp_1);
                        std::mem::swap(&mut mismatch_fp_new, &mut mismatch_fp_1);
                        iters += 1;
                    }
                    likelihood_new = calculate_likelihood(&mismatch_fp_new);
                    // If likelihood decreased, revert to theta_2
                    if likelihood_new.is_infinite()
                        || likelihood_new.is_nan()
                        || likelihood_new <= likelihood_old - lik_increase
                    {
                        // Copy theta_2 to theta_new
                        theta_new.copy_from_slice(&theta_2);
                        std::mem::swap(&mut mismatch_fp_new, &mut mismatch_fp_2);

                        // Update likelihood
                        likelihood_new = calculate_likelihood(&mismatch_fp_new);

                        // Adjust step_max if at boundary
                        if alpha == step_max {
                            step_max = f64::max(step_max_d, step_max / mstep);
                        }
                        alpha = 1.0;
                    }
                    // Increase step_max if we're hitting its boundary
                    if alpha == step_max {
                        step_max = mstep * step_max;
                    }
                    if step_min < 0.0 && alpha == step_min {
                        step_min = mstep * step_min;
                    }
                    likelihood_trace.push(likelihood_new);
                    if let Some(idxs) = track {
                        theta_trace.push(idxs.iter().map(|&i| theta_new[i]).collect());
                    }
                    if (likelihood_new - likelihood_old).abs() < convergence_delta {
                        break;
                    }
                }
                (theta_new, likelihood_trace, theta_trace)
            })
            .collect();

        // Write back all frequencies sequentially (no contention, safe)
        for (sample_idx, (freqs, _, _)) in sample_results.iter().enumerate() {
            for (hap_idx, haplotype) in haplotypes.iter_mut().enumerate() {
                haplotype.frequencies[sample_idx] = freqs[hap_idx];
            }
        }
        // Remove haplotypes with zero frequencies across all samples
        let mut indices_to_remove = Vec::new();
        for (hap_idx, haplotype) in haplotypes.iter().enumerate() {
            if !haplotype
                .frequencies
                .iter()
                .any(|&freq| !freq.is_nan() && freq >= 0.005)
            {
                indices_to_remove.push(hap_idx);
            }
        }
        // Remove haplotypes in reverse order to maintain correct indices
        haplotypes.retain(|h| h.frequencies.iter().any(|&f| !f.is_nan() && f >= 0.005));
        // Rescale frequencies to sum to 1.0 for each sample (like rescaleAlleleFrequencies in C)
        for sample_idx in 0..self.samples.len() {
            let mut sum = 0.0;
            for haplotype in haplotypes.iter() {
                sum += haplotype.frequencies[sample_idx];
            }
            if sum > 0.0 {
                for haplotype in haplotypes.iter_mut() {
                    haplotype.frequencies[sample_idx] /= sum;
                }
            }
        }
        let mut likelihoods: Vec<(String, Vec<f64>)> = Vec::with_capacity(self.samples.len());
        let mut frequencies: Vec<(String, Vec<Vec<f64>>)> = Vec::with_capacity(self.samples.len());
        for (name, (_, ll_trace, theta_trace)) in
            self.samples.iter().zip(sample_results.into_iter())
        {
            likelihoods.push((name.clone(), ll_trace));
            frequencies.push((name.clone(), theta_trace));
        }
        Ok(EmTraceResult {
            likelihoods,
            frequencies,
        })
    }

    fn expectation_maximization(
        &self,
        haplotypes: &mut Vec<Haplotype>,
        convergence_delta: f64,
        track_haplotype_indices: Option<&[usize]>,
    ) -> Result<EmTraceResult, anyhow::Error> {
        let num_haps = haplotypes.len();
        let mismatch_matrix = self.compute_mismatch_matrix(haplotypes);
        let track = track_haplotype_indices;
        // Process all samples in parallel - each sample is completely independent
        // Collect frequency results, likelihood traces, and (optional) tracked-frequency traces
        // per sample
        let sample_results: Vec<(Vec<f64>, Vec<f64>, Vec<Vec<f64>>)> = self
            .samples
            .par_iter()
            .enumerate()
            .map(|(sample_idx, _sample)| {
                let sample_read_indices = &self.reads_by_sample[sample_idx];
                let num_reads = sample_read_indices.len();
                let calculate_likelihood = |mismatch_fp: &[f64]| -> f64 {
                    let mut likelihood = 0.0;
                    for i in 0..num_reads {
                        let row = i * num_haps;
                        let row_sum: f64 = mismatch_fp[row..row + num_haps].iter().sum();
                        if row_sum > 0.0 {
                            likelihood += row_sum.ln();
                        }
                    }
                    likelihood
                };
                if num_haps == 1 {
                    let theta_trace = if let Some(idxs) = track {
                        vec![idxs.iter().map(|_| 1.0).collect()]
                    } else {
                        Vec::new()
                    };
                    return (vec![1.0], vec![], theta_trace);
                }
                // Initialize frequencies uniformly if not already set
                let mut theta: Vec<f64> = haplotypes
                    .iter()
                    .map(|hap| {
                        let f = *hap.frequencies.get(sample_idx).unwrap_or(&0.0);
                        f64::max(1e-10, f)
                    })
                    .collect();
                // Normalize initial frequencies to sum to 1.0
                let sum: f64 = theta.iter().sum();
                if sum > 0.0 {
                    for val in theta.iter_mut() {
                        *val /= sum;
                    }
                }
                let mismatches: Vec<&[f64]> = sample_read_indices
                    .iter()
                    .map(|&read_idx| {
                        &mismatch_matrix[read_idx * num_haps..(read_idx + 1) * num_haps]
                    })
                    .collect();
                // Initialize mismatch_fp_new = mismatches * theta (flat row-major)
                let mut mismatch_fp_new: Vec<f64> = mismatches
                    .iter()
                    .flat_map(|row| row.iter().zip(&theta).map(|(&m, &t)| m * t))
                    .collect();
                // Calculate initial likelihood
                let mut likelihood_old = calculate_likelihood(&mismatch_fp_new);
                let mut likelihood_trace = vec![likelihood_old];
                let mut theta_trace: Vec<Vec<f64>> = Vec::new();
                if let Some(idxs) = track {
                    theta_trace.push(idxs.iter().map(|&i| theta[i]).collect());
                }
                let mut iters = 0;
                // Hoist memberships allocation to avoid re-allocation in every EM iteration
                let mut memberships = vec![0.0; num_reads * num_haps];
                // Main EM loop
                let mut theta_old = vec![0.0; num_haps];
                while iters < self.em_iterations {
                    theta_old.copy_from_slice(&theta);
                    // E-step: Calculate memberships (normalized probabilities) - row-major access
                    for i in 0..num_reads {
                        let row = i * num_haps;
                        let denom: f64 = mismatch_fp_new[row..row + num_haps].iter().sum();
                        if denom > 0.0 {
                            for j in 0..num_haps {
                                memberships[row + j] = mismatch_fp_new[row + j] / denom;
                            }
                        } else {
                            // Clear row if denom is 0 to avoid using stale values
                            memberships[row..row + num_haps].fill(0.0);
                        }
                    }
                    // M-step: Accumulate memberships row-by-row for cache-friendly access
                    theta.fill(0.0);
                    for i in 0..num_reads {
                        let row = i * num_haps;
                        for j in 0..num_haps {
                            theta[j] += memberships[row + j];
                        }
                    }
                    for j in 0..num_haps {
                        theta[j] /= num_reads as f64;
                        // Ensure minimum probability
                        theta[j] = f64::max(1e-10, theta[j]);
                    }
                    for i in 0..num_reads {
                        let row = i * num_haps;
                        for j in 0..num_haps {
                            mismatch_fp_new[row + j] = mismatches[i][j] * theta[j];
                        }
                    }
                    // Normalize to sum to 1.0
                    let sum: f64 = theta.iter().sum();
                    if sum > 0.0 {
                        for val in theta.iter_mut() {
                            *val /= sum;
                        }
                    }
                    // Calculate new likelihood
                    let likelihood_new = calculate_likelihood(&mismatch_fp_new);
                    likelihood_trace.push(likelihood_new);
                    if let Some(idxs) = track {
                        theta_trace.push(idxs.iter().map(|&i| theta[i]).collect());
                    }
                    if (likelihood_new - likelihood_old).abs() < convergence_delta {
                        break;
                    }
                    // Check for parameter convergence as backup
                    let param_change: f64 = theta
                        .iter()
                        .zip(&theta_old)
                        .map(|(&new, &old)| (new - old).abs())
                        .sum();
                    if param_change < convergence_delta {
                        break;
                    }
                    likelihood_old = likelihood_new;
                    iters += 1;
                }
                (theta, likelihood_trace, theta_trace)
            })
            .collect();
        // Write back all frequencies sequentially (no contention, safe)
        for (sample_idx, (freqs, _, _)) in sample_results.iter().enumerate() {
            for (hap_idx, haplotype) in haplotypes.iter_mut().enumerate() {
                haplotype.frequencies[sample_idx] = freqs[hap_idx];
            }
        }
        // Remove haplotypes with zero frequencies across all samples
        let mut indices_to_remove = Vec::new();
        for (hap_idx, haplotype) in haplotypes.iter().enumerate() {
            if !haplotype
                .frequencies
                .iter()
                .any(|&freq| !freq.is_nan() && freq >= 0.005)
            {
                indices_to_remove.push(hap_idx);
            }
        }
        haplotypes.retain(|h| h.frequencies.iter().any(|&f| !f.is_nan() && f >= 0.005));
        // Rescale frequencies to sum to 1.0 for each sample (like rescaleAlleleFrequencies in C)
        for sample_idx in 0..self.samples.len() {
            let mut sum = 0.0;
            for haplotype in haplotypes.iter() {
                sum += haplotype.frequencies[sample_idx];
            }
            if sum > 0.0 {
                for haplotype in haplotypes.iter_mut() {
                    haplotype.frequencies[sample_idx] /= sum;
                }
            }
        }
        let mut likelihoods: Vec<(String, Vec<f64>)> = Vec::with_capacity(self.samples.len());
        let mut frequencies: Vec<(String, Vec<Vec<f64>>)> = Vec::with_capacity(self.samples.len());
        for (name, (_, ll_trace, theta_trace)) in
            self.samples.iter().zip(sample_results.into_iter())
        {
            likelihoods.push((name.clone(), ll_trace));
            frequencies.push((name.clone(), theta_trace));
        }
        Ok(EmTraceResult {
            likelihoods,
            frequencies,
        })
    }

    /// Calculates the minimum number of recombination events required to explain the given set of haplotypes
    /// using the Four Gamete Test (FGT) method.
    ///
    /// The FGT looks at pairs of positions in the haplotypes and checks if all four possible gametes (allele combinations)
    /// are present. If all four gametes are found between two positions, at least one recombination event must have occurred
    /// between those positions.
    ///
    /// # Arguments
    ///
    /// * `haplotypes` - A vector of Haplotype objects to analyze for recombination events
    ///
    /// # Returns
    ///
    /// The minimum number of recombination events (Rmin) required to explain the haplotype data
    ///
    /// # Algorithm
    ///
    /// 1. For each pair of positions, checks if all four gametes are present
    /// 2. Records intervals (position pairs) where four gametes are found
    /// 3. Trims overlapping intervals to avoid double-counting
    /// 4. Returns count of remaining intervals as Rmin
    fn min_recombinations(&self, haplotypes: &Vec<Haplotype>) -> usize {
        if haplotypes.len() <= 1 {
            return 0;
        }
        let num_haps = haplotypes.len();
        let length = haplotypes[0].sequence.len();
        let num_words = (num_haps + 63) / 64;
        let mask_stride = 4 * num_words;
        // Precompute bitmasks: for each (position, nucleotide), a bitset of which
        // haplotypes carry that nucleotide. This turns the per-pair gamete check from
        // O(numHaps) into O(num_words) ≈ O(1) for typical haplotype counts.
        let mut masks = vec![0u64; length * mask_stride];
        for (h, haplotype) in haplotypes.iter().enumerate() {
            let word = h / 64;
            let bit = 1u64 << (h % 64);
            for (pos, &nuc) in haplotype.sequence.iter().enumerate() {
                let nuc_idx = match nuc {
                    b'A' => 0usize,
                    b'C' => 1,
                    b'G' => 2,
                    b'T' => 3,
                    _ => continue,
                };
                masks[pos * mask_stride + nuc_idx * num_words + word] |= bit;
            }
        }
        // Precompute which nucleotides are present at each position (4-bit mask)
        let mut nuc_present = vec![0u8; length];
        for pos in 0..length {
            let base = pos * mask_stride;
            for nuc in 0..4usize {
                let off = base + nuc * num_words;
                for w in 0..num_words {
                    if masks[off + w] != 0 {
                        nuc_present[pos] |= 1u8 << nuc;
                        break;
                    }
                }
            }
        }
        let mut interval_list = vec![-1i32; length];
        'outer: for pos1 in 0..length {
            let np1 = nuc_present[pos1];
            let nc1 = np1.count_ones();
            if nc1 == 0 {
                continue;
            }
            let base1 = pos1 * mask_stride;
            for pos2 in (pos1 + 1)..length {
                let np2 = nuc_present[pos2];
                // Upper bound on distinct gametes is the product of distinct nucleotides
                if nc1 * np2.count_ones() < 3 {
                    continue;
                }
                let base2 = pos2 * mask_stride;
                let mut num_gametes = 0u32;
                for n1 in 0..4usize {
                    if np1 & (1 << n1) == 0 {
                        continue;
                    }
                    let off1 = base1 + n1 * num_words;
                    for n2 in 0..4usize {
                        if np2 & (1 << n2) == 0 {
                            continue;
                        }
                        let off2 = base2 + n2 * num_words;
                        for w in 0..num_words {
                            if masks[off1 + w] & masks[off2 + w] != 0 {
                                num_gametes += 1;
                                if num_gametes > 3 {
                                    interval_list[pos1] = pos2 as i32;
                                    continue 'outer;
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }
        // Trim intervals (Hudson & Kaplan 1985)
        for pos1 in 0..length {
            if interval_list[pos1] == -1 {
                continue;
            }
            for pos2 in 0..length {
                if interval_list[pos2] == -1 || pos2 == pos1 {
                    continue;
                } else if pos2 <= pos1 && interval_list[pos1] <= interval_list[pos2] {
                    interval_list[pos2] = -1;
                } else if pos1 < pos2 && pos2 < interval_list[pos1] as usize {
                    interval_list[pos2] = -1;
                }
            }
        }
        interval_list.iter().filter(|&&x| x != -1).count()
    }

    /// Applies a random annealing operation to the haplotype set
    ///
    /// # Arguments
    /// * `haplotypes` - The haplotype set to modify
    /// * `rng` - Random number generator to use
    ///
    /// # Returns
    /// Whether an operation was successfully applied
    fn random_operation(
        &self,
        haplotypes: &mut Vec<Haplotype>,
        rng: &mut impl Rng,
        state: &mut AnnealingState,
    ) -> bool {
        let operation: i32 = if haplotypes.len() == 1 {
            debug!("Only one haplotype present, forcing add operation");
            // 50/50 between mutate and read-guided when only 1 haplotype
            if rng.gen_bool(0.5) {
                1
            } else {
                2
            }
        } else {
            rng.gen_range(0..3)
        };
        match operation {
            0 if haplotypes.len() >= 2 => {
                debug!("Operation: Recombine");
                self.recombine(haplotypes, rng, state);
                true
            }
            1 if haplotypes.len() < self.reads.len() => {
                debug!("Operation: Mutate");
                self.mutate(haplotypes, rng, state);
                true
            }
            2 if haplotypes.len() < self.reads.len() => {
                debug!("Operation: Read-guided proposal");
                self.read_guided_propose(haplotypes, rng, state);
                true
            }
            _ => {
                trace!("No operation performed - conditions not met");
                false
            }
        }
    }

    /// Applies multi-point recombination operation between two random haplotypes.
    /// Generates ALL possible recombinant children from the crossover points.
    ///
    /// # Arguments
    /// * `haplotypes` - The haplotype set to modify
    /// * `rng` - Random number generator to use
    ///
    /// # Crossover mechanism
    /// With n crossover points creating n+1 segments, each segment can come from either parent.
    /// This generates 2^(n+1) - 2 possible recombinants (excluding the pure parents).
    ///
    /// For example, with 2 crossover points (3 segments [A, B, C]):
    /// - Parent 1 provides: a1, b1, c1
    /// - Parent 2 provides: a2, b2, c2
    /// - Possible children: [a1,b1,c2], [a1,b2,c1], [a1,b2,c2], [a2,b1,c1], [a2,b1,c2], [a2,b2,c1]
    fn recombine(
        &self,
        haplotypes: &mut Vec<Haplotype>,
        rng: &mut impl Rng,
        state: &mut AnnealingState,
    ) {
        let idx1 = rng.gen_range(0..haplotypes.len());
        let mut idx2 = rng.gen_range(0..haplotypes.len());
        let mut attempts = 0;
        const MAX_ATTEMPTS: i32 = 100;
        trace!("Initial recombination pair: indices {} and {}", idx1, idx2);
        loop {
            if attempts >= MAX_ATTEMPTS {
                debug!(
                    "Failed to find compatible haplotypes after {} attempts",
                    attempts
                );
                return;
            }
            if idx1 == idx2 {
                trace!("Same indices, regenerating idx2");
                idx2 = rng.gen_range(0..haplotypes.len());
                attempts += 1;
                continue;
            }
            let seq_len = haplotypes[idx1].sequence.len();
            let num_points = self.recombination_points.min(seq_len.saturating_sub(1));
            if num_points == 0 {
                debug!("Sequence too short for recombination");
                return;
            }
            let mut crossover_points: Vec<usize> = Vec::with_capacity(num_points);
            while crossover_points.len() < num_points {
                let point = rng.gen_range(1..seq_len);
                if !crossover_points.contains(&point) {
                    crossover_points.push(point);
                }
            }
            crossover_points.sort_unstable();
            debug!(
                "Performing multi-point recombination at positions {:?} between haplotypes {} and {}",
                crossover_points, idx1, idx2
            );
            let parent1 = &haplotypes[idx1].sequence;
            let parent2 = &haplotypes[idx2].sequence;
            let num_segments = crossover_points.len() + 1;
            let mut boundaries = Vec::with_capacity(num_segments + 1);
            boundaries.push(0);
            boundaries.extend(&crossover_points);
            boundaries.push(seq_len);
            let mut segments1: Vec<&[u8]> = Vec::with_capacity(num_segments);
            let mut segments2: Vec<&[u8]> = Vec::with_capacity(num_segments);
            for i in 0..num_segments {
                segments1.push(&parent1[boundaries[i]..boundaries[i + 1]]);
                segments2.push(&parent2[boundaries[i]..boundaries[i + 1]]);
            }
            let num_combinations = 1usize << num_segments;
            // Use incremental state instead of rebuilding existing_sequences from scratch
            let mut new_sequences: Vec<Vec<u8>> = Vec::new();
            let mut new_sequences_set: AHashSet<Vec<u8>> = AHashSet::new();
            for combo in 1..(num_combinations - 1) {
                let mut child = Vec::with_capacity(seq_len);
                for seg_idx in 0..num_segments {
                    if (combo >> seg_idx) & 1 == 1 {
                        child.extend_from_slice(segments2[seg_idx]);
                    } else {
                        child.extend_from_slice(segments1[seg_idx]);
                    }
                }
                if !state.contains(&child) && !new_sequences_set.contains(&child) {
                    new_sequences_set.insert(child.clone());
                    new_sequences.push(child);
                }
            }
            debug!("Generated {} new unique sequences", new_sequences.len());
            if new_sequences.is_empty() {
                trace!("No new unique sequences generated, retrying with different indices");
                idx2 = rng.gen_range(0..haplotypes.len());
                attempts += 1;
                continue;
            }
            let original_freq1: Vec<f64> = haplotypes[idx1].frequencies.clone();
            let original_freq2: Vec<f64> = haplotypes[idx2].frequencies.clone();
            let num_new = new_sequences.len();
            let freq_per_child = 1.0 / (num_new as f64 + 2.0);
            for freq in &mut haplotypes[idx1].frequencies {
                *freq *= freq_per_child;
            }
            for freq in &mut haplotypes[idx2].frequencies {
                *freq *= freq_per_child;
            }
            for new_seq in new_sequences {
                // Check if this matches a true haplotype
                let matches = self.check_true_haplotype_match(&new_seq);
                if !matches.is_empty() {
                    debug!(
                        "Recombination generated true haplotype(s) at CSV index(es): {:?}",
                        matches
                    );
                }
                let mut combined_frequencies = vec![0.0; self.samples.len()];
                for s in 0..self.samples.len() {
                    let freq1 = original_freq1.get(s).unwrap_or(&0.0);
                    let freq2 = original_freq2.get(s).unwrap_or(&0.0);
                    combined_frequencies[s] = (freq1 + freq2) * freq_per_child;
                }
                state.add_haplotype(&new_seq);
                haplotypes.push(Haplotype {
                    sequence: new_seq,
                    frequencies: combined_frequencies,
                });
            }
            break;
        }
    }
    /// Proposes a new haplotype guided by poorly explained reads.
    ///
    /// Finds reads that are most mismatched against every current haplotype,
    /// picks one (weighted by mismatch count), then grafts that read's variants
    /// onto its closest haplotype to build a candidate. This is a data-driven
    /// proposal: reads are fragments of true haplotypes, so their variant
    /// patterns carry real biological signal that random mutation cannot reach.
    fn read_guided_propose(
        &self,
        haplotypes: &mut Vec<Haplotype>,
        rng: &mut impl Rng,
        state: &mut AnnealingState,
    ) {
        let num_haps = haplotypes.len();
        // For each read, find its best (min) mismatch count against any haplotype
        let read_best_mismatches: Vec<u32> = self
            .reads
            .iter()
            .map(|read| {
                haplotypes
                    .iter()
                    .map(|hap| {
                        let len = read.sequence.len();
                        let mut mm: u32 = 0;
                        for i in 0..len {
                            mm += (read.sequence[i] != hap.sequence[i]) as u32
                                & read.non_gap_mask[i] as u32;
                        }
                        mm
                    })
                    .min()
                    .unwrap_or(u32::MAX)
            })
            .collect();
        // Build weights: reads with more mismatches are more interesting.
        // Use mismatch^2 to strongly prefer the worst-explained reads.
        let weights: Vec<f64> = read_best_mismatches
            .iter()
            .map(|&mm| (mm as f64) * (mm as f64))
            .collect();
        let total_weight: f64 = weights.iter().sum();
        if total_weight < 1e-12 {
            debug!("Read-guided: all reads perfectly explained, falling back to mutate");
            self.mutate(haplotypes, rng, state);
            return;
        }
        const MAX_ATTEMPTS: usize = 50;
        for _attempt in 0..MAX_ATTEMPTS {
            // Weighted sample a poorly-explained read
            let dist = WeightedIndex::new(&weights)
                .unwrap_or_else(|_| WeightedIndex::new(&vec![1.0; self.reads.len()]).unwrap());
            let read_idx = dist.sample(rng);
            let read = &self.reads[read_idx];
            // Find the closest haplotype to this read (fewest mismatches)
            let best_hap_idx = (0..num_haps)
                .min_by_key(|&j| {
                    let hap = &haplotypes[j];
                    let len = read.sequence.len();
                    let mut mm: u32 = 0;
                    for i in 0..len {
                        mm += (read.sequence[i] != hap.sequence[i]) as u32
                            & read.non_gap_mask[i] as u32;
                    }
                    mm
                })
                .unwrap();
            // Graft the read's variant positions onto the closest haplotype
            let mut new_sequence = haplotypes[best_hap_idx].sequence.clone();
            for i in 0..read.sequence.len() {
                if read.non_gap_mask[i] == 1 && read.sequence[i] != new_sequence[i] {
                    new_sequence[i] = read.sequence[i];
                }
            }
            if state.contains(&new_sequence) {
                trace!("Read-guided proposal produced duplicate, retrying");
                continue;
            }
            let matches = self.check_true_haplotype_match(&new_sequence);
            if !matches.is_empty() {
                debug!(
                    "Read-guided proposal generated true haplotype(s) at CSV index(es): {:?}",
                    matches
                );
            }
            debug!(
                "Adding read-guided haplotype (from read {} with {} mismatches to best hap)",
                read_idx, read_best_mismatches[read_idx]
            );
            for freq in &mut haplotypes[best_hap_idx].frequencies {
                *freq /= 2.0;
            }
            let new_freqs = haplotypes[best_hap_idx].frequencies.clone();
            state.add_haplotype(&new_sequence);
            haplotypes.push(Haplotype {
                sequence: new_sequence,
                frequencies: new_freqs,
            });
            return;
        }
        debug!(
            "Read-guided: failed to produce unique haplotype after {} attempts, falling back to mutate",
            MAX_ATTEMPTS
        );
        self.mutate(haplotypes, rng, state);
    }

    /// Applies mutation operation to create a new haplotype
    /// Mutates a haplotype using distribution-guided nucleotide sampling.
    ///
    /// First picks a random position. At that position, compares the nucleotide
    /// distribution of the proposed haplotypes against the reads nucleotide distribution.
    /// Picks a random haplotype that carries the most overrepresented nucleotide and
    /// replaces it with the most underrepresented nucleotide. Retries with a new random
    /// position until a unique new haplotype is produced.
    fn mutate(
        &self,
        haplotypes: &mut Vec<Haplotype>,
        rng: &mut impl Rng,
        state: &mut AnnealingState,
    ) {
        let seq_len = haplotypes[0].sequence.len();
        let nucleotides = [b'A', b'C', b'G', b'T'];

        // O(L) via incremental counts instead of O(h × L) full recount
        let position_weights = state.compute_position_weights(&self.nucleotide_frequencies);
        let pos_dist = WeightedIndex::new(&position_weights)
            .unwrap_or_else(|_| WeightedIndex::new(&vec![1.0; seq_len]).unwrap());

        const MAX_ATTEMPTS: usize = 100;
        for _attempt in 0..MAX_ATTEMPTS {
            let pos = pos_dist.sample(rng);

            // O(1) lookup from incremental state instead of O(h) recount
            let hap_counts = state.hap_counts_at(pos);
            let num_haplotypes = state.num_haplotypes as f64;
            let hap_freqs: [f64; 4] = [
                hap_counts[0] / num_haplotypes,
                hap_counts[1] / num_haplotypes,
                hap_counts[2] / num_haplotypes,
                hap_counts[3] / num_haplotypes,
            ];

            let reads_freqs = &self.nucleotide_frequencies[pos];
            let overestimates: [f64; 4] = [
                hap_freqs[0] - reads_freqs[0],
                hap_freqs[1] - reads_freqs[1],
                hap_freqs[2] - reads_freqs[2],
                hap_freqs[3] - reads_freqs[3],
            ];
            let over_nuc_idx = overestimates
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();
            let over_nuc = nucleotides[over_nuc_idx];

            let underestimates: [f64; 4] = [
                reads_freqs[0] - hap_freqs[0],
                reads_freqs[1] - hap_freqs[1],
                reads_freqs[2] - hap_freqs[2],
                reads_freqs[3] - hap_freqs[3],
            ];
            let under_nuc_idx = underestimates
                .iter()
                .enumerate()
                .filter(|(idx, _)| *idx != over_nuc_idx)
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();
            let under_nuc = nucleotides[under_nuc_idx];

            let candidates: Vec<usize> = (0..haplotypes.len())
                .filter(|&i| haplotypes[i].sequence[pos] == over_nuc)
                .collect();
            if candidates.is_empty() {
                continue;
            }
            let idx_to_copy = candidates[rng.gen_range(0..candidates.len())];
            let mut new_sequence = haplotypes[idx_to_copy].sequence.clone();
            new_sequence[pos] = under_nuc;

            if state.contains(&new_sequence) {
                trace!(
                    "Distribution-guided mutation at pos {} ({} -> {}) produced duplicate, skipping",
                    pos,
                    over_nuc as char,
                    under_nuc as char
                );
                continue;
            }
            let matches = self.check_true_haplotype_match(&new_sequence);
            if !matches.is_empty() {
                debug!(
                    "Mutation generated true haplotype(s) at CSV index(es): {:?}",
                    matches
                );
            }
            debug!(
                "Adding mutated haplotype (pos {} {} -> {})",
                pos, over_nuc as char, new_sequence[pos] as char
            );
            for freq in &mut haplotypes[idx_to_copy].frequencies {
                *freq /= 2.0;
            }
            let new_freqs = haplotypes[idx_to_copy].frequencies.clone();
            state.add_haplotype(&new_sequence);
            haplotypes.push(Haplotype {
                sequence: new_sequence,
                frequencies: new_freqs,
            });
            return;
        }
        debug!(
            "Failed to generate unique mutated haplotype after {} attempts",
            MAX_ATTEMPTS
        );
    }
}

impl CostFunction for HaplotypeEstimationProblem {
    type Param = Vec<Haplotype>;
    type Output = f64;

    /// Calculates the total cost (objective function) for a set of proposed haplotypes.
    ///
    /// This cost function combines three components:
    /// 1. Sequence mismatch cost: How well the haplotypes explain the observed reads
    /// 2. Recombination penalty: Penalizes solutions requiring many recombination events
    /// 3. Complexity penalty: Penalizes solutions with too many haplotypes
    ///
    /// # Arguments
    ///
    /// * `haplotypes` - The proposed set of haplotypes to evaluate
    ///
    /// # Returns
    ///
    /// * `Ok(f64)` - The total cost value, where lower values indicate better solutions
    /// * `Err(anyhow::Error)` - If there was an error calculating the cost
    ///
    /// # Cost Components
    ///
    /// - Mismatch cost: For each read, calculates probability of observing its mismatches
    ///   against compatible haplotypes (those from same sample). Takes negative log.
    ///
    /// - Recombination penalty: Uses four-gamete test to estimate minimum number of
    ///   recombination events needed. Multiplied by lambda1 parameter.
    ///
    /// - Complexity penalty: Number of haplotypes multiplied by lambda2 parameter.
    ///
    /// # Implementation Details
    ///
    /// - Ignores gap positions ('-') when counting mismatches
    /// - Uses positional likelihood model for mismatches: e^k * (1-e)^(n-k)
    /// - Only considers haplotypes from matching sample when calculating read probabilities
    /// - Higher costs indicate worse solutions
    fn cost(&self, haplotypes: &Self::Param) -> std::result::Result<Self::Output, anyhow::Error> {
        let num_haps = haplotypes.len();
        let mismatch_matrix = self.compute_mismatch_matrix(haplotypes);
        let total_cost: f64 = self
            .samples
            .par_iter()
            .enumerate()
            .map(|(sample_idx, _sample)| {
                let sample_read_indices = &self.reads_by_sample[sample_idx];
                let mut sample_cost = 0.0;
                for &read_idx in sample_read_indices {
                    let mut total_mismatch_probability = 0.0;
                    let row = read_idx * num_haps;
                    for (hap_idx, haplotype) in haplotypes.iter().enumerate() {
                        let probability = mismatch_matrix[row + hap_idx];
                        let frequency = *haplotype.frequencies.get(sample_idx).unwrap_or(&0.0);
                        total_mismatch_probability += probability * frequency;
                    }
                    if total_mismatch_probability > 0.0 {
                        sample_cost -= total_mismatch_probability.ln();
                    }
                }
                sample_cost
            })
            .sum();
        // Penalty from four gamete test
        let recombinations = self.min_recombinations(haplotypes) as f64;
        info!("Recombination: {}", recombinations);
        let total_cost = total_cost + self.lambda1 * recombinations;
        // Penalty for number of haplotypes
        let total_cost = total_cost + self.lambda2 * haplotypes.len() as f64;
        info!("Total cost: {}", total_cost);
        Ok(total_cost)
    }
}

impl Anneal for HaplotypeEstimationProblem {
    type Param = Vec<Haplotype>;
    type Output = Vec<Haplotype>;
    type Float = f64;

    /// Performs a single annealing step by randomly modifying the current set of haplotypes.
    ///
    /// This function implements two possible operations, chosen randomly:
    /// 1. Recombine two random haplotypes by performing a crossover (if there are at least 2 haplotypes)
    /// 2. Add a new haplotype by mutating an existing one (if number of haplotypes < number of reads)
    ///
    /// Operations are performed `operations_per_step` times to "explode" the haplotype set
    /// before running EM optimization.
    ///
    /// After structural modifications, it runs Square EM to optimize the frequencies.
    /// If EM removes all haplotypes, it retries with different operations (matching C code behavior).
    ///
    /// # Arguments
    ///
    /// * `param` - Current set of haplotypes to modify
    /// * `temp` - Current temperature in the annealing process (between 0 and 1)
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<Haplotype>)` - A new set of haplotypes after applying random modifications
    /// * `Err(anyhow::Error)` - If an error occurs during frequency optimization
    fn anneal(
        &self,
        param: &Self::Param,
        temp: Self::Float,
    ) -> Result<Self::Output, anyhow::Error> {
        debug!(
            "Starting annealing step with temperature {}, operations_per_step={}",
            temp, self.operations_per_step
        );
        if param.is_empty() {
            debug!("No haplotypes available for annealing operations, returning original set");
            return Ok(param.clone());
        }
        // EM convergence schedule: starts loose (0.1) at high temperature,
        // tightens to em_convergence_delta at low temperature (matching C code)
        let em_temp_start = 0.1;
        let sa_progress = temp / self.sa_max_temperature;
        let convergence_delta =
            self.em_convergence_delta + (em_temp_start - self.em_convergence_delta) * sa_progress;
        // Retry mechanism: keep trying until we get a non-empty result
        const MAX_RETRIES: usize = 10000;
        for retry_count in 0..=MAX_RETRIES {
            let mut haplotypes = param.clone();
            // Create RNG with potentially different seed for each retry
            let mut rng = if let Some(seed) = self.seed {
                rand::rngs::StdRng::seed_from_u64(seed + retry_count as u64)
            } else {
                rand::rngs::StdRng::from_entropy()
            };
            // Build incremental state once; mutate/recombine update it in O(L) per op
            let mut annealing_state = AnnealingState::from_haplotypes(&haplotypes);
            let mut operations_applied = 0;
            for op_num in 0..self.operations_per_step {
                if self.random_operation(&mut haplotypes, &mut rng, &mut annealing_state) {
                    operations_applied += 1;
                    trace!(
                        "Operation {}/{} applied, now have {} haplotypes",
                        op_num + 1,
                        self.operations_per_step,
                        haplotypes.len()
                    );
                }
            }
            if operations_applied == 0 {
                debug!(
                    "No operations could be applied on attempt {}",
                    retry_count + 1
                );
                if retry_count == MAX_RETRIES {
                    break;
                }
                continue;
            }

            debug!(
                "Applied {} operations, running EM optimization on {} haplotypes",
                operations_applied,
                haplotypes.len()
            );
            if haplotypes.len() > 30 {
                self.square_expectation_maximization(&mut haplotypes, convergence_delta, None)?;
            } else {
                self.expectation_maximization(&mut haplotypes, convergence_delta, None)?;
            }
            if !haplotypes.is_empty() {
                debug!(
                    "Annealing step complete, returning {} haplotypes",
                    haplotypes.len()
                );
                return Ok(haplotypes);
            }
            debug!(
                "EM optimization removed all haplotypes (attempt {}/{}), retrying",
                retry_count + 1,
                MAX_RETRIES + 1
            );
        }
        debug!(
            "Failed to find valid haplotype set after {} retries, using original set",
            MAX_RETRIES
        );
        Ok(param.clone())
    }
}

/// Propose most likely haplotypes with
/// simulated annealing and expectation-maximization
///
/// # Arguments
///
/// * `reads` - A list of reads from samples
/// * `haplotypes` - A list of initial haplotypes
///
/// # Returns
///
/// List of newly proposed haplotypes
fn propose_haplotypes(
    reads: &Vec<Read>,
    initial_haplotypes: &Vec<Haplotype>,
    optimization_parameters: OptimizationParameters,
) -> Vec<Haplotype> {
    let samples = optimization_parameters.samples.clone();
    let mut reads_by_sample: Vec<Vec<usize>> = vec![Vec::new(); samples.len()];
    for (read_idx, read) in reads.iter().enumerate() {
        if let Some(sample_idx) = samples.iter().position(|s| s == &read.sample) {
            reads_by_sample[sample_idx].push(read_idx);
        }
    }
    let problem = HaplotypeEstimationProblem::new(
        samples,
        reads.to_vec(),
        optimization_parameters.error_rate,
        optimization_parameters.lambda1,
        optimization_parameters.lambda2,
        optimization_parameters.max_mismatches,
        optimization_parameters.em_iterations,
        optimization_parameters.em_cdelta,
        optimization_parameters.sa_max_temperature,
        optimization_parameters.original_read_length,
        optimization_parameters.seed,
        reads_by_sample,
        optimization_parameters.nucleotide_frequencies.clone(),
        optimization_parameters.recombination_points,
        optimization_parameters.operations_per_step,
        optimization_parameters.true_haplotype_sequences.clone(),
    );
    info!(
        "Estimating haplotypes with parameters: samples={}, reads={}, error_rate={}, lambda1={}, lambda2={}, em_max_mismatches={}, em_iterations={}, em_convergence_delta={}, sa_max_temperature={}, sa_iterations={}, sa_reruns={}, original_read_length={}, seed={:?}, recombination_points={}, operations_per_step={}",
        problem.samples.len(),
        reads.len(),
        optimization_parameters.error_rate,
        optimization_parameters.lambda1,
        optimization_parameters.lambda2,
        optimization_parameters.max_mismatches,
        optimization_parameters.em_iterations,
        optimization_parameters.em_cdelta,
        optimization_parameters.sa_max_temperature,
        optimization_parameters.sa_iterations,
        optimization_parameters.sa_reruns,
        optimization_parameters.original_read_length,
        optimization_parameters.seed,
        optimization_parameters.recombination_points,
        optimization_parameters.operations_per_step
    );
    // Optimize initial haplotypes with EM before starting SA
    let mut best_haplotypes = initial_haplotypes.clone();
    info!(
        "Running EM optimization on initial {} haplotypes",
        best_haplotypes.len()
    );
    // Pre-SA EM: convergence_delta=2.0 gives tol=0.2 (parameter-space early stopping in
    // SQUAREM). Stops EM before spurious haplotypes stabilize above the 0.5% threshold.
    // Old code accidentally used 1.8 (tol=0.18) which gave ~11 surviving haplotypes.
    if let Err(e) = problem.square_expectation_maximization(&mut best_haplotypes, 2.0, None) {
        info!(
            "EM optimization failed: {}, proceeding with unoptimized haplotypes",
            e
        );
    }
    info!(
        "EM optimization completed. {} haplotypes remain",
        best_haplotypes.len()
    );
    let sa_reruns = optimization_parameters.sa_reruns;
    let num_samples = optimization_parameters.samples.len();
    let results: Vec<Vec<Haplotype>> = (0..sa_reruns)
        .into_par_iter()
        .map(|i| {
            let _span = tracing::info_span!("sa_run", run = i + 1, of = sa_reruns).entered();
            info!("Running SA with {} haplotypes", best_haplotypes.len(),);
            let run_rng = if let Some(seed) = optimization_parameters.seed {
                rand::rngs::StdRng::seed_from_u64(seed.wrapping_add(i as u64))
            } else {
                rand::rngs::StdRng::from_entropy()
            };
            let run_solver = SimulatedAnnealing::new_with_rng(
                optimization_parameters.sa_max_temperature,
                run_rng,
            )
            .unwrap()
            .with_temp_func(SATempFunc::Boltzmann)
            .with_reannealing_fixed(optimization_parameters.sa_reannealing_fixed)
            .with_reannealing_accepted(optimization_parameters.sa_reannealing_accepted)
            .with_reannealing_best(optimization_parameters.sa_reannealing_best)
            .with_stall_accepted(optimization_parameters.sa_stall_accepted)
            .with_stall_best(optimization_parameters.sa_stall_best);
            let mut run_problem = problem.clone();
            if let Some(seed) = optimization_parameters.seed {
                run_problem.seed = Some(seed.wrapping_add(i as u64 * 1000));
            }
            let result = Executor::new(run_problem, run_solver)
                .configure(|state| state.param(best_haplotypes.clone()))
                .run()
                .unwrap();
            let best_cost = result.state().best_cost;
            info!("SA run complete, cost: {}", best_cost);
            result
                .state()
                .best_param
                .clone()
                .unwrap_or_else(|| best_haplotypes.clone())
        })
        .collect();
    let mut merged: AHashMap<Vec<u8>, Vec<f64>> = AHashMap::new();
    for run_haplotypes in &results {
        for hap in run_haplotypes {
            let entry = merged
                .entry(hap.sequence.clone())
                .or_insert_with(|| vec![0.0; num_samples]);
            for (j, freq) in hap.frequencies.iter().enumerate() {
                entry[j] += freq;
            }
        }
    }
    let mut merged_haplotypes: Vec<Haplotype> = merged
        .into_iter()
        .map(|(seq, freqs)| Haplotype {
            sequence: seq,
            frequencies: freqs,
        })
        .collect();
    let sample_sums: Vec<f64> = (0..num_samples)
        .map(|s| {
            merged_haplotypes
                .iter()
                .map(|h| h.frequencies[s])
                .sum::<f64>()
        })
        .collect();
    for hap in &mut merged_haplotypes {
        for (s, freq) in hap.frequencies.iter_mut().enumerate() {
            if sample_sums[s] > 0.0 {
                *freq /= sample_sums[s];
            }
        }
    }
    info!(
        "Merged {} total haplotypes from {} parallel runs into {} unique haplotypes",
        results.iter().map(|r| r.len()).sum::<usize>(),
        sa_reruns,
        merged_haplotypes.len()
    );
    let _final_span = tracing::info_span!("sa_final").entered();
    info!(
        "Running final SA pass on {} merged haplotypes",
        merged_haplotypes.len()
    );
    if let Err(e) = problem.square_expectation_maximization(
        &mut merged_haplotypes,
        optimization_parameters.em_cdelta,
        None,
    ) {
        info!(
            "EM optimization failed: {}, proceeding with unoptimized haplotypes",
            e
        );
    }
    info!(
        "EM optimization completed. {} haplotypes remain",
        merged_haplotypes.len()
    );
    let final_rng = if let Some(seed) = optimization_parameters.seed {
        rand::rngs::StdRng::seed_from_u64(seed.wrapping_add(sa_reruns as u64))
    } else {
        rand::rngs::StdRng::from_entropy()
    };
    let final_solver =
        SimulatedAnnealing::new_with_rng(optimization_parameters.sa_max_temperature, final_rng)
            .unwrap()
            .with_temp_func(SATempFunc::Boltzmann)
            .with_reannealing_fixed(optimization_parameters.sa_reannealing_fixed)
            .with_reannealing_accepted(optimization_parameters.sa_reannealing_accepted)
            .with_reannealing_best(optimization_parameters.sa_reannealing_best)
            .with_stall_accepted(optimization_parameters.sa_stall_accepted)
            .with_stall_best(optimization_parameters.sa_stall_best);
    let final_result = Executor::new(problem.clone(), final_solver)
        .configure(|state| state.param(merged_haplotypes.clone()))
        .run()
        .unwrap();
    best_haplotypes = if let Some(ref param) = final_result.state().best_param {
        info!(
            "Final SA pass complete, {} haplotypes with cost {}",
            param.len(),
            final_result.state().best_cost
        );
        param.clone()
    } else {
        info!("Final SA pass produced no result, keeping merged haplotypes");
        merged_haplotypes
    };
    best_haplotypes
}

fn haplotype_frequencies_output(
    haplotypes: &Vec<Haplotype>,
    invariant_positions: &[(usize, u8)],
    samples: &Vec<String>,
) -> String {
    let mut output = String::new();
    output.push_str("sequence");
    for sample in samples {
        output.push_str(&format!(",{}", sample));
    }
    output.push('\n');
    for haplotype in haplotypes {
        let restored_sequence = restore_invariants(&haplotype.sequence, invariant_positions);
        output.push_str(&String::from_utf8_lossy(&restored_sequence));
        for s_idx in 0..samples.len() {
            output.push_str(&format!(",{}", haplotype.frequencies[s_idx]));
        }
        output.push('\n');
    }
    output.push_str("SUM");
    for s_idx in 0..samples.len() {
        let sum: f64 = haplotypes.iter().map(|h| h.frequencies[s_idx]).sum();
        output.push_str(&format!(",{}", sum));
    }
    output.push('\n');
    output
}

/// Run the estimate subcommand
fn run_estimate(mut args: EstimateArgs) -> Result<()> {
    args.files.sort_by(|a, b| natord::compare(a, b));
    let unaligned = unaligned_samples(&args.files)?;
    if !unaligned.is_empty() {
        unaligned
            .iter()
            .for_each(|sample| eprintln!("Sample {sample} is not aligned"));
        exit(1);
    }
    let reads = extract_reads(&args.files);
    info!(
        "Original read length: {} nucleotides",
        reads[0].sequence.len()
    );
    let (variant_only_reads, invariant_positions) = remove_invariants(&reads);
    info!(
        "Read length after removing {} invariant positions: {} nucleotides",
        invariant_positions.len(),
        variant_only_reads[0].sequence.len()
    );
    let initial_haplotypes = init_haplotypes(&variant_only_reads, &args.files);
    if initial_haplotypes.len() == 1 && initial_haplotypes[0].sequence.is_empty() {
        eprintln!("No initial haplotypes that have meaningful information");
        exit(1);
    }
    let nucleotide_frequencies = compute_nucleotide_frequencies(&variant_only_reads);
    // Parse true haplotypes if provided and remove invariant positions
    let true_haplotype_sequences = if let Some(ref csv_path) = args.true_haplotypes_csv {
        match parse_sequences_from_csv(csv_path) {
            Ok(sequences) => {
                let invariant_indices: HashSet<usize> =
                    invariant_positions.iter().map(|(pos, _)| *pos).collect();
                let variant_only_sequences: Vec<Vec<u8>> = sequences
                    .into_iter()
                    .map(|seq| {
                        seq.iter()
                            .enumerate()
                            .filter(|(i, _)| !invariant_indices.contains(i))
                            .map(|(_, &b)| b)
                            .collect()
                    })
                    .collect();
                info!(
                    "Loaded {} true haplotypes from {} for comparison",
                    variant_only_sequences.len(),
                    csv_path
                );
                Some(variant_only_sequences)
            }
            Err(e) => {
                eprintln!("Warning: Failed to parse true haplotypes CSV: {}", e);
                None
            }
        }
    } else {
        None
    };
    let optimization_parameters = OptimizationParameters {
        samples: args.files.clone(),
        max_mismatches: args.mismatches,
        em_cdelta: args.em_cdelta,
        em_iterations: args.em_iterations,
        error_rate: args.error_rate,
        lambda1: args.lambda1,
        lambda2: args.lambda2,
        sa_iterations: args.sa_iterations,
        sa_max_temperature: args.sa_max_temperature,
        sa_reruns: args.sa_reruns,
        original_read_length: reads[0].sequence.len(),
        seed: args.seed,
        sa_reannealing_accepted: args.sa_reannealing_accepted,
        sa_reannealing_best: args.sa_reannealing_best,
        sa_reannealing_fixed: args.sa_reannealing_fixed,
        sa_stall_accepted: args.sa_stall_accepted,
        sa_stall_best: args.sa_stall_best,
        nucleotide_frequencies,
        recombination_points: args.recombination_points,
        operations_per_step: args.operations_per_step,
        true_haplotype_sequences,
    };
    let proposed_haplotypes = propose_haplotypes(
        &variant_only_reads,
        &initial_haplotypes,
        optimization_parameters,
    );
    let output =
        haplotype_frequencies_output(&proposed_haplotypes, &invariant_positions, &args.files);
    println!("{}", output);
    if let Some(output_file) = args.output {
        std::fs::write(output_file, output).unwrap();
    }
    Ok(())
}

/// Run the cost subcommand
fn run_cost(mut args: CostArgs) -> Result<()> {
    args.files.sort_by(|a, b| natord::compare(a, b));
    let unaligned = unaligned_samples(&args.files)?;
    if !unaligned.is_empty() {
        unaligned
            .iter()
            .for_each(|sample| eprintln!("Sample {sample} is not aligned"));
        exit(1);
    }
    let reads = extract_reads(&args.files);
    let original_read_length = reads[0].sequence.len();
    info!("Original read length: {} nucleotides", original_read_length);
    let (variant_only_reads, invariant_positions) = remove_invariants(&reads);
    info!(
        "Read length after removing {} invariant positions: {} nucleotides",
        invariant_positions.len(),
        variant_only_reads[0].sequence.len()
    );

    // Parse haplotypes from CSV and remove the SAME invariant positions as determined by reads
    let haplotypes = parse_haplotypes_csv(&args.haplotypes_csv, &args.files)?;
    let invariant_indices: HashSet<usize> =
        invariant_positions.iter().map(|(pos, _)| *pos).collect();
    let variant_only_haplotypes: Vec<Haplotype> = haplotypes
        .into_iter()
        .map(|h| Haplotype {
            sequence: h
                .sequence
                .iter()
                .enumerate()
                .filter(|(i, _)| !invariant_indices.contains(i))
                .map(|(_, &b)| b)
                .collect(),
            frequencies: h.frequencies,
        })
        .collect();
    let mut reads_by_sample: Vec<Vec<usize>> = vec![Vec::new(); args.files.len()];
    for (read_idx, read) in variant_only_reads.iter().enumerate() {
        if let Some(sample_idx) = args.files.iter().position(|s| s == &read.sample) {
            reads_by_sample[sample_idx].push(read_idx);
        }
    }
    // Create the problem instance to compute cost
    let problem = HaplotypeEstimationProblem::new(
        args.files.clone(),
        variant_only_reads.clone(),
        args.error_rate,
        args.lambda1,
        args.lambda2,
        args.mismatches,
        1,   // em_iterations not used for cost calculation
        0.0, // em_convergence_delta not used
        0.0, // sa_max_temperature not used
        original_read_length,
        None, // seed not used
        reads_by_sample,
        Vec::new(), // nucleotide_frequencies not used for cost calculation
        0,          // recombination_points not used for cost calculation
        0,          // operations_per_step not used for cost calculation
        None,       // true_haplotype_sequences not used for cost calculation
    );
    let cost = problem.cost(&variant_only_haplotypes)?;
    println!("Total cost: {}", cost);
    Ok(())
}

/// Run the em subcommand
fn run_em(mut args: EmArgs) -> Result<()> {
    args.files.sort_by(|a, b| natord::compare(a, b));
    let unaligned = unaligned_samples(&args.files)?;
    if !unaligned.is_empty() {
        unaligned
            .iter()
            .for_each(|sample| eprintln!("Sample {sample} is not aligned"));
        exit(1);
    }
    let reads = extract_reads(&args.files);
    let original_read_length = reads[0].sequence.len();
    info!("Original read length: {} nucleotides", original_read_length);
    let (variant_only_reads, invariant_positions) = remove_invariants(&reads);
    info!(
        "Read length after removing {} invariant positions: {} nucleotides",
        invariant_positions.len(),
        variant_only_reads[0].sequence.len()
    );
    let haplotypes = parse_haplotypes_csv(&args.haplotypes_csv, &args.files)?;
    let invariant_indices: HashSet<usize> =
        invariant_positions.iter().map(|(pos, _)| *pos).collect();
    let mut variant_only_haplotypes: Vec<Haplotype> = haplotypes
        .into_iter()
        .map(|h| Haplotype {
            sequence: h
                .sequence
                .iter()
                .enumerate()
                .filter(|(i, _)| !invariant_indices.contains(i))
                .map(|(_, &b)| b)
                .collect(),
            frequencies: h.frequencies,
        })
        .collect();
    // Parse true haplotypes early so we can merge missing ones before EM
    let true_variant_seqs: Option<Vec<Vec<u8>>> =
        if let Some(ref true_csv) = args.true_haplotypes_csv {
            match parse_sequences_from_csv(true_csv) {
                Ok(sequences) => {
                    let seqs: Vec<Vec<u8>> = sequences
                        .into_iter()
                        .map(|seq| {
                            seq.iter()
                                .enumerate()
                                .filter(|(i, _)| !invariant_indices.contains(i))
                                .map(|(_, &b)| b)
                                .collect()
                        })
                        .collect();
                    info!("Loaded {} true haplotypes from {}", seqs.len(), true_csv);
                    Some(seqs)
                }
                Err(e) => {
                    eprintln!("Warning: Failed to parse true haplotypes CSV: {}", e);
                    None
                }
            }
        } else {
            None
        };
    // Add missing true haplotypes with random frequencies, then normalize
    if let Some(ref true_seqs) = true_variant_seqs {
        let existing: AHashSet<Vec<u8>> = variant_only_haplotypes
            .iter()
            .map(|h| h.sequence.clone())
            .collect();
        let missing: Vec<&Vec<u8>> = true_seqs
            .iter()
            .filter(|seq| !existing.contains(*seq))
            .collect();
        let added = missing.len();
        let num_samples = args.files.len();
        let mut rng = thread_rng();
        for seq in missing {
            let random_freqs: Vec<f64> = (0..num_samples).map(|_| rng.gen::<f64>()).collect();
            variant_only_haplotypes.push(Haplotype {
                sequence: seq.clone(),
                frequencies: random_freqs,
            });
        }
        if added > 0 {
            // Normalize frequencies per sample so they sum to 1.0
            for sample_idx in 0..num_samples {
                let sum: f64 = variant_only_haplotypes
                    .iter()
                    .map(|h| h.frequencies[sample_idx])
                    .sum();
                if sum > 0.0 {
                    for h in variant_only_haplotypes.iter_mut() {
                        h.frequencies[sample_idx] /= sum;
                    }
                }
            }
            eprintln!(
                "Added {} missing true haplotypes to input (total now {})",
                added,
                variant_only_haplotypes.len()
            );
        }
    }
    let nucleotide_frequencies = compute_nucleotide_frequencies(&variant_only_reads);
    let mut reads_by_sample: Vec<Vec<usize>> = vec![Vec::new(); args.files.len()];
    for (read_idx, read) in variant_only_reads.iter().enumerate() {
        if let Some(sample_idx) = args.files.iter().position(|s| s == &read.sample) {
            reads_by_sample[sample_idx].push(read_idx);
        }
    }
    let problem = HaplotypeEstimationProblem::new(
        args.files.clone(),
        variant_only_reads.clone(),
        args.error_rate,
        0.0001, // lambda1 not critical for standalone EM
        0.0001, // lambda2 not critical for standalone EM
        args.mismatches,
        args.em_iterations,
        args.em_cdelta,
        0.0, // sa_max_temperature not used
        original_read_length,
        None,
        reads_by_sample,
        nucleotide_frequencies,
        0, // recombination_points not used
        0, // operations_per_step not used
        None,
    );
    let before_count = variant_only_haplotypes.len();
    let mut em_haplotypes = variant_only_haplotypes.clone();
    info!("Running EM on {} haplotypes", before_count);
    // Map each true haplotype (in CSV order) to its index inside `em_haplotypes`.
    // `em_haplotypes` is a clone of `variant_only_haplotypes` *after* missing true
    // haplotypes were merged in, so every true sequence must be findable here.
    let track_indices: Option<Vec<usize>> = true_variant_seqs.as_ref().map(|true_seqs| {
        true_seqs
            .iter()
            .map(|seq| {
                em_haplotypes
                    .iter()
                    .position(|h| &h.sequence == seq)
                    .expect("true haplotype should be present in em_haplotypes after merge")
            })
            .collect()
    });
    let used_squarem = em_haplotypes.len() > 30;
    let em_start = Instant::now();
    let em_result = if used_squarem {
        problem.square_expectation_maximization(
            &mut em_haplotypes,
            args.em_cdelta,
            track_indices.as_deref(),
        )
    } else {
        problem.expectation_maximization(
            &mut em_haplotypes,
            args.em_cdelta,
            track_indices.as_deref(),
        )
    };
    let em_result = match em_result {
        Ok(traces) => traces,
        Err(e) => {
            eprintln!("EM failed: {}", e);
            exit(1);
        }
    };
    let em_elapsed = em_start.elapsed();
    info!(
        "EM ({}) completed in {:.3?} on {} haplotypes",
        if used_squarem {
            "square_expectation_maximization"
        } else {
            "expectation_maximization"
        },
        em_elapsed,
        before_count
    );
    let likelihood_traces = &em_result.likelihoods;
    eprintln!("EM_LIKELIHOOD_START");
    eprintln!("sample,iteration,log_likelihood");
    let mut likelihood_csv = String::from("sample,iteration,log_likelihood\n");
    for (sample, trace) in likelihood_traces {
        for (iter, ll) in trace.iter().enumerate() {
            eprintln!("{},{},{}", sample, iter, ll);
            likelihood_csv.push_str(&format!("{},{},{}\n", sample, iter, ll));
        }
    }
    eprintln!("EM_LIKELIHOOD_END");
    if let Some(ref path) = args.likelihood_output {
        std::fs::write(path, likelihood_csv.as_str())
            .with_context(|| format!("write EM log-likelihood CSV to {path}"))?;
    }
    // Emit per-iteration true-haplotype frequency trace, including a `sum` row that
    // is the sum of all true-haplotype frequencies at that iteration. We *always*
    // write the file when the flag is set (with at least the header row) so it's
    // never ambiguous whether the option was honored. The body is only populated
    // when --true-haplotypes-csv is also provided.
    if let Some(ref path) = args.true_haplotypes_trace_output {
        let mut trace_csv = String::from("sample,iteration,haplotype,frequency\n");
        let mut total_rows = 0usize;
        let mut total_iterations = 0usize;
        if track_indices.is_some() {
            for (sample, freq_trace) in &em_result.frequencies {
                total_iterations += freq_trace.len();
                for (iter, per_iter) in freq_trace.iter().enumerate() {
                    let mut sum = 0.0;
                    for (true_idx, &freq) in per_iter.iter().enumerate() {
                        trace_csv.push_str(&format!(
                            "{},{},true_hap_{},{}\n",
                            sample, iter, true_idx, freq
                        ));
                        sum += freq;
                        total_rows += 1;
                    }
                    trace_csv.push_str(&format!("{},{},sum,{}\n", sample, iter, sum));
                    total_rows += 1;
                }
            }
        }
        std::fs::write(path, trace_csv.as_str())
            .with_context(|| format!("write EM true-haplotype frequency trace CSV to {path}"))?;
        let abs_path = std::fs::canonicalize(path)
            .ok()
            .and_then(|p| p.to_str().map(|s| s.to_string()))
            .unwrap_or_else(|| path.to_string());
        match &track_indices {
            Some(idxs) if !idxs.is_empty() => eprintln!(
                "Wrote true-haplotype frequency trace to {} ({} row(s); {} true \
                 haplotype(s) tracked across {} sample(s), {} total iterations)",
                abs_path,
                total_rows,
                idxs.len(),
                em_result.frequencies.len(),
                total_iterations
            ),
            Some(_) => eprintln!(
                "Wrote header-only true-haplotype frequency trace to {}: the supplied \
                 --true-haplotypes-csv yielded zero true haplotypes after parsing \
                 (check the CSV format / header row)",
                abs_path
            ),
            None => eprintln!(
                "Wrote header-only true-haplotype frequency trace to {}: \
                 --true-haplotypes-csv was not provided, so no true haplotypes \
                 were tracked",
                abs_path
            ),
        }
    }
    let after_sequences: AHashSet<Vec<u8>> =
        em_haplotypes.iter().map(|h| h.sequence.clone()).collect();
    let after_count = em_haplotypes.len();
    let removed: Vec<&Haplotype> = variant_only_haplotypes
        .iter()
        .filter(|h| !after_sequences.contains(&h.sequence))
        .collect();
    eprintln!(
        "EM: {} -> {} haplotypes ({} removed)",
        before_count,
        after_count,
        removed.len()
    );
    if !removed.is_empty() {
        eprintln!("\nRemoved haplotypes:");
        for hap in &removed {
            let restored = restore_invariants(&hap.sequence, &invariant_positions);
            let seq_str = String::from_utf8_lossy(&restored);
            let freqs: Vec<String> = hap
                .frequencies
                .iter()
                .zip(args.files.iter())
                .map(|(f, s)| format!("  {}={:.6}", s, f))
                .collect();
            eprintln!("  {}{}", seq_str, freqs.join(""));
        }
    }
    // Surviving haplotypes to stdout in CSV format
    let output = haplotype_frequencies_output(&em_haplotypes, &invariant_positions, &args.files);
    print!("{}", output);
    if let Some(ref path) = args.output {
        std::fs::write(path, output.as_str())
            .with_context(|| format!("write post-EM haplotypes CSV to {path}"))?;
    }
    // True haplotype survival report
    if let Some(ref true_seqs) = true_variant_seqs {
        let mut true_removed = Vec::new();
        let mut true_kept = Vec::new();
        for seq in true_seqs {
            if after_sequences.contains(seq) {
                true_kept.push(seq);
            } else {
                true_removed.push(seq);
            }
        }
        eprintln!(
            "\nTrue haplotypes: {} total, {} kept, {} removed by EM",
            true_seqs.len(),
            true_kept.len(),
            true_removed.len()
        );
        if !true_removed.is_empty() {
            eprintln!("\nWARNING: EM removed the following TRUE haplotypes:");
            for seq in &true_removed {
                let restored = restore_invariants(seq, &invariant_positions);
                eprintln!("  {}", String::from_utf8_lossy(&restored));
            }
            exit(1);
        } else {
            eprintln!("OK: All true haplotypes survived EM");
        }
    }
    Ok(())
}

/// Main function
fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .init();
    let args = Args::parse();
    if let Some(num_threads) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .expect("Failed to initialize thread pool");
    }
    match args.command {
        Some(Command::Estimate(estimate_args)) => run_estimate(estimate_args),
        Some(Command::Cost(cost_args)) => run_cost(cost_args),
        Some(Command::Em(em_args)) => run_em(em_args),
        None => {
            // Default behavior: run estimate with flattened args
            run_estimate(args.estimate)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    fn create_test_reads(sequences: Vec<&str>, sample: &str) -> Vec<Read> {
        sequences
            .into_iter()
            .map(|seq| {
                let mut sequence = seq.as_bytes().to_vec();
                // Normalize to uppercase to match extract_reads behavior
                for nucleotide in &mut sequence {
                    match nucleotide {
                        b'a' => *nucleotide = b'A',
                        b'c' => *nucleotide = b'C',
                        b'g' => *nucleotide = b'G',
                        b't' => *nucleotide = b'T',
                        _ => {}
                    }
                }
                let non_gap_mask = sequence.iter().map(|&b| u8::from(b != b'-')).collect();
                Read {
                    sequence,
                    sample: sample.to_string(),
                    non_gap_mask,
                }
            })
            .collect()
    }

    fn create_test_haplotypes(sequences: Vec<&str>) -> Vec<Haplotype> {
        sequences
            .into_iter()
            .map(|seq| {
                let mut sequence = seq.as_bytes().to_vec();
                // Normalize to uppercase to match extract_reads behavior
                for nucleotide in &mut sequence {
                    match nucleotide {
                        b'a' => *nucleotide = b'A',
                        b'c' => *nucleotide = b'C',
                        b'g' => *nucleotide = b'G',
                        b't' => *nucleotide = b'T',
                        _ => {}
                    }
                }
                Haplotype {
                    sequence,
                    frequencies: vec![],
                }
            })
            .collect()
    }

    fn create_test_problem() -> HaplotypeEstimationProblem {
        HaplotypeEstimationProblem::new(
            vec![],      // samples
            vec![],      // reads
            0.01,        // error_rate
            1.0,         // lambda1
            1.0,         // lambda2
            3,           // em_max_mismatches
            100,         // em_iterations
            0.001,       // em_convergence_delta
            10.0,        // sa_max_temperature
            100,         // original_read_length
            Some(12345), // seed
            vec![],      // reads_by_sample
            vec![],      // nucleotide_frequencies
            2,           // recombination_points
            3,           // operations_per_step
            None,        // true_haplotype_sequences
        )
    }

    #[test]
    fn test_basic_invariant_removal() {
        let reads = create_test_reads(vec!["AAGTC", "AAATC", "AACTC"], "sample1");
        let result = remove_invariants(&reads);

        for (i, read) in result.0.iter().enumerate() {
            assert_eq!(read.sample, "sample1", "Sample mismatch for read {}", i + 1);
        }
        assert_eq!(result.0[0].sequence, b"G");
        assert_eq!(result.0[1].sequence, b"A");
        assert_eq!(result.0[2].sequence, b"C");
    }

    #[test]
    fn test_all_invariant_sequence() {
        let reads = create_test_reads(vec!["AAAAA", "AAAAA", "AAAAA"], "sample1");
        let result = remove_invariants(&reads);

        for (i, read) in result.0.iter().enumerate() {
            assert!(
                read.sequence.is_empty(),
                "Sequence for read {} should be empty, but got: {:?}",
                i + 1,
                String::from_utf8_lossy(&read.sequence)
            );
        }
    }

    #[test]
    fn test_no_invariants() {
        let reads = create_test_reads(vec!["ACTG", "GCTA", "TGCA"], "sample1");
        let result = remove_invariants(&reads);

        assert_eq!(result.0[0].sequence, b"ACTG");
        assert_eq!(result.0[1].sequence, b"GCTA");
        assert_eq!(result.0[2].sequence, b"TGCA");
    }

    #[test]
    fn test_with_gaps() {
        let reads = create_test_reads(vec!["A-CTG", "A-CTG", "A-CTG"], "sample1");
        let result = remove_invariants(&reads);

        for (i, read) in result.0.iter().enumerate() {
            assert!(
                read.sequence.is_empty(),
                "Sequence for read {} should be empty, but got: {:?}",
                i + 1,
                String::from_utf8_lossy(&read.sequence)
            );
        }
    }

    #[test]
    fn test_mixed_gaps_and_invariants() {
        let reads = create_test_reads(vec!["A-CTA", "A-CTA", "A-GTA"], "sample1");
        let result = remove_invariants(&reads);

        assert_eq!(result.0[0].sequence, b"C",);
        assert_eq!(result.0[1].sequence, b"C",);
        assert_eq!(result.0[2].sequence, b"G",);
    }

    #[test]
    fn test_mixed_gaps_with_single_invariants() {
        let reads = create_test_reads(vec!["-ACTA", "A-CTA", "A-GTA"], "sample1");
        let result = remove_invariants(&reads);

        assert_eq!(result.0[0].sequence, b"C",);
        assert_eq!(result.0[1].sequence, b"C",);
        assert_eq!(result.0[2].sequence, b"G",);
    }

    #[test]
    fn test_single_read() {
        let reads = create_test_reads(vec!["ACGT"], "sample1");
        let result = remove_invariants(&reads);

        assert_eq!(result.0.len(), 1, "Should have exactly one result");
        assert!(
            result.0[0].sequence.is_empty(),
            "Single read sequence should be empty, but got: {:?}",
            String::from_utf8_lossy(&result.0[0].sequence)
        );
    }

    #[test]
    fn test_empty_sequences() {
        let reads = create_test_reads(vec!["", "", ""], "sample1");
        let result = remove_invariants(&reads);

        assert_eq!(result.0.len(), 3, "Should have three results");
        for (i, read) in result.0.iter().enumerate() {
            assert!(
                read.sequence.is_empty(),
                "Sequence {} should be empty, but got: {:?}",
                i + 1,
                String::from_utf8_lossy(&read.sequence)
            );
        }
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_different_sequence_lengths() {
        let mut reads = Vec::new();
        reads.push(Read {
            sequence: b"ACGT".to_vec(),
            sample: "sample1".to_string(),
            non_gap_mask: vec![1; 4],
        });
        reads.push(Read {
            sequence: b"ACG".to_vec(),
            sample: "sample1".to_string(),
            non_gap_mask: vec![1; 3],
        });

        remove_invariants(&reads);
    }

    #[test]
    fn test_preserve_metadata() {
        let reads = vec![
            Read {
                sequence: b"ACGT".to_vec(),
                sample: "sample_A".to_string(),
                non_gap_mask: vec![1; 4],
            },
            Read {
                sequence: b"AGGT".to_vec(),
                sample: "sample_B".to_string(),
                non_gap_mask: vec![1; 4],
            },
        ];
        let result = remove_invariants(&reads);
        assert_eq!(result.0[0].sample, "sample_A",);
        assert_eq!(result.0[1].sample, "sample_B",);
    }

    #[test]
    fn test_large_sequences() {
        let long_seq_a = "A".repeat(1000);
        let long_seq_b = format!("{}T", "A".repeat(999));
        let reads = create_test_reads(vec![&long_seq_a, &long_seq_b], "sample1");
        let result = remove_invariants(&reads);

        assert_eq!(result.0[0].sequence, b"A",);
        assert_eq!(result.0[1].sequence, b"T",);
    }

    #[test]
    fn test_all_gaps() {
        let reads = create_test_reads(vec!["----", "----", "----"], "sample1");
        let result = remove_invariants(&reads);

        for (i, read) in result.0.iter().enumerate() {
            assert!(
                read.sequence.is_empty(),
                "Sequence for read {} should be empty, but got: {:?}",
                i + 1,
                String::from_utf8_lossy(&read.sequence)
            );
        }
    }

    #[test]
    fn test_single_read_no_blanks() {
        // FIXME: If there's no C(for example) in the column, then you don't need to put that in as a possability
        let reads = create_test_reads(vec!["ACGT"], "sample1");
        let haplotypes = init_haplotypes(&reads, &vec!["sample1".to_string()]);

        assert_eq!(haplotypes.len(), 1);
        assert_eq!(haplotypes[0].sequence, b"ACGT");
        assert_eq!(haplotypes[0].frequencies.len(), 1);
        assert!((haplotypes[0].frequencies[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_single_read_with_blanks() {
        let reads = create_test_reads(vec!["A-C"], "sample1");
        let haplotypes = init_haplotypes(&reads, &vec!["sample1".to_string()]);

        // With new MAF-based logic, single read with gaps should create MAF haplotype
        // MAF at position 0: A (1 count), position 1: no valid nucleotides so defaults to T, position 2: C (1 count)
        // So MAF should be "ATC"
        assert_eq!(haplotypes.len(), 1);
        assert_eq!(haplotypes[0].sequence, b"ATC");
        assert_eq!(haplotypes[0].frequencies.len(), 1);
        assert!((haplotypes[0].frequencies[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_multiple_reads_no_blanks() {
        let reads = create_test_reads(vec!["ACGT", "TGCA"], "sample1");
        let haplotypes = init_haplotypes(&reads, &vec!["sample1".to_string()]);

        // With new MAF-based logic:
        // MAF: pos 0: A=1,T=1 -> A (ties go to A), pos 1: C=1,G=1 -> A, pos 2: G=1,C=1 -> A, pos 3: T=1,A=1 -> A
        // So MAF = "AAAA", and then greedy algorithm will add haplotypes for reads that don't match
        // Both "ACGT" and "TGCA" differ from "AAAA", so they should be added as separate haplotypes
        assert!(haplotypes.len() >= 2); // At least MAF + variations

        // Check that we have haplotypes that account for the input reads
        let sequences: HashSet<Vec<u8>> = haplotypes.iter().map(|h| h.sequence.clone()).collect();

        // Should contain some combination that can explain both input reads
        assert!(sequences.len() >= 2);
        assert_eq!(haplotypes[0].frequencies.len(), 1);
    }

    #[test]
    fn test_multiple_reads_with_blanks() {
        let reads = create_test_reads(vec!["A-C", "T-G"], "sample1");
        let haplotypes = init_haplotypes(&reads, &vec!["sample1".to_string()]);

        // With new MAF-based logic:
        // MAF: pos 0: A=1,T=1 -> A (ties go to A), pos 1: gaps ignored, defaults to A, pos 2: C=1,G=1 -> A
        // So MAF = "AAA"
        // Then greedy algorithm processes reads:
        // - "A-C" vs "AAA": differs at pos 2 (C vs A), so creates new haplotype "AAC"
        // - "T-G" vs "AAA" and "AAC": differs from both, so creates new haplotype "AAG"

        assert!(haplotypes.len() >= 2); // At least some haplotypes to explain the diversity
        assert_eq!(haplotypes[0].frequencies.len(), 1);

        // Each haplotype should have uniform initial frequency
        let freq_sum: f64 = haplotypes.iter().map(|h| h.frequencies[0]).sum();
        assert!((freq_sum - 1.0).abs() < 1e-10); // Should sum to 1.0
    }

    #[test]
    fn test_deduplication_of_haplotypes() {
        let reads = create_test_reads(vec!["A-C", "A-C"], "sample1");
        let haplotypes = init_haplotypes(&reads, &vec!["sample1".to_string()]);

        // With new MAF-based logic, identical reads should still create just the MAF haplotype
        // Both reads contribute to MAF calculation: A at pos 0, C at pos 2, default T at pos 1
        assert_eq!(haplotypes.len(), 1);
        assert_eq!(haplotypes[0].sequence, b"ATC");
        assert_eq!(haplotypes[0].frequencies.len(), 1);
        assert!((haplotypes[0].frequencies[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_reads_from_different_samples() {
        let reads = vec![
            Read {
                sequence: b"A-C".to_vec(),
                sample: "sample1".to_string(),
                non_gap_mask: vec![1, 0, 1],
            },
            Read {
                sequence: b"T-G".to_vec(),
                sample: "sample2".to_string(),
                non_gap_mask: vec![1, 0, 1],
            },
        ];
        let haplotypes =
            init_haplotypes(&reads, &vec!["sample1".to_string(), "sample2".to_string()]);

        // With new MAF-based logic, all samples contribute to MAF calculation
        // MAF: pos 0: A=1,T=1 -> A, pos 1: gaps ignored -> A, pos 2: C=1,G=1 -> A
        // So MAF = "AAA"
        // Then greedy algorithm will create haplotypes for reads that don't match MAF

        assert!(haplotypes.len() >= 1);

        // Check that each haplotype has frequencies for both samples
        for haplotype in &haplotypes {
            assert_eq!(haplotype.frequencies.len(), 2);
        }

        // Frequencies should sum to 1.0 for each sample
        let sample1_sum: f64 = haplotypes.iter().map(|h| h.frequencies[0]).sum();
        let sample2_sum: f64 = haplotypes.iter().map(|h| h.frequencies[1]).sum();

        assert!((sample1_sum - 1.0).abs() < 1e-10);
        assert!((sample2_sum - 1.0).abs() < 1e-10);

        // With random frequency initialization, we can no longer check for uniform
        // frequencies. The checks above that ensure the sum is 1.0 are sufficient.
    }

    #[test]
    fn test_large_sequences_with_no_blanks() {
        let long_seq = "A".repeat(1000);
        let reads = create_test_reads(vec![&long_seq], "sample1");
        let haplotypes = init_haplotypes(&reads, &vec!["sample1".to_string()]);

        assert_eq!(haplotypes.len(), 1);
        assert_eq!(haplotypes[0].sequence.len(), 1000);
        assert_eq!(haplotypes[0].sequence, long_seq.as_bytes());
        assert_eq!(haplotypes[0].frequencies.len(), 1);
        assert!((haplotypes[0].frequencies[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    #[ignore]
    fn test_large_sequences_with_blanks() {
        let long_seq_with_blanks = format!("A{}C", "-".repeat(998));
        let reads = create_test_reads(vec![&long_seq_with_blanks], "sample1");
        let haplotypes = init_haplotypes(&reads, &vec!["sample1".to_string()]);

        let expected_count = 4_usize.pow(998); // 998 blanks => 4^998 combinations
        assert_eq!(haplotypes.len(), expected_count);
    }

    #[test]
    fn test_empty_reads() {
        let reads = create_test_reads(vec![], "sample1");
        let haplotypes = init_haplotypes(&reads, &vec!["sample1".to_string()]);

        assert!(
            haplotypes.is_empty(),
            "Haplotypes should be empty for empty reads"
        );
    }

    #[test]
    fn test_no_recombination() {
        let problem = create_test_problem();

        // Only two alleles present - no recombination needed
        let haplotypes = create_test_haplotypes(vec!["A", "A", "C"]);
        assert_eq!(problem.min_recombinations(&haplotypes), 0);
    }

    #[test]
    fn test_single_recombination() {
        let problem = create_test_problem();

        // Three allele combinations require one recombination
        let haplotypes = create_test_haplotypes(vec![
            "AC", // Looking at positions (0,1), we have AC
            "CC", // CC
            "AC", // AC
        ]);
        // With positions (0,1), we have gametes: AC, CC
        // This is only 2 gametes, so should be 0 recombinations
        assert_eq!(problem.min_recombinations(&haplotypes), 0);
    }

    #[test]
    fn test_multiple_recombinations() {
        let problem = create_test_problem();

        let haplotypes = create_test_haplotypes(vec!["AAA", "CCC", "ACC", "AAC"]);
        assert_eq!(problem.min_recombinations(&haplotypes), 2);
    }

    #[test]
    fn test_overlapping_intervals() {
        let problem = create_test_problem();

        let haplotypes = create_test_haplotypes(vec!["AAA", "CGC", "AGC", "AGC", "AGA"]);
        assert_eq!(problem.min_recombinations(&haplotypes), 2);
    }

    #[test]
    fn test_non_acgt_characters() {
        let problem = create_test_problem();

        let haplotypes = create_test_haplotypes(vec!["AA", "CC", "AC", "NN"]);
        assert_eq!(problem.min_recombinations(&haplotypes), 1);
    }

    #[test]
    fn test_empty_or_single_haplotype() {
        let problem = create_test_problem();

        // Empty set
        let empty_haplotypes = Vec::new();
        assert_eq!(problem.min_recombinations(&empty_haplotypes), 0);

        // Single haplotype - can't have recombination with just one sequence
        let single_haplotype = create_test_haplotypes(vec!["A"]);
        assert_eq!(problem.min_recombinations(&single_haplotype), 0);
    }

    #[test]
    fn test_all_possible_gametes() {
        let problem = create_test_problem();

        let haplotypes =
            create_test_haplotypes(vec!["AA", "AC", "AG", "AT", "CA", "CC", "CG", "CT"]);
        assert_eq!(problem.min_recombinations(&haplotypes), 1);
    }

    #[test]
    fn test_complex_recombination_pattern() {
        let problem = create_test_problem();

        // Let's use a simpler but still complex pattern
        let haplotypes = create_test_haplotypes(vec![
            "ACGT", // Looking at adjacent positions, we get:
            "CGTA", // (0,1): AC,CG,CG -> 2 gametes
            "CGTA", // (1,2): CG,GT,GT -> 2 gametes
            "CGTA", // (2,3): GT,TA,TA -> 2 gametes
        ]);
        // Since we need 3+ gametes for recombination, this should be 0
        assert_eq!(problem.min_recombinations(&haplotypes), 0);
    }

    #[test]
    fn test_restore_invariants() {
        // Basic test
        let sequence = b"AC";
        let invariant_positions = vec![(1, b'T')];
        let result = restore_invariants(sequence, &invariant_positions);
        assert_eq!(result, b"ATC");

        // Multiple invariant positions
        let sequence = b"AC";
        let invariant_positions = vec![(1, b'T'), (3, b'G')];
        let result = restore_invariants(sequence, &invariant_positions);
        assert_eq!(result, b"ATCG");

        // Invariant at start
        let sequence = b"AC";
        let invariant_positions = vec![(0, b'T')];
        let result = restore_invariants(sequence, &invariant_positions);
        assert_eq!(result, b"TAC");

        // Invariant at end
        let sequence = b"AC";
        let invariant_positions = vec![(2, b'T')];
        let result = restore_invariants(sequence, &invariant_positions);
        assert_eq!(result, b"ACT");

        // Multiple invariants in middle
        let sequence = b"AC";
        let invariant_positions = vec![(1, b'T'), (2, b'G')];
        let result = restore_invariants(sequence, &invariant_positions);
        assert_eq!(result, b"ATGC");

        // Empty sequence
        let sequence = b"";
        let invariant_positions = vec![(0, b'T')];
        let result = restore_invariants(sequence, &invariant_positions);
        assert_eq!(result, b"T");

        // No invariants
        let sequence = b"AC";
        let invariant_positions = vec![];
        let result = restore_invariants(sequence, &invariant_positions);
        assert_eq!(result, b"AC");

        // Unsorted positions
        let sequence = b"AC";
        let invariant_positions = vec![(2, b'G'), (1, b'T')];
        let result = restore_invariants(sequence, &invariant_positions);
        assert_eq!(result, b"ATGC");
    }

    #[test]
    fn test_case_insensitive_sequence_handling() {
        // Test that sequences with different cases are treated as identical
        let reads = create_test_reads(vec!["acgt", "ACGT", "AcGt", "aCgT"], "sample1");
        let haplotypes = init_haplotypes(&reads, &vec!["sample1".to_string()]);

        // All case variations should produce a single haplotype
        assert_eq!(haplotypes.len(), 1);
        assert_eq!(haplotypes[0].sequence, b"ACGT"); // Normalized to uppercase
        assert_eq!(haplotypes[0].frequencies.len(), 1);
        assert!((haplotypes[0].frequencies[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mixed_case_sequences_with_differences() {
        // Test mixed case sequences that are actually different
        let reads = create_test_reads(vec!["acgt", "ACGT", "atgc", "ATGC"], "sample1");
        let haplotypes = init_haplotypes(&reads, &vec!["sample1".to_string()]);

        // Should produce at least 2 haplotypes (may include MAF haplotype)
        assert!(haplotypes.len() >= 2);

        // Check that both ACGT and ATGC sequences are present (normalized)
        let sequences: HashSet<Vec<u8>> = haplotypes.iter().map(|h| h.sequence.clone()).collect();
        assert!(sequences.contains(&b"ACGT".to_vec()));
        assert!(sequences.contains(&b"ATGC".to_vec()));

        // Check frequencies sum to 1.0
        let total_freq: f64 = haplotypes.iter().map(|h| h.frequencies[0]).sum();
        assert!((total_freq - 1.0).abs() < 1e-10);
    }
}
