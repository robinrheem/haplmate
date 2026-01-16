use ahash::AHashSet;
use anyhow::Result;
use argmin::core::{CostFunction, Executor};
use argmin::solver::simulatedannealing::{Anneal, SATempFunc, SimulatedAnnealing};
use rand::prelude::*;
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use seq_io::fasta::{Reader, Record};
use statrs::distribution::{Binomial, Discrete};
use std::collections::HashSet;
use std::process::exit;
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

#[derive(Debug, Clone)]
struct Read {
    sequence: Vec<u8>,
    sample: String,
}

#[derive(Debug, Clone)]
struct Haplotype {
    sequence: Vec<u8>,
    frequencies: Vec<f64>,
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
            if let Some(&nucleotide) = unique_nucleotides.iter().next() {
                invariant_positions.push((i, nucleotide));
            }
            continue;
        }
        for (j, c) in column.into_iter().enumerate() {
            filtered_sequences[j].push(c);
        }
    }
    let filtered_reads = reads
        .iter()
        .enumerate()
        .map(|(i, read)| Read {
            sequence: filtered_sequences[i].clone(),
            sample: read.sample.clone(),
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
                reads.push(Read {
                    sequence,
                    sample: sample.to_string(),
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
    // Pre-computed binomial PMF lookup table: mismatch_prob_table[num_mismatches] = probability
    mismatch_prob_table: Vec<f64>,
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
    ) -> Self {
        // Pre-compute binomial PMF lookup table (like legacy C code's initialize_Mismatch)
        let mismatch_prob_table = match Binomial::new(error_rate, original_read_length as u64) {
            Ok(binomial) => (0..=em_max_mismatches)
                .map(|m| binomial.pmf(m as u64))
                .collect(),
            Err(_) => {
                eprintln!("Failed to create binomial distribution");
                vec![0.0; em_max_mismatches + 1]
            }
        };
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
        }
    }

    /// Looks up the pre-computed probability for a given mismatch count.
    #[inline]
    fn mismatch_probability(&self, mismatches: usize) -> f64 {
        if mismatches > self.em_max_mismatches {
            return 0.0;
        }
        self.mismatch_prob_table[mismatches]
    }

    /// Calculate mismatch probability between a read and haplotype.
    /// Optimized with early stopping when mismatches exceed threshold.
    fn compute_mismatch_probability(&self, read: &Read, haplotype: &Haplotype) -> f64 {
        // Early stopping optimization: stop counting when we exceed max_mismatches
        let mut mismatches = 0;
        for (&r, &h) in read.sequence.iter().zip(&haplotype.sequence) {
            if r != h && r != b'-' {
                mismatches += 1;
                // Early exit if we've already exceeded the threshold
                if mismatches > self.em_max_mismatches {
                    return 0.0;
                }
            }
        }
        self.mismatch_probability(mismatches)
    }

    /// Pre-compute mismatch probability matrix for all reads against all haplotypes.
    /// Returns a Vec where mismatch_matrix[read_idx][hap_idx] = probability.
    /// This avoids synchronization overhead from caching in tight parallel loops.
    fn compute_mismatch_matrix(&self, haplotypes: &[Haplotype]) -> Vec<Vec<f64>> {
        self.reads
            .par_iter()
            .map(|read| {
                haplotypes
                    .iter()
                    .map(|hap| self.compute_mismatch_probability(read, hap))
                    .collect()
            })
            .collect()
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
    ) -> Result<(), anyhow::Error> {
        let num_haps = haplotypes.len();
        let mismatch_matrix = self.compute_mismatch_matrix(haplotypes);
        // Process all samples in parallel - each sample is completely independent
        // Collect frequency results: Vec<Vec<f64>> where outer = samples, inner = haplotype freqs
        let sample_frequencies: Vec<Vec<f64>> = self
            .samples
            .par_iter()
            .enumerate()
            .map(|(sample_idx, _sample)| {
                let sample_read_indices = &self.reads_by_sample[sample_idx];
                let num_reads = sample_read_indices.len();
                if num_haps == 1 {
                    return vec![1.0];
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
                    .map(|&read_idx| mismatch_matrix[read_idx].as_slice())
                    .collect();
                // Initialize mismatch_fp_new = mismatches * theta (like mismatchesFP_new in C)
                let mut mismatch_fp_new: Vec<Vec<f64>> = mismatches
                    .iter()
                    .map(|row| row.iter().zip(&theta_new).map(|(&m, &t)| m * t).collect())
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

                // Create intermediate matrices for SQUAREM
                let mut mismatch_fp_1 = vec![vec![0.0; num_haps]; num_reads];
                let mut mismatch_fp_2 = vec![vec![0.0; num_haps]; num_reads];

                // Hoist memberships allocation to avoid re-allocation in every EM update
                let mut memberships = vec![vec![0.0; num_haps]; num_reads];
                // EM update closure - equivalent to sqEMUpdate in C
                let mut em_update =
                    |mismatch_fp_in: &Vec<Vec<f64>>,
                     theta_out: &mut [f64],
                     mismatch_fp_out: &mut Vec<Vec<f64>>| {
                        // E-step: Calculate memberships (normalized probabilities)
                        for i in 0..num_reads {
                            let denom: f64 = mismatch_fp_in[i].iter().sum();
                            if denom > 0.0 {
                                for j in 0..num_haps {
                                    memberships[i][j] = mismatch_fp_in[i][j] / denom;
                                }
                            } else {
                                // clear row if denom is 0 to avoid using stale values
                                memberships[i].iter_mut().for_each(|m| *m = 0.0);
                            }
                        }
                        // M-step: Update frequencies based on memberships
                        for j in 0..num_haps {
                            theta_out[j] = 0.0;
                            for i in 0..num_reads {
                                theta_out[j] += memberships[i][j];
                            }
                            theta_out[j] /= num_reads as f64;
                            // Update mismatch_fp with new frequencies
                            for i in 0..num_reads {
                                mismatch_fp_out[i][j] = mismatches[i][j] * theta_out[j];
                            }
                        }
                    };
                // Calculate likelihood closure - equivalent to EM_likelihood_sq in C
                let calculate_likelihood = |mismatch_fp: &Vec<Vec<f64>>| -> f64 {
                    let mut likelihood = 0.0;
                    for i in 0..num_reads {
                        let row_sum: f64 = mismatch_fp[i].iter().sum();
                        if row_sum > 0.0 {
                            likelihood += row_sum.ln();
                        }
                    }
                    likelihood
                };
                // Initial likelihood calculation
                let mut likelihood_old = calculate_likelihood(&mismatch_fp_new);
                let mut likelihood_new = likelihood_old;
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
                        // Parameter projection
                        theta_new[j] = f64::max(0.01, theta_new[j]);
                    }
                    // Renormalize to the simplex after projection to keep a valid mixture
                    let sum_theta: f64 = theta_new.iter().sum();
                    if sum_theta > 0.0 {
                        for val in theta_new.iter_mut() {
                            *val /= sum_theta;
                        }
                    }
                    // Recompute mismatch_fp_new using the normalized theta
                    for i in 0..num_reads {
                        for j in 0..num_haps {
                            mismatch_fp_new[i][j] = mismatches[i][j] * theta_new[j];
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
                    // Check for convergence
                    if (likelihood_new - likelihood_old).abs() <= self.em_convergence_delta {
                        break;
                    }
                }
                // Return computed frequencies for this sample
                theta_new
            })
            .collect();

        // Write back all frequencies sequentially (no contention, safe)
        for (sample_idx, freqs) in sample_frequencies.iter().enumerate() {
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
        // Log haplotypes being removed
        for &idx in &indices_to_remove {
            let haplotype = &haplotypes[idx];
            let freq_str: Vec<String> = self
                .samples
                .iter()
                .enumerate()
                .map(|(s_idx, sample)| format!("{}:{:.6}", sample, haplotype.frequencies[s_idx]))
                .collect();
            trace!(
                "Removing haplotype {}: sequence={}, frequencies=[{}]",
                idx,
                String::from_utf8_lossy(&haplotype.sequence),
                freq_str.join(", ")
            );
        }
        // Remove haplotypes in reverse order to maintain correct indices
        for &idx in indices_to_remove.iter().rev() {
            haplotypes.remove(idx);
        }
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
        Ok(())
    }

    fn expectation_maximization(
        &self,
        haplotypes: &mut Vec<Haplotype>,
        convergence_delta: f64,
    ) -> Result<(), anyhow::Error> {
        let num_haps = haplotypes.len();
        let mismatch_matrix = self.compute_mismatch_matrix(haplotypes);
        // Process all samples in parallel - each sample is completely independent
        // Collect frequency results: Vec<Vec<f64>> where outer = samples, inner = haplotype freqs
        let sample_frequencies: Vec<Vec<f64>> = self
            .samples
            .par_iter()
            .enumerate()
            .map(|(sample_idx, _sample)| {
                let sample_read_indices = &self.reads_by_sample[sample_idx];
                let num_reads = sample_read_indices.len();
                let calculate_likelihood = |mismatch_fp: &Vec<Vec<f64>>| -> f64 {
                    let mut likelihood = 0.0;
                    for i in 0..num_reads {
                        let row_sum: f64 = mismatch_fp[i].iter().sum();
                        if row_sum > 0.0 {
                            likelihood += row_sum.ln();
                        }
                    }
                    likelihood
                };
                if num_haps == 1 {
                    return vec![1.0];
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
                    .map(|&read_idx| mismatch_matrix[read_idx].as_slice())
                    .collect();
                // Initialize mismatch_fp_new = mismatches * theta (like mismatchesFP_new in C)
                let mut mismatch_fp_new: Vec<Vec<f64>> = mismatches
                    .iter()
                    .map(|row| row.iter().zip(&theta).map(|(&m, &t)| m * t).collect())
                    .collect();
                // Calculate initial likelihood
                let mut likelihood_old = calculate_likelihood(&mismatch_fp_new);
                let mut iters = 0;
                // Hoist memberships allocation to avoid re-allocation in every EM iteration
                let mut memberships = vec![vec![0.0; num_haps]; num_reads];
                // Main EM loop
                while iters < self.em_iterations {
                    let theta_old = theta.clone();
                    // E-step: Calculate memberships (normalized probabilities)
                    for i in 0..num_reads {
                        let denom: f64 = mismatch_fp_new[i].iter().sum();
                        if denom > 0.0 {
                            for j in 0..num_haps {
                                memberships[i][j] = mismatch_fp_new[i][j] / denom;
                            }
                        } else {
                            // Clear row if denom is 0 to avoid using stale values
                            memberships[i].iter_mut().for_each(|m| *m = 0.0);
                        }
                    }
                    // M-step: Update frequencies based on memberships
                    for j in 0..num_haps {
                        theta[j] = 0.0;
                        for i in 0..num_reads {
                            theta[j] += memberships[i][j];
                        }
                        theta[j] /= num_reads as f64;

                        // Ensure minimum probability
                        theta[j] = f64::max(1e-10, theta[j]);
                        for i in 0..num_reads {
                            mismatch_fp_new[i][j] = mismatches[i][j] * theta[j];
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
                    // Check for convergence
                    let converged = if likelihood_old.abs() > 1e-10 {
                        // Use relative convergence criterion
                        ((likelihood_new - likelihood_old) / likelihood_old.abs()).abs()
                            < convergence_delta
                    } else {
                        // Use absolute convergence criterion for small likelihoods
                        (likelihood_new - likelihood_old).abs() < convergence_delta
                    };
                    if converged {
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
                // Return computed frequencies for this sample
                theta
            })
            .collect();
        // Write back all frequencies sequentially (no contention, safe)
        for (sample_idx, freqs) in sample_frequencies.iter().enumerate() {
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
        // Log haplotypes being removed
        for &idx in &indices_to_remove {
            let haplotype = &haplotypes[idx];
            let freq_str: Vec<String> = self
                .samples
                .iter()
                .enumerate()
                .map(|(s_idx, sample)| format!("{}:{:.6}", sample, haplotype.frequencies[s_idx]))
                .collect();
            trace!(
                "Removing haplotype {}: sequence={}, frequencies=[{}]",
                idx,
                String::from_utf8_lossy(&haplotype.sequence),
                freq_str.join(", ")
            );
        }
        // Remove haplotypes in reverse order to maintain correct indices
        for &idx in indices_to_remove.iter().rev() {
            haplotypes.remove(idx);
        }
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
        Ok(())
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
        let length = haplotypes[0].sequence.len();
        // Matrix of possible gamete pairs for ACTG (4x4)
        let mut gamete_counts = [[0; 4]; 4];
        // Index is start, value is end of interval
        let mut interval_list = vec![-1i32; length];
        // Create rough intervals - list positions with recombinant gamete pairs
        'outer: for pos1 in 0..length {
            for pos2 in (pos1 + 1)..length {
                // Reset gamete counts for this position pair
                for row in gamete_counts.iter_mut() {
                    row.fill(0);
                }
                // Count gamete pairs at these positions
                for haplotype in haplotypes {
                    let nuc1 = haplotype.sequence[pos1];
                    let nuc2 = haplotype.sequence[pos2];
                    let (i, j) = match (nuc1, nuc2) {
                        (b'A', b'A') => (0, 0),
                        (b'A', b'C') => (0, 1),
                        (b'A', b'G') => (0, 2),
                        (b'A', b'T') => (0, 3),
                        (b'C', b'A') => (1, 0),
                        (b'C', b'C') => (1, 1),
                        (b'C', b'G') => (1, 2),
                        (b'C', b'T') => (1, 3),
                        (b'G', b'A') => (2, 0),
                        (b'G', b'C') => (2, 1),
                        (b'G', b'G') => (2, 2),
                        (b'G', b'T') => (2, 3),
                        (b'T', b'A') => (3, 0),
                        (b'T', b'C') => (3, 1),
                        (b'T', b'G') => (3, 2),
                        (b'T', b'T') => (3, 3),
                        _ => continue, // Skip non-ACGT characters
                    };
                    gamete_counts[i][j] = 1;
                }
                // Count number of gametes
                let mut num_gametes = 0;
                for row in &gamete_counts {
                    for &count in row {
                        num_gametes += count;
                    }
                }
                // If we found 4 gametes, record this interval
                if num_gametes >= 3 {
                    interval_list[pos1] = pos2 as i32;
                    continue 'outer;
                }
            }
        }
        // Trim intervals
        for pos1 in 0..length {
            if interval_list[pos1] == -1 {
                continue;
            }
            for pos2 in 0..length {
                if interval_list[pos2] == -1 || pos2 == pos1 {
                    continue;
                }
                // Remove completely overlapped intervals
                else if pos2 <= pos1 && interval_list[pos1] <= interval_list[pos2] {
                    interval_list[pos2] = -1;
                }
                // Remove intervals that start within another interval
                else if pos1 < pos2 && pos2 < interval_list[pos1] as usize {
                    interval_list[pos2] = -1;
                }
            }
        }
        // Count number of remaining intervals/recombinations
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
    fn random_operation(&self, haplotypes: &mut Vec<Haplotype>, rng: &mut impl Rng) -> bool {
        // Determine which operation to perform based on current state
        let operation: i32 = if haplotypes.len() == 1 {
            debug!("Only one haplotype present, forcing add operation");
            2 // Force add operation for single haplotype
        } else {
            rng.gen_range(0..3)
        };

        match operation {
            0 if haplotypes.len() > 1 => {
                // Delete a random haplotype
                let idx_to_remove = rng.gen_range(0..haplotypes.len());
                debug!(
                    "Operation: Delete - Removing haplotype at index {}",
                    idx_to_remove
                );
                haplotypes.remove(idx_to_remove);
                true
            }
            1 if haplotypes.len() >= 2 => {
                // Recombine two random haplotypes
                debug!("Operation: Recombine");
                self.recombine(haplotypes, rng);
                true
            }
            2 if haplotypes.len() < self.reads.len() => {
                // Add a new haplotype by mutating an existing one
                debug!("Operation: Add new haplotype by mutation");
                self.mutate(haplotypes, rng);
                true
            }
            _ => {
                trace!("No operation performed - conditions not met");
                false
            }
        }
    }

    /// Applies recombination operation between two random haplotypes
    fn recombine(&self, haplotypes: &mut Vec<Haplotype>, rng: &mut impl Rng) {
        let idx1 = rng.gen_range(0..haplotypes.len());
        let mut idx2 = rng.gen_range(0..haplotypes.len());
        let mut attempts = 0;
        const MAX_ATTEMPTS: i32 = 100;

        trace!("Initial recombination pair: indices {} and {}", idx1, idx2);

        // Try to find compatible haplotypes for recombination
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

            let crossover_point = rng.gen_range(0..haplotypes[idx1].sequence.len());
            debug!(
                "Performing recombination at position {} between haplotypes {} and {}",
                crossover_point, idx1, idx2
            );

            let mut recombined1 = haplotypes[idx1].sequence.clone();
            let mut recombined2 = haplotypes[idx2].sequence.clone();
            recombined1[crossover_point..]
                .copy_from_slice(&haplotypes[idx2].sequence[crossover_point..]);
            recombined2[crossover_point..]
                .copy_from_slice(&haplotypes[idx1].sequence[crossover_point..]);

            let mut new_sequences = Vec::new();
            if !haplotypes.iter().any(|h| h.sequence == recombined1) {
                trace!("Adding first recombined sequence");
                new_sequences.push(recombined1);
            }
            if !haplotypes.iter().any(|h| h.sequence == recombined2) {
                trace!("Adding second recombined sequence");
                new_sequences.push(recombined2);
            }
            debug!("Generated {} new unique sequences", new_sequences.len());
            // Must generate exactly 2 new sequences, otherwise retry
            if new_sequences.len() != 2 {
                trace!("Did not generate 2 new sequences, retrying with different indices");
                idx2 = rng.gen_range(0..haplotypes.len());
                attempts += 1;
                continue;
            }
            for new_seq in new_sequences {
                let mut combined_frequencies = vec![0.0; self.samples.len()];
                for s in 0..self.samples.len() {
                    let freq1 = *haplotypes[idx1].frequencies.get(s).unwrap_or(&0.0);
                    let freq2 = *haplotypes[idx2].frequencies.get(s).unwrap_or(&0.0);
                    combined_frequencies[s] = (freq1 + freq2) / 2.0;
                }
                haplotypes.push(Haplotype {
                    sequence: new_seq,
                    frequencies: combined_frequencies,
                });
            }
            break;
        }
    }

    /// Applies mutation operation to create a new haplotype
    fn mutate(&self, haplotypes: &mut Vec<Haplotype>, rng: &mut impl Rng) {
        let idx_to_copy = rng.gen_range(0..haplotypes.len());
        let mut attempts = 0;
        const MAX_ATTEMPTS: usize = 100;
        loop {
            if attempts >= MAX_ATTEMPTS {
                debug!(
                    "Failed to generate unique mutated haplotype after {} attempts",
                    MAX_ATTEMPTS
                );
                break;
            }
            let mut new_sequence = haplotypes[idx_to_copy].sequence.clone();
            let pos_to_change = rng.gen_range(0..new_sequence.len());
            let new_nucleotide = [b'A', b'C', b'G', b'T'][rng.gen_range(0..4)];
            trace!(
                "Mutating haplotype {} at position {} to {} (attempt {})",
                idx_to_copy,
                pos_to_change,
                new_nucleotide as char,
                attempts + 1
            );
            new_sequence[pos_to_change] = new_nucleotide;
            // Only add if this sequence doesn't already exist
            if !haplotypes.iter().any(|h| h.sequence == new_sequence) {
                debug!("Adding new mutated haplotype");
                // Halve the frequencies for the original haplotype
                for freq in &mut haplotypes[idx_to_copy].frequencies {
                    *freq /= 2.0;
                }
                let new_freqs = haplotypes[idx_to_copy].frequencies.clone();
                haplotypes.push(Haplotype {
                    sequence: new_sequence,
                    frequencies: new_freqs,
                });
                break;
            } else {
                trace!("Mutated sequence already exists, trying again");
                attempts += 1;
            }
        }
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
    /// - Uses binomial probability model for mismatches
    /// - Only considers haplotypes from matching sample when calculating read probabilities
    /// - Higher costs indicate worse solutions
    fn cost(&self, haplotypes: &Self::Param) -> std::result::Result<Self::Output, anyhow::Error> {
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
                    for (hap_idx, haplotype) in haplotypes.iter().enumerate() {
                        let probability = mismatch_matrix[read_idx][hap_idx];
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
        let total_cost = total_cost + self.lambda1 * self.min_recombinations(haplotypes) as f64;
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
    /// This function implements three possible operations, chosen randomly:
    /// 1. Delete a random haplotype (if there are at least 2 haplotypes)
    /// 2. Recombine two random haplotypes by performing a crossover (if there are at least 2 haplotypes)
    /// 3. Add a new haplotype by mutating an existing one (if number of haplotypes < number of reads)
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
        debug!("Starting annealing step with temperature {}", temp);
        if param.is_empty() {
            debug!("No haplotypes available for annealing operations, returning original set");
            return Ok(param.clone());
        }
        // Calculate EM convergence parameters
        let em_temp_end = 0.00001;
        let sa_progress = temp / self.sa_max_temperature;
        let convergence_delta =
            em_temp_end + (self.em_convergence_delta - em_temp_end) * sa_progress;
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
            // Apply random operation
            if !self.random_operation(&mut haplotypes, &mut rng) {
                debug!(
                    "No operation could be applied on attempt {}",
                    retry_count + 1
                );
                if retry_count == MAX_RETRIES {
                    break;
                }
                continue;
            }
            debug!("Running EM optimization on {} haplotypes", haplotypes.len());
            self.expectation_maximization(&mut haplotypes, convergence_delta)?;
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
    );
    info!(
        "Estimating haplotypes with parameters: samples={}, reads={}, error_rate={}, lambda1={}, lambda2={}, em_max_mismatches={}, em_iterations={}, em_convergence_delta={}, sa_max_temperature={}, sa_iterations={}, sa_reruns={}, original_read_length={}, seed={:?}",
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
        optimization_parameters.seed
    );
    let rng = if let Some(seed) = optimization_parameters.seed {
        rand::rngs::StdRng::seed_from_u64(seed)
    } else {
        rand::rngs::StdRng::from_entropy()
    };
    let solver = SimulatedAnnealing::new_with_rng(optimization_parameters.sa_max_temperature, rng)
        .unwrap()
        .with_temp_func(SATempFunc::Boltzmann)
        .with_reannealing_fixed(optimization_parameters.sa_reannealing_fixed)
        .with_reannealing_accepted(optimization_parameters.sa_reannealing_accepted)
        .with_reannealing_best(optimization_parameters.sa_reannealing_best)
        .with_stall_accepted(optimization_parameters.sa_stall_accepted)
        .with_stall_best(optimization_parameters.sa_stall_best);
    // Optimize initial haplotypes with EM before starting SA
    let mut best_haplotypes = initial_haplotypes.clone();
    info!(
        "Running EM optimization on initial {} haplotypes",
        best_haplotypes.len()
    );
    // Calculate EM convergence parameters
    let em_temp_end = 0.00001;
    let sa_progress = optimization_parameters.sa_max_temperature;
    let convergence_delta =
        em_temp_end + (optimization_parameters.em_cdelta - em_temp_end) * sa_progress;
    if let Err(e) = problem.square_expectation_maximization(&mut best_haplotypes, convergence_delta)
    {
        info!(
            "EM optimization failed: {}, proceeding with unoptimized haplotypes",
            e
        );
    }
    info!(
        "EM optimization completed. {} haplotypes remain",
        best_haplotypes.len()
    );
    let mut best_objective = f64::INFINITY;
    let mut best_likelihood = f64::INFINITY;
    for i in 0..optimization_parameters.sa_reruns {
        info!(
            "Running SA with {} haplotypes, iteration {}",
            best_haplotypes.len(),
            i
        );
        let result = Executor::new(problem.clone(), solver.clone())
            .configure(|state| state.param(best_haplotypes.clone()))
            .run()
            .unwrap();
        let best_cost = result.state().best_cost;
        if best_cost < best_objective {
            if let Some(ref param) = result.state().best_param {
                best_haplotypes = param.clone();
                best_objective = best_cost;
                info!("New best haplotypes: {}", best_haplotypes.len());
                info!("New best objective: {}", best_objective);
            }
        }
        // Track best likelihood across all runs
        if best_cost < best_likelihood {
            best_likelihood = best_cost;
            info!("New global best likelihood: {}", best_likelihood);
        }
    }
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
    let (variant_only_reads, invariant_positions) = remove_invariants(&reads);
    let initial_haplotypes = init_haplotypes(&variant_only_reads, &args.files);
    if initial_haplotypes.len() == 1 && initial_haplotypes[0].sequence.is_empty() {
        eprintln!("No initial haplotypes that have meaningful information");
        exit(1);
    }
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
    let (variant_only_reads, invariant_positions) = remove_invariants(&reads);

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
    );
    let cost = problem.cost(&variant_only_haplotypes)?;
    println!("Total cost: {}", cost);
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
                Read {
                    sequence,
                    sample: sample.to_string(),
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
        });
        reads.push(Read {
            sequence: b"ACG".to_vec(),
            sample: "sample1".to_string(),
        });

        remove_invariants(&reads);
    }

    #[test]
    fn test_preserve_metadata() {
        let reads = vec![
            Read {
                sequence: b"ACGT".to_vec(),
                sample: "sample_A".to_string(),
            },
            Read {
                sequence: b"AGGT".to_vec(),
                sample: "sample_B".to_string(),
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
            },
            Read {
                sequence: b"T-G".to_vec(),
                sample: "sample2".to_string(),
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
