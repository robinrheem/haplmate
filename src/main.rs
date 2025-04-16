use anyhow::Result;
use argmin::core::{CostFunction, Executor};
use argmin::solver::simulatedannealing::{Anneal, SATempFunc, SimulatedAnnealing};
use rand::prelude::*;
use seq_io::fasta::{Reader, Record};
use statrs::distribution::{Binomial, Discrete};
use std::collections::{HashMap, HashSet, VecDeque};
use std::process::exit;
use tracing::{debug, info, trace};
use tracing_subscriber;

use clap::Parser;
use sprs::indexing::SpIndex;
use sprs::{CsMat, TriMat};

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
    #[arg(short = 'd', long, default_value = "0.04")]
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
    /// Random seed for deterministic output(testing purposes only)
    #[arg(long)]
    seed: Option<u64>,
}

#[derive(Debug, Clone)]
struct Read {
    sequence: Vec<u8>,
    sample: String,
}

#[derive(Debug, Clone)]
struct Haplotype {
    sequence: Vec<u8>,
    frequencies: HashMap<String, f64>,
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
    seed: Option<u64>,
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
                reads.push(Read {
                    sequence: record.seq().to_vec(),
                    sample: sample.to_string(),
                });
            });
    });
    reads
}

/// Propose initial haplotype set
/// All combinations that can happen from read information
///
/// # Arguments
///
/// * `reads` - A list of reads from samples
///
/// # Returns
///
/// List of haplotypes(full sequences) with initial frequencies
fn init_haplotypes(reads: &Vec<Read>) -> Vec<Haplotype> {
    info!("Initializing haplotypes from {} reads", reads.len());
    // Track unique sequences only (without sample information)
    let mut sequence_set: HashSet<Vec<u8>> = HashSet::new();
    let mut sequence_counts: HashMap<Vec<u8>, HashMap<String, usize>> = HashMap::new();
    let samples: HashSet<String> = reads.iter().map(|r| r.sample.clone()).collect();
    debug!("Found {} unique samples", samples.len());
    // First pass - collect all possible sequences and count their occurrences per sample
    for (read_idx, read) in reads.iter().enumerate() {
        let mut queue: VecDeque<Vec<u8>> = VecDeque::new();
        queue.push_back(vec![]);
        // Expand blanks iteratively
        for (pos, &nucleotide) in read.sequence.iter().enumerate() {
            let mut level_size = queue.len();
            trace!("Position {}: expanding {} sequences", pos, level_size);

            while level_size > 0 {
                level_size -= 1;
                let mut current = queue.pop_front().unwrap();
                if nucleotide == b'-' {
                    for &fill in b"ACGT" {
                        let mut next = current.clone();
                        next.push(fill);
                        queue.push_back(next);
                    }
                } else {
                    current.push(nucleotide);
                    queue.push_back(current);
                }
            }
        }
        debug!(
            "Read {}: generated {} possible sequences",
            read_idx + 1,
            queue.len()
        );
        // Add sequences to the set and update counts
        for sequence in queue {
            sequence_set.insert(sequence.clone());
            sequence_counts
                .entry(sequence)
                .or_default()
                .entry(read.sample.clone())
                .and_modify(|count| *count += 1)
                .or_insert(1);
        }
    }
    info!(
        "Found {} unique sequences across all samples",
        sequence_set.len()
    );
    // Calculate total expansions per sample once before processing individual sequences
    let mut total_expansions_per_sample: HashMap<String, f64> = HashMap::new();
    for sample in &samples {
        let total: f64 = sequence_counts
            .values()
            .map(|sample_counts| sample_counts.get(sample).copied().unwrap_or(0) as f64)
            .sum();
        total_expansions_per_sample.insert(sample.clone(), total);
    }
    // Convert to haplotypes with frequencies
    let haplotypes: Vec<Haplotype> = sequence_set
        .into_iter()
        .map(|sequence| {
            let mut frequencies = HashMap::new();
            let counts = sequence_counts.get(&sequence).unwrap();
            // Calculate frequencies for each sample using the pre-calculated totals
            for sample in &samples {
                let sample_count = counts.get(sample).copied().unwrap_or(0) as f64;
                let total_expansions: f64 =
                    *total_expansions_per_sample.get(sample).unwrap_or(&0.0);
                frequencies.insert(
                    sample.clone(),
                    if total_expansions > 0.0 {
                        sample_count / total_expansions
                    } else {
                        0.0
                    },
                );
            }
            Haplotype {
                sequence,
                frequencies,
            }
        })
        .collect();
    debug!(
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
    error_rate: f64,
    lambda1: f64,
    lambda2: f64,
    em_max_mismatches: usize,
    em_iterations: usize,
    em_convergence_delta: f64,
    seed: Option<u64>,
}

impl HaplotypeEstimationProblem {
    /// Calculates the probability of observing a given number of mismatches in a sequence
    /// using a binomial distribution model.
    ///
    /// # Arguments
    ///
    /// * `mismatches` - Number of mismatches observed between a read and haplotype
    /// * `sequence_length` - Length of the sequence being compared
    ///
    /// # Returns
    ///
    /// The probability of observing exactly `mismatches` number of mismatches in a sequence
    /// of length `sequence_length`, given the error rate. Returns 0.0 if mismatches exceed
    /// the maximum allowed mismatches or if creating the binomial distribution fails.
    fn mismatch_probability(&self, mismatches: usize, sequence_length: usize) -> f64 {
        if mismatches > self.em_max_mismatches {
            return 0.0;
        }
        match Binomial::new(self.error_rate, sequence_length as u64) {
            Ok(binomial) => binomial.pmf(mismatches as u64),
            Err(_) => {
                eprintln!("Failed to create binomial distribution");
                return 0.0;
            }
        }
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
    ///   - frequencies: HashMap mapping sample IDs to frequency estimates
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
    ) -> Result<(), anyhow::Error> {
        // Create a cache for mismatch counts between reads and haplotypes
        let mut mismatch_cache: HashMap<(usize, usize), usize> = HashMap::new();
        // Helper function to calculate likelihood from sparse matrices
        let calculate_likelihood = |mismatch_fp_mat: &CsMat<f64>, num_reads: usize| -> f64 {
            let mut likelihood = 0.0;
            for i in 0..num_reads {
                let row_sum: f64 = mismatch_fp_mat
                    .outer_view(i)
                    .unwrap()
                    .iter()
                    .map(|(_, &val)| val)
                    .sum();
                if row_sum > 0.0 {
                    likelihood += row_sum.ln();
                }
            }
            likelihood
        };
        for sample in &self.samples {
            let sample_reads: Vec<&Read> =
                self.reads.iter().filter(|r| r.sample == *sample).collect();
            let num_reads = sample_reads.len();
            let num_haps = haplotypes.len();
            // Initialize theta vectors (these remain dense since they're small)
            let mut theta_old = vec![1.0 / num_haps as f64; num_haps];
            let mut theta_new = vec![0.0; num_haps];
            let mut theta_2 = vec![0.0; num_haps]; // Added for square EM

            // Create sparse matrices instead of dense ones
            let mut mismatches_triplet = TriMat::new((num_reads, num_haps));
            // Calculate initial mismatches
            for (i, read) in sample_reads.iter().enumerate() {
                for (j, haplotype) in haplotypes.iter().enumerate() {
                    // Use cached mismatch count or calculate and cache it
                    let mismatch_count = *mismatch_cache.entry((i, j)).or_insert_with(|| {
                        read.sequence
                            .iter()
                            .zip(&haplotype.sequence)
                            .filter(|(&r, &h)| r != h && r != b'-')
                            .count()
                    });

                    let prob = self.mismatch_probability(mismatch_count, read.sequence.len());
                    if prob > 0.0 {
                        mismatches_triplet.add_triplet(i, j, prob);
                    }
                }
            }
            // Convert to CSR format for efficient row operations
            let mismatches_mat = mismatches_triplet.to_csr::<usize>();
            // Initialize mismatch_fp matrices
            let mut mismatch_fp_old_triplet = TriMat::new((num_reads, num_haps));
            // Compute initial mismatch_fp values
            for i in 0..num_reads {
                if let Some(row) = mismatches_mat.outer_view(i) {
                    for (j, &val) in row.iter() {
                        mismatch_fp_old_triplet.add_triplet(
                            i,
                            j.index(),
                            val * theta_old[j.index()],
                        );
                    }
                }
            }
            let mut mismatch_fp_old = mismatch_fp_old_triplet.to_csr::<usize>();
            let mut mismatch_fp_new: CsMat<f64>;
            let mut mismatch_fp_2: CsMat<f64> = CsMat::zero((num_reads, num_haps));
            let mut likelihood_old = 0.0;
            let mut likelihood_new;
            // EM iterations
            for _ in 0..self.em_iterations {
                // Create a new TriMat for E-step since we can't clear existing ones
                let mut mismatch_fp_new_triplet = TriMat::new((num_reads, num_haps));
                // E-step: Calculate expected values of latent variables
                for i in 0..num_reads {
                    if let Some(row) = mismatch_fp_old.outer_view(i) {
                        let sum: f64 = row.iter().map(|(_, &val)| val).sum();
                        if sum > 0.0 {
                            // For non-zero sum, normalize the row
                            for (j, &val) in row.iter() {
                                mismatch_fp_new_triplet.add_triplet(i, j.index(), val / sum);
                            }
                        }
                    }
                }
                mismatch_fp_new = mismatch_fp_new_triplet.to_csr::<usize>();
                // M-step: Update parameters (theta_new) based on expectations
                for j in 0..num_haps {
                    // Sum column j of mismatch_fp_new
                    let col_sum: f64 = (0..num_reads)
                        .filter_map(|i| mismatch_fp_new.get(i, j))
                        .sum();

                    theta_new[j] = col_sum / num_reads as f64;
                }

                // Create a new triplet for next iteration
                let mut next_fp_triplet = TriMat::new((num_reads, num_haps));

                // Update mismatch_fp for next iteration
                for i in 0..num_reads {
                    if let Some(row) = mismatches_mat.outer_view(i) {
                        for (j, &val) in row.iter() {
                            let j_idx = j.index();
                            if theta_new[j_idx] > 0.0 {
                                next_fp_triplet.add_triplet(i, j_idx, val * theta_new[j_idx]);
                            }
                        }
                    }
                }
                mismatch_fp_new = next_fp_triplet.to_csr::<usize>();
                // Calculate likelihood and check convergence
                likelihood_new = calculate_likelihood(&mismatch_fp_new, num_reads);
                if likelihood_new <= likelihood_old - 1e-5 {
                    // Revert to theta_2 if likelihood decreases
                    theta_new.copy_from_slice(&theta_2);
                    mismatch_fp_new = mismatch_fp_2.clone();
                    likelihood_new = calculate_likelihood(&mismatch_fp_new, num_reads);
                }
                if (likelihood_new - likelihood_old).abs() < self.em_convergence_delta {
                    // Store final frequencies
                    for (j, haplotype) in haplotypes.iter_mut().enumerate() {
                        if theta_new[j] >= 0.001 {
                            haplotype.frequencies.insert(sample.clone(), theta_new[j]);
                        } else {
                            haplotype.frequencies.insert(sample.clone(), 0.0);
                        }
                    }
                    break;
                }
                // Update for next iteration
                likelihood_old = likelihood_new;
                theta_2.copy_from_slice(&theta_old);
                theta_old.copy_from_slice(&theta_new);
                mismatch_fp_2 = mismatch_fp_old;
                mismatch_fp_old = mismatch_fp_new;
            }
        }
        // Remove haplotypes with zero frequencies across all samples
        let mut indices_to_remove = Vec::new();
        for (hap_idx, haplotype) in haplotypes.iter().enumerate() {
            let mut has_nonzero = false;
            for sample in &self.samples {
                if let Some(&freq) = haplotype.frequencies.get(sample) {
                    if !freq.is_nan() && freq >= 0.001 {
                        has_nonzero = true;
                        break;
                    }
                }
            }
            if !has_nonzero {
                indices_to_remove.push(hap_idx);
            }
        }
        // Remove haplotypes in reverse order to maintain correct indices
        for &idx in indices_to_remove.iter().rev() {
            haplotypes.remove(idx);
        }
        // Rescale frequencies to sum to 1.0 for each sample
        for sample in &self.samples {
            let mut sum = 0.0;
            for haplotype in haplotypes.iter() {
                if let Some(&freq) = haplotype.frequencies.get(sample) {
                    sum += freq;
                }
            }
            if sum > 0.0 {
                for haplotype in haplotypes.iter_mut() {
                    if let Some(freq) = haplotype.frequencies.get_mut(sample) {
                        *freq /= sum;
                    }
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
        let mut total_cost = 0.0;
        // Calculate mismatch cost between reads and haplotypes
        for read in &self.reads {
            let mut total_mismatch_probability = 0.0;
            for haplotype in haplotypes
                .iter()
                .filter(|h| h.frequencies.contains_key(&read.sample))
            {
                let mismatches = read
                    .sequence
                    .iter()
                    .zip(&haplotype.sequence)
                    .filter(|(&r, &h)| r != h && r != b'-')
                    .count();
                total_mismatch_probability +=
                    self.mismatch_probability(mismatches, read.sequence.len());
            }
            total_cost -= total_mismatch_probability.ln();
        }
        // Penalty from four gamete test
        total_cost += self.lambda1 * self.min_recombinations(haplotypes) as f64;
        // Penalty for number of haplotypes
        total_cost += self.lambda2 * haplotypes.len() as f64;
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
    /// Additionally, when temperature is high, it may perform an extra random mutation to help
    /// explore the solution space more broadly.
    ///
    /// After structural modifications, it runs Square EM to optimize the frequencies.
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
    ///
    /// # Implementation Details
    ///
    /// - Uses thread_rng for random number generation
    /// - Ensures no duplicate sequences are added
    /// - Mutation operations scale with temperature
    /// - Maintains sample consistency when recombining haplotypes
    /// - Runs Square EM after modifications to optimize frequencies
    fn anneal(
        &self,
        param: &Self::Param,
        temp: Self::Float,
    ) -> Result<Self::Output, anyhow::Error> {
        trace!("Starting annealing step with temperature {}", temp);

        let mut rng = if let Some(seed) = self.seed {
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::rngs::StdRng::from_entropy()
        };
        let mut new_haplotypes = param.clone();
        debug!("Current number of haplotypes: {}", new_haplotypes.len());

        // If there's only one haplotype, we can't delete it
        // So we just add a new haplotype
        let operation: i32 = if new_haplotypes.len() == 1 {
            debug!("Only one haplotype present, forcing add operation");
            2
        } else {
            rng.gen_range(0..3)
        };
        match operation {
            0 if new_haplotypes.len() > 1 => {
                // Delete a random haplotype
                let idx_to_remove = rng.gen_range(0..new_haplotypes.len());
                debug!(
                    "Operation: Delete - Removing haplotype at index {}",
                    idx_to_remove
                );
                new_haplotypes.remove(idx_to_remove);
            }
            1 if new_haplotypes.len() >= 2 => {
                // Recombine two random haplotypes
                debug!("Operation: Recombine");
                let idx1 = rng.gen_range(0..new_haplotypes.len());
                let mut idx2 = rng.gen_range(0..new_haplotypes.len());
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
                        break;
                    }
                    if idx1 == idx2 {
                        trace!("Same indices, regenerating idx2");
                        idx2 = rng.gen_range(0..new_haplotypes.len());
                        attempts += 1;
                        continue;
                    }
                    // Check if sequences are different enough to recombine
                    let mismatches = new_haplotypes[idx1]
                        .sequence
                        .iter()
                        .zip(&new_haplotypes[idx2].sequence)
                        .filter(|(&a, &b)| a != b)
                        .count();

                    trace!(
                        "Attempt {}: Found {} mismatches between haplotypes {} and {}",
                        attempts + 1,
                        mismatches,
                        idx1,
                        idx2
                    );

                    // Only recombine if sequences differ by at least 2 positions
                    if mismatches >= 2 {
                        let crossover_point = rng.gen_range(0..new_haplotypes[idx1].sequence.len());
                        debug!(
                            "Performing recombination at position {} between haplotypes {} and {}",
                            crossover_point, idx1, idx2
                        );

                        let mut recombined1 = new_haplotypes[idx1].sequence.clone();
                        let mut recombined2 = new_haplotypes[idx2].sequence.clone();
                        recombined1[crossover_point..]
                            .copy_from_slice(&new_haplotypes[idx2].sequence[crossover_point..]);
                        recombined2[crossover_point..]
                            .copy_from_slice(&new_haplotypes[idx1].sequence[crossover_point..]);
                        let mut new_sequences = Vec::new();
                        if !new_haplotypes.iter().any(|h| h.sequence == recombined1) {
                            trace!("Adding first recombined sequence");
                            new_sequences.push(recombined1);
                        }
                        if !new_haplotypes.iter().any(|h| h.sequence == recombined2) {
                            trace!("Adding second recombined sequence");
                            new_sequences.push(recombined2);
                        }

                        debug!("Generated {} new unique sequences", new_sequences.len());

                        for new_seq in new_sequences {
                            let mut combined_frequencies = HashMap::new();
                            for (sample, &freq1) in &new_haplotypes[idx1].frequencies {
                                let freq2 =
                                    new_haplotypes[idx2].frequencies.get(sample).unwrap_or(&0.0);
                                combined_frequencies.insert(sample.clone(), (freq1 + freq2) / 4.0);
                            }
                            new_haplotypes.push(Haplotype {
                                sequence: new_seq,
                                frequencies: combined_frequencies,
                            });
                        }
                        break;
                    }
                    attempts += 1;
                    idx2 = rng.gen_range(0..new_haplotypes.len());
                }
            }
            2 if new_haplotypes.len() < self.reads.len() => {
                // Add a new haplotype by mutating an existing one
                debug!("Operation: Add new haplotype by mutation");
                let idx_to_copy = rng.gen_range(0..new_haplotypes.len());
                let mut new_sequence = new_haplotypes[idx_to_copy].sequence.clone();
                let pos_to_change = rng.gen_range(0..new_sequence.len());
                let new_nucleotide = [b'A', b'C', b'G', b'T'][rng.gen_range(0..4)];

                trace!(
                    "Mutating haplotype {} at position {} to {}",
                    idx_to_copy,
                    pos_to_change,
                    new_nucleotide as char
                );

                new_sequence[pos_to_change] = new_nucleotide;
                // Only add if this sequence doesn't already exist
                if !new_haplotypes.iter().any(|h| h.sequence == new_sequence) {
                    debug!("Adding new mutated haplotype");
                    new_haplotypes.push(Haplotype {
                        sequence: new_sequence,
                        frequencies: new_haplotypes[idx_to_copy].frequencies.clone(),
                    });
                } else {
                    debug!("Mutated sequence already exists, skipping addition");
                }
            }
            _ => {
                trace!("No operation performed - conditions not met");
            }
        }

        // Scale mutations based on temperature
        if temp > 0.0 && rng.gen::<f64>() < temp {
            // Additional random mutation when temperature is high
            debug!(
                "Performing additional temperature-based mutation (temp: {})",
                temp
            );
            if let Some(idx) = (0..new_haplotypes.len()).choose(&mut rng) {
                let haplotype = &mut new_haplotypes[idx];
                let pos = rng.gen_range(0..haplotype.sequence.len());
                let new_nucleotide = [b'A', b'C', b'G', b'T'][rng.gen_range(0..4)];
                trace!(
                    "Temperature mutation: haplotype {} at position {} to {}",
                    idx,
                    pos,
                    new_nucleotide as char
                );
                haplotype.sequence[pos] = new_nucleotide;
            }
        }

        debug!(
            "Running EM optimization on {} haplotypes",
            new_haplotypes.len()
        );
        // Mutate haplotype frequencies with and zero-out with Square EM
        self.square_expectation_maximization(&mut new_haplotypes)?;

        debug!(
            "Annealing step complete, returning {} haplotypes",
            new_haplotypes.len()
        );
        Ok(new_haplotypes)
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
    let problem = HaplotypeEstimationProblem {
        samples: optimization_parameters.samples,
        reads: reads.to_vec(),
        error_rate: optimization_parameters.error_rate,
        lambda1: optimization_parameters.lambda1,
        lambda2: optimization_parameters.lambda2,
        em_max_mismatches: optimization_parameters.max_mismatches,
        em_iterations: optimization_parameters.em_iterations,
        em_convergence_delta: optimization_parameters.em_cdelta,
        seed: optimization_parameters.seed,
    };
    let rng = if let Some(seed) = optimization_parameters.seed {
        rand::rngs::StdRng::seed_from_u64(seed)
    } else {
        rand::rngs::StdRng::from_entropy()
    };
    let solver = SimulatedAnnealing::new_with_rng(optimization_parameters.sa_max_temperature, rng)
        .unwrap()
        .with_temp_func(SATempFunc::TemperatureFast)
        .with_stall_best(optimization_parameters.sa_iterations as u64);
    let mut best_haplotypes = initial_haplotypes.clone();
    let mut best_objective = f64::INFINITY;
    for i in 0..optimization_parameters.sa_reruns {
        info!(
            "Running SA with {} haplotypes, iteration {}",
            initial_haplotypes.len(),
            i
        );
        let result = Executor::new(problem.clone(), solver.clone())
            .configure(|state| state.param(initial_haplotypes.clone()))
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
    }
    best_haplotypes
}

/// Main function
///
/// TODO: Parallel processing
///
/// # Arguments
///
/// * `args` - Command line arguments
///
/// # Returns
///
/// * `Ok(())` - If the program runs successfully
/// * `Err(anyhow::Error)` - If there was an error
fn main() -> Result<()> {
    // Initialize tracing subscriber
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .init();
    let args = Args::parse();
    let unaligned = unaligned_samples(&args.files)?;
    if !unaligned.is_empty() {
        unaligned
            .iter()
            .for_each(|sample| eprintln!("Sample {sample} is not aligned"));
        exit(1);
    }
    let reads = extract_reads(&args.files);
    let (variant_only_reads, invariant_positions) = remove_invariants(&reads);
    let initial_haplotypes = init_haplotypes(&variant_only_reads);
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
        seed: args.seed,
    };
    let proposed_haplotypes =
        propose_haplotypes(&reads, &initial_haplotypes, optimization_parameters);
    // Print CSV headers
    print!("sequence");
    for sample in &args.files {
        print!(",{}", sample);
    }
    println!();
    // Print proposed haplotypes with restored invariant positions
    for haplotype in &proposed_haplotypes {
        let restored_sequence = restore_invariants(&haplotype.sequence, &invariant_positions);
        print!("{}", String::from_utf8_lossy(&restored_sequence));
        for sample in &args.files {
            print!(",{}", haplotype.frequencies.get(sample).unwrap_or(&0.0));
        }
        println!();
    }
    // Print sums as the last row
    print!("SUM");
    for sample in &args.files {
        let sum: f64 = proposed_haplotypes
            .iter()
            .map(|h| h.frequencies.get(sample).unwrap_or(&0.0))
            .sum();
        print!(",{}", sum);
    }
    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    fn create_test_reads(sequences: Vec<&str>, sample: &str) -> Vec<Read> {
        sequences
            .into_iter()
            .map(|seq| Read {
                sequence: seq.as_bytes().to_vec(),
                sample: sample.to_string(),
            })
            .collect()
    }

    fn create_test_haplotypes(sequences: Vec<&str>) -> Vec<Haplotype> {
        sequences
            .into_iter()
            .map(|seq| Haplotype {
                sequence: seq.as_bytes().to_vec(),
                frequencies: HashMap::new(),
            })
            .collect()
    }

    fn create_test_problem() -> HaplotypeEstimationProblem {
        HaplotypeEstimationProblem {
            samples: vec![],
            reads: vec![],
            error_rate: 0.01,
            lambda1: 1.0,
            lambda2: 1.0,
            em_max_mismatches: 3,
            em_iterations: 100,
            em_convergence_delta: 0.001,
            seed: Some(12345),
        }
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
        let haplotypes = init_haplotypes(&reads);

        assert_eq!(haplotypes.len(), 1);
        assert_eq!(haplotypes[0].sequence, b"ACGT");
        assert_eq!(haplotypes[0].frequencies.len(), 1);
        assert_eq!(haplotypes[0].frequencies.get("sample1"), Some(&1.0));
    }

    #[test]
    fn test_single_read_with_blanks() {
        let reads = create_test_reads(vec!["A-C"], "sample1");
        let haplotypes = init_haplotypes(&reads);

        let expected: HashSet<Vec<u8>> = HashSet::from([
            b"AAC".to_vec(),
            b"ACC".to_vec(),
            b"AGC".to_vec(),
            b"ATC".to_vec(),
        ]);

        assert_eq!(haplotypes.len(), expected.len());
        for haplotype in haplotypes {
            assert!(expected.contains(&haplotype.sequence));
            assert_eq!(haplotype.frequencies.len(), 1);
            assert_eq!(haplotype.frequencies.get("sample1"), Some(&0.25));
        }
    }

    #[test]
    fn test_multiple_reads_no_blanks() {
        let reads = create_test_reads(vec!["ACGT", "TGCA"], "sample1");
        let haplotypes = init_haplotypes(&reads);

        let expected: HashSet<Vec<u8>> = HashSet::from([b"ACGT".to_vec(), b"TGCA".to_vec()]);

        assert_eq!(haplotypes.len(), expected.len());
        for haplotype in haplotypes {
            assert!(expected.contains(&haplotype.sequence));
            assert_eq!(haplotype.frequencies.len(), 1);
            assert_eq!(haplotype.frequencies.get("sample1"), Some(&0.5));
        }
    }

    #[test]
    fn test_multiple_reads_with_blanks() {
        let reads = create_test_reads(vec!["A-C", "T-G"], "sample1");
        let haplotypes = init_haplotypes(&reads);

        let expected: HashSet<Vec<u8>> = HashSet::from([
            b"AAC".to_vec(),
            b"ACC".to_vec(),
            b"AGC".to_vec(),
            b"ATC".to_vec(),
            b"TGG".to_vec(),
            b"TAG".to_vec(),
            b"TCG".to_vec(),
            b"TTG".to_vec(),
        ]);

        assert_eq!(haplotypes.len(), expected.len());
        for haplotype in haplotypes {
            assert!(expected.contains(&haplotype.sequence));
            assert_eq!(haplotype.frequencies.len(), 1);
            assert_eq!(haplotype.frequencies.get("sample1"), Some(&0.125));
        }
    }

    #[test]
    fn test_deduplication_of_haplotypes() {
        let reads = create_test_reads(vec!["A-C", "A-C"], "sample1");
        let haplotypes = init_haplotypes(&reads);

        let expected: HashSet<Vec<u8>> = HashSet::from([
            b"AAC".to_vec(),
            b"ACC".to_vec(),
            b"AGC".to_vec(),
            b"ATC".to_vec(),
        ]);

        assert_eq!(haplotypes.len(), expected.len());
        for haplotype in haplotypes {
            assert!(expected.contains(&haplotype.sequence));
            assert_eq!(haplotype.frequencies.len(), 1);
            assert_eq!(haplotype.frequencies.get("sample1"), Some(&0.25));
        }
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
        let haplotypes = init_haplotypes(&reads);

        let expected_sample1: HashMap<Vec<u8>, f64> = HashMap::from([
            (b"AAC".to_vec(), 0.25),
            (b"ACC".to_vec(), 0.25),
            (b"AGC".to_vec(), 0.25),
            (b"ATC".to_vec(), 0.25),
        ]);
        let expected_sample2: HashMap<Vec<u8>, f64> = HashMap::from([
            (b"TAG".to_vec(), 0.25),
            (b"TCG".to_vec(), 0.25),
            (b"TGG".to_vec(), 0.25),
            (b"TTG".to_vec(), 0.25),
        ]);

        for haplotype in &haplotypes {
            // Check frequencies are set correctly
            assert!(haplotype.frequencies.contains_key("sample1"));
            assert!(haplotype.frequencies.contains_key("sample2"));

            // Check if sequence is from sample1's expansions
            if let Some(&expected_freq) = expected_sample1.get(&haplotype.sequence) {
                assert!((haplotype.frequencies["sample1"] - expected_freq).abs() < 1e-10);
                assert_eq!(haplotype.frequencies["sample2"], 0.0);
            }

            // Check if sequence is from sample2's expansions
            if let Some(&expected_freq) = expected_sample2.get(&haplotype.sequence) {
                assert!((haplotype.frequencies["sample2"] - expected_freq).abs() < 1e-10);
                assert_eq!(haplotype.frequencies["sample1"], 0.0);
            }
        }

        // Check we found all expected sequences
        let found_sequences: HashSet<_> = haplotypes.iter().map(|h| &h.sequence).collect();
        let expected_sequences: HashSet<_> = expected_sample1
            .keys()
            .chain(expected_sample2.keys())
            .collect();
        assert_eq!(found_sequences, expected_sequences);
    }

    #[test]
    fn test_large_sequences_with_no_blanks() {
        let long_seq = "A".repeat(1000);
        let reads = create_test_reads(vec![&long_seq], "sample1");
        let haplotypes = init_haplotypes(&reads);

        assert_eq!(haplotypes.len(), 1);
        assert_eq!(haplotypes[0].sequence.len(), 1000);
        assert_eq!(haplotypes[0].sequence, long_seq.as_bytes());
        assert_eq!(haplotypes[0].frequencies.len(), 1);
        assert_eq!(haplotypes[0].frequencies.get("sample1"), Some(&1.0));
    }

    #[test]
    #[ignore]
    fn test_large_sequences_with_blanks() {
        let long_seq_with_blanks = format!("A{}C", "-".repeat(998));
        let reads = create_test_reads(vec![&long_seq_with_blanks], "sample1");
        let haplotypes = init_haplotypes(&reads);

        let expected_count = 4_usize.pow(998); // 998 blanks => 4^998 combinations
        assert_eq!(haplotypes.len(), expected_count);
    }

    #[test]
    fn test_empty_reads() {
        let reads = create_test_reads(vec![], "sample1");
        let haplotypes = init_haplotypes(&reads);

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
}
