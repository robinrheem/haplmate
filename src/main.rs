use anyhow::Result;
use argmin::core::{CostFunction, Executor};
use argmin::solver::simulatedannealing::{Anneal, SATempFunc, SimulatedAnnealing};
use rand::prelude::*;
use seq_io::fasta::{Reader, Record};
use std::collections::{HashSet, VecDeque};
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

#[derive(Debug, Clone)]
struct Read {
    id: String,
    sequence: Vec<u8>,
    sample: String,
}

#[derive(Debug, Clone)]
struct Haplotype {
    sequence: Vec<u8>,
    sample: String,
}

#[derive(Debug)]
struct OptimizationParameters {
    max_mismatches: usize,
    em_iterations: usize,
    sa_schedule: f64,
    lambda1: f64,
    lambda2: f64,
    error_rate: f64,
    sa_min_temperature: f64,
    sa_max_temperature: f64,
    sa_iterations: usize,
    em_interval: usize,
    sa_reruns: usize,
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

/// Remove all invariants from all reads
///
/// # Arguments
///
/// * `reads` - A list of reads
///
/// # Returns
///
/// A list of reads that have removed invariants
fn remove_invariants(reads: &Vec<Read>) -> Vec<Read> {
    let mut filtered_sequences = vec![Vec::new(); reads.len()];
    for i in 0..reads.first().unwrap().sequence.len() {
        let column: Vec<u8> = reads.iter().map(|read| read.sequence[i]).collect();
        let unique_nucleotides: HashSet<u8> =
            column.iter().filter(|&&c| c != b'-').copied().collect();
        // Skip if only one type of nucleotide (or all gaps)
        if unique_nucleotides.len() <= 1 {
            continue;
        }
        for (j, c) in column.into_iter().enumerate() {
            filtered_sequences[j].push(c);
        }
    }
    reads
        .iter()
        .enumerate()
        .map(|(i, read)| Read {
            id: read.id.clone(),
            sequence: filtered_sequences[i].clone(),
            sample: read.sample.clone(),
        })
        .collect()
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
                    id: record.id().unwrap().to_string(),
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
/// List of haplotypes(full sequences)
fn init_haplotypes(reads: &Vec<Read>) -> Vec<Haplotype> {
    let mut haplotype_set: HashSet<(Vec<u8>, String)> = HashSet::new();
    for read in reads {
        // Queue to store intermediate sequences during expansion
        let mut queue: VecDeque<Vec<u8>> = VecDeque::new();
        queue.push_back(vec![]); // Start with an empty sequence

        // Expand blanks iteratively
        for &nucleotide in &read.sequence {
            let mut level_size = queue.len();
            while level_size > 0 {
                level_size -= 1;
                let mut current = queue.pop_front().unwrap();
                if nucleotide == b'-' {
                    // For blanks, enqueue all possible nucleotides
                    for &fill in b"ACGT" {
                        let mut next = current.clone();
                        next.push(fill);
                        queue.push_back(next);
                    }
                } else {
                    // For normal nucleotides, continue the sequence
                    current.push(nucleotide);
                    queue.push_back(current);
                }
            }
        }
        // Add all fully expanded sequences with the sample to the haplotype set
        for sequence in queue {
            haplotype_set.insert((sequence, read.sample.clone()));
        }
    }
    haplotype_set
        .into_iter()
        .map(|(sequence, sample)| Haplotype { sequence, sample })
        .collect()
}

#[derive(Clone)]
struct HaplotypeObjective {
    reads: Vec<Read>,
    error_rate: f64,
    lambda1: f64,
    lambda2: f64,
    max_iterations: usize,
    convergence_delta: f64,
}

impl HaplotypeObjective {
    /// E-step: Calculate posterior probabilities
    /// M-step: Update haplotype frequencies
    /// Repeat until convergence or max iterations
    fn expectation_maximization(&self, current_haplotypes: &Vec<Haplotype>) {
        let mut frequencies: Vec<f64> =
            vec![1.0 / current_haplotypes.len() as f64; current_haplotypes.len()];
        let mut old_frequencies: Vec<f64>;

        for _ in 0..self.max_iterations {
            old_frequencies = frequencies.clone();

            // E-step: Calculate posterior probabilities
            let mut responsibilities: Vec<Vec<f64>> =
                vec![vec![0.0; current_haplotypes.len()]; self.reads.len()];
            for (i, read) in self.reads.iter().enumerate() {
                let mut total_prob = 0.0;

                // Calculate probabilities for each haplotype
                for (j, haplotype) in current_haplotypes.iter().enumerate() {
                    if read.sample != haplotype.sample {
                        continue;
                    }
                    let mismatches = read
                        .sequence
                        .iter()
                        .zip(&haplotype.sequence)
                        .filter(|(&r, &h)| r != h && r != b'-')
                        .count();
                    let prob = frequencies[j]
                        * (1.0 - self.error_rate).powi((read.sequence.len() - mismatches) as i32)
                        * self.error_rate.powi(mismatches as i32);
                    responsibilities[i][j] = prob;
                    total_prob += prob;
                }

                // Normalize probabilities
                if total_prob > 0.0 {
                    for prob in responsibilities[i].iter_mut() {
                        *prob /= total_prob;
                    }
                }
            }

            // M-step: Update frequencies
            for j in 0..current_haplotypes.len() {
                frequencies[j] = responsibilities.iter().map(|resp| resp[j]).sum::<f64>()
                    / self.reads.len() as f64;
            }

            // Check convergence
            let max_diff = frequencies
                .iter()
                .zip(&old_frequencies)
                .map(|(new, old)| (new - old).abs())
                .fold(0.0, f64::max);

            if max_diff < self.convergence_delta {
                break;
            }
        }
    }
}

impl CostFunction for HaplotypeObjective {
    type Param = Vec<Haplotype>;
    type Output = f64;

    fn cost(&self, haplotypes: &Self::Param) -> std::result::Result<Self::Output, anyhow::Error> {
        let mut total_cost = 0.0;

        // Calculate mismatch cost between reads and haplotypes
        for read in &self.reads {
            let mut min_mismatches = usize::MAX;
            for haplotype in haplotypes {
                if read.sample != haplotype.sample {
                    continue;
                }
                let mismatches = read
                    .sequence
                    .iter()
                    .zip(&haplotype.sequence)
                    .filter(|(&r, &h)| r != h && r != b'-')
                    .count();
                min_mismatches = min_mismatches.min(mismatches);
            }
            total_cost += (min_mismatches as f64) * -self.error_rate.ln();
        }

        // Add penalty terms
        let num_haplotypes = haplotypes.len() as f64;
        total_cost += self.lambda1 * num_haplotypes; // Penalty for number of haplotypes

        // Penalty for diversity between haplotypes
        let mut diversity_penalty = 0.0;
        for (i, h1) in haplotypes.iter().enumerate() {
            for h2 in haplotypes.iter().skip(i + 1) {
                if h1.sample != h2.sample {
                    continue;
                }
                let differences = h1
                    .sequence
                    .iter()
                    .zip(&h2.sequence)
                    .filter(|(&a, &b)| a != b && a != b'-' && b != b'-')
                    .count();
                diversity_penalty += differences as f64;
            }
        }
        total_cost += self.lambda2 * diversity_penalty;

        Ok(total_cost)
    }
}

impl Anneal for HaplotypeObjective {
    type Param = Vec<Haplotype>;
    type Output = Vec<Haplotype>;
    type Float = f64;

    fn anneal(
        &self,
        param: &Self::Param,
        temp: Self::Float,
    ) -> Result<Self::Output, anyhow::Error> {
        let mut rng = rand::thread_rng();
        let mut new_haplotypes = param.clone();
        let rand_val: f64 = rng.gen();

        // Similar to legacy code's getNewHaplotypes function
        if new_haplotypes.len() > 1 && rand_val < 0.33 {
            // Delete a random haplotype
            let idx_to_remove = rng.gen_range(0..new_haplotypes.len());
            new_haplotypes.remove(idx_to_remove);
        } else if new_haplotypes.len() < self.reads.len() && rand_val >= 0.67 {
            // Add a new haplotype by mutating an existing one
            let idx_to_copy = rng.gen_range(0..new_haplotypes.len());
            let mut new_sequence = new_haplotypes[idx_to_copy].sequence.clone();
            let pos_to_change = rng.gen_range(0..new_sequence.len());
            let new_nucleotide = [b'A', b'C', b'G', b'T'][rng.gen_range(0..4)];
            new_sequence[pos_to_change] = new_nucleotide;

            // Only add if this sequence doesn't already exist
            if !new_haplotypes.iter().any(|h| h.sequence == new_sequence) {
                new_haplotypes.push(Haplotype {
                    sequence: new_sequence,
                    sample: new_haplotypes[idx_to_copy].sample.clone(),
                });
            }
        } else if new_haplotypes.len() >= 2 && rand_val >= 0.33 && rand_val < 0.67 {
            // Recombine two random haplotypes
            let idx1 = rng.gen_range(0..new_haplotypes.len());
            let mut idx2;
            loop {
                idx2 = rng.gen_range(0..new_haplotypes.len());
                if idx1 != idx2 && new_haplotypes[idx1].sample == new_haplotypes[idx2].sample {
                    break;
                }
            }

            // Create recombined sequence
            let crossover_point = rng.gen_range(0..new_haplotypes[idx1].sequence.len());
            let mut recombined = new_haplotypes[idx1].sequence.clone();
            recombined[crossover_point..]
                .copy_from_slice(&new_haplotypes[idx2].sequence[crossover_point..]);

            // Only add if this sequence doesn't already exist
            if !new_haplotypes.iter().any(|h| h.sequence == recombined) {
                new_haplotypes.push(Haplotype {
                    sequence: recombined,
                    sample: new_haplotypes[idx1].sample.clone(),
                });
            }
        }

        // Scale mutations based on temperature
        if temp > 0.0 && rng.gen::<f64>() < temp {
            // Additional random mutation when temperature is high
            if let Some(idx) = (0..new_haplotypes.len()).choose(&mut rng) {
                let haplotype = &mut new_haplotypes[idx];
                let pos = rng.gen_range(0..haplotype.sequence.len());
                let new_nucleotide = [b'A', b'C', b'G', b'T'][rng.gen_range(0..4)];
                haplotype.sequence[pos] = new_nucleotide;
            }
        }

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
    let objective_function = HaplotypeObjective {
        reads: reads.to_vec(),
        error_rate: optimization_parameters.error_rate,
        lambda1: optimization_parameters.lambda1,
        lambda2: optimization_parameters.lambda2,
        max_iterations: optimization_parameters.em_iterations,
        convergence_delta: optimization_parameters.em_cdelta,
    };

    // Configure simulated annealing solver
    let solver = SimulatedAnnealing::new(optimization_parameters.sa_max_temperature)
        .unwrap()
        .with_temp_func(SATempFunc::TemperatureFast)
        .with_stall_best(optimization_parameters.sa_iterations as u64);

    // Run the solver multiple times with different initial conditions
    let mut best_haplotypes = initial_haplotypes.clone();
    let mut best_objective = f64::INFINITY;

    for i in 0..optimization_parameters.sa_reruns {
        // Run simulated annealing
        let result = Executor::new(objective_function.clone(), solver.clone())
            .configure(|state| state.param(initial_haplotypes.clone()))
            .run()
            .unwrap();

        let state = result.state();
        let best_cost = state.best_cost;
        if best_cost < best_objective {
            if let Some(ref param) = state.best_param {
                best_haplotypes = param.clone();
                best_objective = best_cost;
            }
        }

        // Run EM algorithm periodically
        if i % optimization_parameters.em_interval == 0 {
            objective_function.expectation_maximization(&best_haplotypes);
        }
    }

    best_haplotypes
}

fn main() -> Result<()> {
    let args = Args::parse();
    let unaligned = unaligned_samples(&args.files)?;
    if !unaligned.is_empty() {
        unaligned
            .iter()
            .for_each(|sample| eprintln!("Sample {sample} is not aligned"));
        exit(1);
    }
    let args = dbg!(args);
    let reads = extract_reads(&args.files);
    let variant_only_reads = remove_invariants(&reads);
    let initial_haplotypes = init_haplotypes(&variant_only_reads);
    let optimization_parameters = OptimizationParameters {
        max_mismatches: args.mismatches,
        em_cdelta: args.em_cdelta,
        em_interval: args.em_interval,
        em_iterations: args.em_iterations,
        error_rate: args.error_rate,
        lambda1: args.lambda1,
        lambda2: args.lambda2,
        sa_iterations: args.sa_iterations,
        sa_max_temperature: args.sa_max_temperature,
        sa_min_temperature: args.sa_min_temperature,
        sa_reruns: args.sa_reruns,
        sa_schedule: args.sa_schedule,
    };
    let _proposed_haplotypes = dbg!(propose_haplotypes(
        &reads,
        &initial_haplotypes,
        optimization_parameters
    ));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    // Helper function to create test reads
    fn create_test_reads(sequences: Vec<&str>, sample: &str) -> Vec<Read> {
        sequences
            .into_iter()
            .enumerate()
            .map(|(i, seq)| Read {
                id: format!("read{}", i + 1),
                sequence: seq.as_bytes().to_vec(),
                sample: sample.to_string(),
            })
            .collect()
    }

    #[test]
    fn test_basic_invariant_removal() {
        let reads = create_test_reads(vec!["AAGTC", "AAATC", "AACTC"], "sample1");
        let result = remove_invariants(&reads);

        for (i, read) in result.iter().enumerate() {
            assert_eq!(
                read.id,
                format!("read{}", i + 1),
                "ID mismatch for read {}",
                i + 1
            );
            assert_eq!(read.sample, "sample1", "Sample mismatch for read {}", i + 1);
        }
        assert_eq!(result[0].sequence, b"G");
        assert_eq!(result[1].sequence, b"A");
        assert_eq!(result[2].sequence, b"C");
    }

    #[test]
    fn test_all_invariant_sequence() {
        let reads = create_test_reads(vec!["AAAAA", "AAAAA", "AAAAA"], "sample1");
        let result = remove_invariants(&reads);

        for (i, read) in result.iter().enumerate() {
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

        assert_eq!(result[0].sequence, b"ACTG");
        assert_eq!(result[1].sequence, b"GCTA");
        assert_eq!(result[2].sequence, b"TGCA");
    }

    #[test]
    fn test_with_gaps() {
        let reads = create_test_reads(vec!["A-CTG", "A-CTG", "A-CTG"], "sample1");
        let result = remove_invariants(&reads);

        for (i, read) in result.iter().enumerate() {
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

        assert_eq!(result[0].sequence, b"C",);
        assert_eq!(result[1].sequence, b"C",);
        assert_eq!(result[2].sequence, b"G",);
    }

    #[test]
    fn test_mixed_gaps_with_single_invariants() {
        let reads = create_test_reads(vec!["-ACTA", "A-CTA", "A-GTA"], "sample1");
        let result = remove_invariants(&reads);

        assert_eq!(result[0].sequence, b"C",);
        assert_eq!(result[1].sequence, b"C",);
        assert_eq!(result[2].sequence, b"G",);
    }

    #[test]
    fn test_single_read() {
        let reads = create_test_reads(vec!["ACGT"], "sample1");
        let result = remove_invariants(&reads);

        assert_eq!(result.len(), 1, "Should have exactly one result");
        assert!(
            result[0].sequence.is_empty(),
            "Single read sequence should be empty, but got: {:?}",
            String::from_utf8_lossy(&result[0].sequence)
        );
    }

    #[test]
    fn test_empty_sequences() {
        let reads = create_test_reads(vec!["", "", ""], "sample1");
        let result = remove_invariants(&reads);

        assert_eq!(result.len(), 3, "Should have three results");
        for (i, read) in result.iter().enumerate() {
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
            id: "read1".to_string(),
            sequence: b"ACGT".to_vec(),
            sample: "sample1".to_string(),
        });
        reads.push(Read {
            id: "read2".to_string(),
            sequence: b"ACG".to_vec(),
            sample: "sample1".to_string(),
        });

        remove_invariants(&reads);
    }

    #[test]
    fn test_preserve_metadata() {
        let reads = vec![
            Read {
                id: "custom_id_1".to_string(),
                sequence: b"ACGT".to_vec(),
                sample: "sample_A".to_string(),
            },
            Read {
                id: "custom_id_2".to_string(),
                sequence: b"AGGT".to_vec(),
                sample: "sample_B".to_string(),
            },
        ];
        let result = remove_invariants(&reads);
        assert_eq!(result[0].id, "custom_id_1",);
        assert_eq!(result[0].sample, "sample_A",);
        assert_eq!(result[1].id, "custom_id_2",);
        assert_eq!(result[1].sample, "sample_B",);
    }

    #[test]
    fn test_large_sequences() {
        let long_seq_a = "A".repeat(1000);
        let long_seq_b = format!("{}T", "A".repeat(999));
        let reads = create_test_reads(vec![&long_seq_a, &long_seq_b], "sample1");
        let result = remove_invariants(&reads);

        assert_eq!(result[0].sequence, b"A",);
        assert_eq!(result[1].sequence, b"T",);
    }

    #[test]
    fn test_all_gaps() {
        let reads = create_test_reads(vec!["----", "----", "----"], "sample1");
        let result = remove_invariants(&reads);

        for (i, read) in result.iter().enumerate() {
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
        assert_eq!(haplotypes[0].sample, "sample1");
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
            assert_eq!(haplotype.sample, "sample1");
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
            assert_eq!(haplotype.sample, "sample1");
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
            assert_eq!(haplotype.sample, "sample1");
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
            assert_eq!(haplotype.sample, "sample1");
        }
    }

    #[test]
    fn test_reads_from_different_samples() {
        let reads = vec![
            Read {
                id: "read1".to_string(),
                sequence: b"A-C".to_vec(),
                sample: "sample1".to_string(),
            },
            Read {
                id: "read2".to_string(),
                sequence: b"T-G".to_vec(),
                sample: "sample2".to_string(),
            },
        ];
        let haplotypes = init_haplotypes(&reads);

        let expected_sample1: HashSet<Vec<u8>> = HashSet::from([
            b"AAC".to_vec(),
            b"ACC".to_vec(),
            b"AGC".to_vec(),
            b"ATC".to_vec(),
        ]);
        let expected_sample2: HashSet<Vec<u8>> = HashSet::from([
            b"TGG".to_vec(),
            b"TAG".to_vec(),
            b"TCG".to_vec(),
            b"TTG".to_vec(),
        ]);

        let mut haplotype_sample1 = vec![];
        let mut haplotype_sample2 = vec![];

        for haplotype in haplotypes {
            if haplotype.sample == "sample1" {
                haplotype_sample1.push(haplotype.sequence);
            } else if haplotype.sample == "sample2" {
                haplotype_sample2.push(haplotype.sequence);
            }
        }

        assert_eq!(haplotype_sample1.len(), expected_sample1.len());
        assert_eq!(haplotype_sample2.len(), expected_sample2.len());

        for seq in haplotype_sample1 {
            assert!(expected_sample1.contains(&seq));
        }
        for seq in haplotype_sample2 {
            assert!(expected_sample2.contains(&seq));
        }
    }

    #[test]
    fn test_large_sequences_with_no_blanks() {
        let long_seq = "A".repeat(1000);
        let reads = create_test_reads(vec![&long_seq], "sample1");
        let haplotypes = init_haplotypes(&reads);

        assert_eq!(haplotypes.len(), 1);
        assert_eq!(haplotypes[0].sequence.len(), 1000);
        assert_eq!(haplotypes[0].sequence, long_seq.as_bytes());
        assert_eq!(haplotypes[0].sample, "sample1");
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
}
