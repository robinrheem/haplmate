use anyhow::Result;
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
    let _initial_haplotypes = dbg!(init_haplotypes(&variant_only_reads));
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
