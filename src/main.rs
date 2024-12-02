use clap::Parser;

/// Estimating haplotypes with Simulated Annealing and Expectation-Maximization
#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Args {
    /// Input FASTA file(s)
    #[arg(value_name = "FILE", default_value = "-")]
    files: Vec<String>,
    /// Number of samples
    #[arg(short, long)]
    samples: usize,
    /// Number of reads in the alignment file
    #[arg(short, long)]
    reads: usize,
    /// Length of aligned reads
    #[arg(short('l'), long)]
    sequence_length: usize,
}

fn main() {
    let args = Args::parse();
    dbg!(args);
}
