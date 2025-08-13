## Haplmate

Estimate haplotypes and per-sample frequencies from aligned reads using Simulated Annealing (SA) and SQUAREM (squared EM).

### Features
- MAF-based initialization with greedy haplotype discovery
- Stochastic search via Simulated Annealing
- Faster convergence using SQUAREM
- Multi-sample frequency estimation with CSV output

## Installation

### Download pre-built binaries (recommended)
Download the latest release for your platform from [GitHub Releases](https://github.com/robinrheem/haplmate/releases):

**Linux (x86_64):**
```bash
# Download and extract (replace VERSION with actual version, e.g., v0.1.0)
curl -L https://github.com/robinrheem/haplmate/releases/latest/download/haplmate-x86_64-unknown-linux-gnu.tar.gz | tar xz
sudo mv haplmate-x86_64-unknown-linux-gnu /usr/local/bin/haplmate
chmod +x /usr/local/bin/haplmate
```

**macOS (Apple Silicon):**
```bash
curl -L https://github.com/robinrheem/haplmate/releases/latest/download/haplmate-aarch64-apple-darwin.tar.gz | tar xz
sudo mv haplmate-aarch64-apple-darwin /usr/local/bin/haplmate
chmod +x /usr/local/bin/haplmate
```

**macOS (Intel):**
```bash
curl -L https://github.com/robinrheem/haplmate/releases/latest/download/haplmate-x86_64-apple-darwin.tar.gz | tar xz
sudo mv haplmate-x86_64-apple-darwin /usr/local/bin/haplmate
chmod +x /usr/local/bin/haplmate
```

**Windows:**
1. Download `haplmate-x86_64-pc-windows-msvc.zip` from [releases](https://github.com/robinrheem/haplmate/releases/latest)
2. Extract and add `haplmate.exe` to your PATH

### Build from source

**Prerequisites:**
- Rust toolchain (stable). Install with `rustup` from `https://rustup.rs`.
- macOS/Linux/WSL supported.

**Build locally:**
```bash
git clone https://github.com/robinrheem/haplmate.git
cd haplmate
cargo build --release
# Binary at target/release/haplmate
```

**Install into cargo bin:**
```bash
cargo install --path .
# Binary available as `haplmate` on your PATH
```

## Input requirements
- Input files are FASTA files, one or more per sample.
- Sequences across all reads in every provided file must be aligned to the same length.
  - The program checks alignment and exits with an error if any file contains unaligned reads.
- Gaps (`-`) are allowed and handled during initialization and mismatch counting.

## Quick start

### Minimal example
Create two small samples, then run Haplmate.
```bash
cat > sample1.fa << 'EOF'
>read1
ACGT
>read2
ACGT
EOF

cat > sample2.fa << 'EOF'
>read1
TGCA
>read2
TGCA
EOF

# Run with deterministic seed for reproducibility
./target/release/haplmate sample1.fa sample2.fa --sa-reruns=1 --sa-iterations=1 --lambda1=0.0 --lambda2=0.0 --seed=12345 --error-rate=0.04
```

Example output (CSV):
```text
sequence,sample1.fa,sample2.fa
ACGT,1,0
TGCA,0,1
SUM,1,1
```

### Using repository test data
```bash
./target/release/haplmate tests/data/simulated_reads_0.fa tests/data/simulated_reads_1.fa --seed=12345
```

## Usage

```bash
haplmate [OPTIONS] <FILE>...

Estimating haplotypes with Simulated Annealing and Expectation-Maximization

Usage: haplmate-x86_64-unknown-linux-gnu [OPTIONS] [FILE]...

Arguments:
  [FILE]...  Input FASTA file(s) [default: -]

Options:
  -o, --output <OUTPUT>
          Output file [default: estimated_haplotypes.csv]
  -m, --mismatches <MISMATCHES>
          Maximum allowed mismatch between haplotypes and reads [default: 15]
  -e, --em-iterations <EM_ITERATIONS>
          Maximum number of EM iterations during intermediate steps [default: 20000]
      --lambda1 <LAMBDA1>
          Lambda1 value (for testing purposes only) [default: 0.0001]
      --lambda2 <LAMBDA2>
          Lambda2 value (for testing purposes only) [default: 0.0001]
  -d, --error-rate <ERROR_RATE>
          Sequencing error [default: 0.00001]
      --sa-max-temperature <SA_MAX_TEMPERATURE>
          Starting maximum temp in simulated annealing [default: 10.0]
      --sa-iterations <SA_ITERATIONS>
          Number of iterations in simulated annealing [default: 2000]
      --sa-reruns <SA_RERUNS>
          Number of reruns of optimization algorithm [default: 5]
      --em-cdelta <EM_CDELTA>
          Delta to determine intermediate EM convergence steps [default: 0.1]
      --seed <SEED>
          Random seed for deterministic output(testing purposes only)
  -h, --help
          Print help
  -V, --version
          Print version
```

- FILE: One or more FASTA files (each file is treated as a separate sample).

### Output
Writes a CSV to stdout (and optionally to a file via `--output`) with columns:
- `sequence`: inferred haplotype sequence (with invariant positions restored)
- One column per input file (sample), giving the estimated frequency for that sample
- Final `SUM` row with column-wise sums (should be ~1.0 per sample)

### Options
- `-o, --output <FILE>`: Path to write the CSV (also printed to stdout)
- `-m, --mismatches <N>`: Max mismatches allowed in EM (default: 15)
- `--em-iterations <N>`: Max EM iterations (default: 20000)
- `--lambda1 <F>`: Recombination penalty weight (default: 0.0001)
- `--lambda2 <F>`: Complexity penalty weight for number of haplotypes (default: 0.0001)
- `-d, --error-rate <F>`: Sequencing error rate (default: 0.00001)
- `--sa-max-temperature <F>`: Starting max temperature for SA (default: 10.0)
- `--sa-iterations <N>`: SA iterations per run (default: 2000)
- `--sa-reruns <N>`: Number of SA reruns (default: 5)
- `--em-cdelta <F>`: EM convergence tolerance (default: 0.1)
- `--seed <U64>`: Random seed for reproducible runs (optional)

## How it works (high level)
1. Reads are parsed and invariant columns removed to speed up initialization.
2. A MAF (major-allele-frequency) haplotype is built, then a greedy pass proposes more haplotypes from reads.
3. Frequencies are initialized per sample and refined via EM.
4. Simulated Annealing explores structural edits (add/recombine/delete haplotypes).
5. After each structural step, SQUAREM estimates frequencies to local optimality.
6. Final haplotype set and per-sample frequencies are written as CSV.

## Tips
- Set `--seed` for deterministic behavior in tests and debugging.
- If frequencies look too diffuse or too sharp, tune `--error-rate`, `--lambda1`, and `--lambda2`.
- Large inputs: raise `--em-iterations` and/or `--sa-iterations`; expect higher runtime.
- Ensure all input reads are length-aligned; otherwise the program will exit early.

## Troubleshooting
- "Failed to open sample file": check file paths and permissions.
- "is not aligned": at least one file has reads with differing lengths. Re-align inputs.
- Empty output or single haplotype: inputs may be too uniform; verify your data.
