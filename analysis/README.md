# Analysis

Contains a self-contained parameterized Jupyter notebook for post-analysis of the estimated results from `haplmate` and the original haplotype frequencies(when population is known) to verify the results of `haplmate`.

Contents:
- y: log-likelihood, x: iterations, see if log-likelihood converged and estimated correctly
- y: frequency, x: haplotype, each data point has true frequency, estimated frequency, and average estimated frequency for all samples
- Graph of the connections between the original and estimated haplotypes
    - True Positive: Original exists, estimated correctly
    - False Positive: Original does not exist, estimated random haplotype
    - True Negative: Original exists, did not estimate
    - False Negative: Original does not exist, did not estimate(can't track)

## Usage

~~~bash
# TBD
~~~
