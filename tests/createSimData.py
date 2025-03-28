# This script will make full length reads (of the haplotype) with various
# haplotype sample frequencies. It will return both the haplotype frequencies
# and the read samples

### Future to do: add sequencing error and put into samples ###
from random import randrange
import numpy as np

# Use dictionary since we want to keep track of the names of each sequence
sequences = {}

with open("data/sampleHaps.aln2") as fp:
    seqname = fp.readline()
    while seqname:
        # print(seqname)
        sequence = fp.readline()
        # print(sequence)
        sequences[seqname.strip()] = sequence.strip()
        seqname = fp.readline()

sampleNum = 10 #for now
haploNum = 14 #for now

frequencyOutput = "data/simulated_haplotype_frequencies.csv"
readOutput = "data/simulated_reads.fa"

#may want to allow for flexible amount as its probably unrealistic to believe that
# samples will have equal number of reads
numReads = 100
readLength = 100 # or whatever length you want

seqNum = 0

frequencies = np.random.rand(haploNum, sampleNum)
frequencies = frequencies/frequencies.sum(axis = 0, keepdims=1)

# print(frequencies)

# np.savetxt(frequencyOutput, frequencies, delimiter=",")

with open(readOutput, 'w') as fp, open(frequencyOutput, 'w') as hp:
    for key in sequences:
        totHapReads = 0
        hp.write(key)
        for i in range(0,sampleNum):
            totalReads = int(round(frequencies[seqNum][i]*numReads))
            hp.write("," + str(totalReads/numReads))
            for k in range(0,totalReads):
                fp.write(f">read{key}-{str(totHapReads).zfill(2)} {i}\n")
                # fp.write(sequences[key])
                splitIndex = randrange(len(sequences[key])-readLength)
                for j in range(0, splitIndex):
                    fp.write('-')
                    # print('N', end='')
                for j in range(splitIndex, splitIndex+readLength):
                    fp.write(sequences[key][j])
                    # print(sequences[key][j], end='')
                for j in range(splitIndex+readLength, len(sequences[key])):
                    fp.write('-')
                    # print('N', end='')
                fp.write('\n')
                totHapReads = totHapReads + 1
        hp.write('\n')
        seqNum = seqNum + 1
