# This script will make full length reads (of the haplotype) with various
# haplotype sample frequencies. It will return both the haplotype frequencies
# and the read samples

# Future to do: add sequencing error and put into samples ###
from random import randrange
import numpy as np

# Use dictionary since we want to keep track of the names of each sequence
frequencyOutput = "data/simulated_haplotype_frequencies.csv"
# readOutput = "data/simulated_reads.fa"

# may want to allow for flexible amount as its probably unrealistic to believe that
# samples will have equal number of reads
numReads = 1000  # to change/maybe add some randomness to this
readLength = 500  # or whatever length you want
sampleNum = 20

frequencies = list()
sequences = list()

# Under assumption that testSet haplotypes and sampleFrequencies haplotypes are in the same order
with open("data/hapFreqs_og.csv") as fp:
    line = fp.readline()
    while line:
        data = line.strip().split(",")
        sequences.append(data[0])   # should get the haplotype
        data = data[1:sampleNum+1]  # [0] is the haplotype and so end is samplenum + 1
        data = [float(i) for i in data]
        frequencies.append(data)
        line = fp.readline()

# frequencies = np.random.rand(haploNum, sampleNum)
frequencies = np.array([np.array(xi) for xi in frequencies])
frequencies = frequencies/frequencies.sum(axis=0, keepdims=1)

# print(frequencies)

hapNum = len(sequences)

# np.savetxt(frequencyOutput, frequencies, delimiter=",")

# Need to rewrite this loop to match lists of sequences now
with open(frequencyOutput, 'w') as hp:
    for pos, key in enumerate(sequences):
        totHapReads = 0
        hp.write(key)
        for i in range(0, sampleNum):
            totalReads = int(round(frequencies[pos][i]*numReads))
            hp.write("," + str(totalReads/numReads))
            readOutput = f"data/simulated_reads_{i}.fa"
            for k in range(0, totalReads):
                with open(readOutput, "a") as fp:
                    # fp.write(str(pos) + "-" + str(totHapReads).zfill(2) + " " + str(i) + "\n")
                    fp.write(f">read{pos}-{str(totHapReads).zfill(2)} {i}\n")
                    # fp.write(sequences[key])
                    # splitIndex = 0
                    if (len(key) == readLength):
                        splitIndex = 0
                    else:
                        splitIndex = randrange(len(key)-readLength)
                    for j in range(0, splitIndex):
                        fp.write('-')
                        # print('N', end='')
                    for j in range(splitIndex, splitIndex+readLength):
                        fp.write(sequences[pos][j])
                        # print(sequences[key][j], end='')
                    for j in range(splitIndex+readLength, len(key)):
                        fp.write('-')
                        # print('N', end='')
                    fp.write('\n')
                    totHapReads = totHapReads + 1
        hp.write('\n')
