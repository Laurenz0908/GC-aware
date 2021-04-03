import os
from datetime import datetime
import pandas as pd
import numpy as np
import math
import subprocess


def print_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)


def get_taxids(kraken, threshold=100000, taxonomy = "R7"):

    """
    extract taxids of organisms that have a higher number
    of reads than specified in THRESHOLD
    :param kraken: kraken report file
    :param threshold: lowest read number of organisms to be quantified
    shouldn't be too low gc of coverage limitation
    :return: vector containing the Taxids
    """
    df = pd.read_csv(kraken, sep="\t")
    df_sub = df.loc[(df.iloc[:, 1] > threshold) & (df.iloc[:, 3] == taxonomy), :]
    return np.unique(df_sub.iloc[:, 4])


def get_refgenomes(lib_path, txids, bin_size=None, write_kmers=False):

    """
    This function extracts reference genomes as fna files
    from a Kraken database file (library.fna) based on taxonomy IDs
    The files are written as fastas to refgenomes/
    :param lib_path: path to library.fna from Kraken database
    :param txids: vector containing taxonomy IDs of the genomes to be extracted
    :param bin_size: length of k-mer to calculate GC from
    :param write_kmers: write kmers as fasta for kraken correction
    :return: None
    """
    txids_pipe = ["|" + x + "|" for x in map(str, txids)]

    if not os.path.exists("refgenomes"):
        os.mkdir('refgenomes')

    if bin_size is not None:
        occ = np.zeros((bin_size, len(txids)), dtype=int)
        idx = dict(zip(txids, range(len(txids))))

    # open files to write lines to
    dct = {}
    for txid in txids:
        dct[str(txid)] = open("refgenomes/" + str(txid) + ".fna", 'a')

    if write_kmers:
        kmers = open("refgenomes/" + str(bin_size) + "mers.fasta", 'a')

    with open(lib_path, 'r') as f:

        is_txid = False
        for line in f:

            if ">" not in line and not is_txid:
                continue

            id_in_line = [x in line for x in txids_pipe]
            if ">" in line and not any(id_in_line):
                is_txid = False
                ln_before = None

            elif ">" in line and any(id_in_line):
                is_txid = True
                ind_id = [i for i,val in enumerate(id_in_line) if val]
                txid = txids[ind_id[0]]
                dct[str(txid)].write(line)
                ln_before = None
                i = 1

            elif is_txid:
                txid = txids[ind_id[0]]

                if bin_size is not None and ln_before is not None:
                    seq = line.strip() + ln_before
                    # get GC of all k-mers
                    for i in range(len(seq)-bin_size+1):
                        kmer = seq.upper()[i: i+bin_size]
                        bases = set('ATGC')
                        if any((c not in bases) for c in kmer):
                            continue
                        gc = kmer.count("G") + kmer.count("C")
                        gc_content = round((gc/bin_size)*100)
                        occ[gc_content, idx[int(txid)]] += 1

                        if write_kmers:
                            read_name = ">txid|" + str(txid) + "|" + str(i)
                            kmers.write(read_name + '\n')
                            kmers.write(kmer + '\n')

                ln_before = line.strip()
                dct[str(txid)].write(line)

    for txid in txids:
        dct[str(txid)].close()

    if write_kmers:
        for txid in txids:
            kmers.close()

    if bin_size is not None:
        header = "\t".join(map(str, txids))
        np.savetxt("refgenomes/references_bin_" + str(bin_size) + ".dist", occ,
                   fmt='%i', delimiter="\t", header=header)


def distribute_reads(kraken, fastq, txids, paired, bin_size=None, write=True, fasta=False):

    """
    Distributes reads from a metagenomic sample according to classification
    from Kraken2 into separate fastq files
    :param kraken: path to .kraken file
    :param fastq: path to fastq from a metagenomic sample
    :param txids: vector of taxonomy IDs of organisms to be extracted
    :param paired: true if paired end reads
    :param bin_size: no of bases to be binned for GC
    :param write: True if reads should be written to separate fastq files
    :param fasta
    (for mapping pipeline)
    :return: None
    """
    if paired:
        j = 8

    elif fasta:
        j = 2
    else:
        j = 4

    if not os.path.exists("class_reads"):
        os.mkdir("class_reads")

    if write:
        # open files to write reads to
        dct = {}
        for txid in txids:
            dct[str(txid)] = open("class_reads/reads_" + str(txid) + ".fastq", 'a')

    i = 0
    if bin_size is not None:
        occ = np.zeros((bin_size, len(txids)), dtype=int)
        idx = dict(zip(txids, range(len(txids))))

    with open(fastq, 'r') as fq, open(kraken, 'r') as kr:

        while kr:
            ln_fq = fq.readline()

            # in paired reads one kraken line corresponds to 8 fastq lines
            # if not paired to 4 fastq lines
            if i % j == 0:
                ln_kr = kr.readline()

            if ln_fq.strip() == "" or ln_kr.strip() == "":
                break

            current_txid = ln_kr.split(sep="\t")[2]

            if int(current_txid) in txids:
                if write:
                    dct[current_txid].write(ln_fq)

                if bin_size is not None and ((i-1) % 4 == 0 or i == 1):
                    seq = ln_fq.strip().upper()
                    gc = seq.count('G') + seq.count('C')
                    gc_content = math.floor((bin_size / len(seq)) * (gc + np.random.uniform()))
                    occ[gc_content, idx[int(current_txid)]] += 1

            i += 1

    if write:
        for txid in txids:
            dct[str(txid)].close()

    if bin_size is not None:
        header = "\t".join(map(str, txids))
        np.savetxt("class_reads/sample_bin_" + str(bin_size) + ".dist", occ,
                   fmt='%i', delimiter="\t", header=header)


def map_reads(txid):
    """
    maps the reads from a fastq file (from distribute_reads()) to the
    corresponding reference genome of the given txid
    Creates sorted bam and pileup file in mapped_bt/
    :param txid: taxonomy ID(s) of the organism, can be vector
    :return: None
    """

    if not np.isscalar(txid):
        for x in txid:
            map_reads(txid=x)
        return None

    if not os.path.exists("mapped_bt"):
        os.mkdir("mapped_bt")
        os.mkdir("mapped_bt/idx")
        os.mkdir("mapped_bt/pileup")

    refgen = "refgenomes/" + str(txid) + ".fna"
    bt_index = "mapped_bt/idx/" + str(txid)
    subprocess.run(["bowtie2-build", refgen, bt_index], stdout=subprocess.PIPE)

    read_file = "class_reads/reads_" + str(txid) + ".fastq"
    out_file = "mapped_bt/" + str(txid) + ".sam"
    subprocess.run(["bowtie2", "-x", bt_index, "--interleaved", read_file,
                    "--no-unal", "-p2", "-S", out_file], stdout=subprocess.PIPE)
    sorted_file = "mapped_bt/" + str(txid) + "_sorted.sam"
    subprocess.run(["samtools", "sort", "-o", sorted_file, out_file],
                   stdout=subprocess.PIPE)
    os.remove(out_file)
    os.rename(sorted_file, out_file)

    pileup_file = "mapped_bt/pileup/" + str(txid) + "_pileup.txt"
    cmd = "samtools mpileup -A -f " + refgen + " " + out_file + " | cut -f-4" + " > " + pileup_file
    with open(pileup_file, 'w') as f:
        subprocess.Popen(cmd, shell=True, stdout=f)


print("START")
print_time()
os.chdir('/home/laurenz/Documents/Uni/phd/hmp')
taxids = get_taxids(kraken="simsample_report.txt")
get_refgenomes(lib_path="library.fna", txids=taxids, bin_size=100, write_kmers=True)
print("refgenomes extracted")
print_time()

"""
fastq = "fastq/simsample.fastq"
kraken = "simsample.kraken"
distribute_reads(kraken, fastq, txids=taxids, paired=True, bin_size=100, write=True)
print("reads distributed")
print_time()

map_reads(taxids)
print("DONE")
print_time()
"""


