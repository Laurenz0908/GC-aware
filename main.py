import getRefgenomes_from_krakendb as preprocess
import python_lib as lib
import subprocess


def kraken_correct(kraken, lib_path, fastq, threshold=10000, bin_size=None, taxonomy="R7"):
    """
    pipeline for kraken corrected abundance estimation without read mapping
    :param kraken:
    :param threshold:
    :param bin_size:
    :param taxonomy:
    :param lib_path:
    :param fastq:
    :return:
    """

    taxids = preprocess.get_taxids(kraken=kraken, threshold=threshold, taxonomy=taxonomy)
    preprocess.get_refgenomes(lib_path=lib_path, txids=taxids, bin_size=bin_size, write_kmers=True)
    preprocess.distribute_reads(kraken=kraken, fastq=fastq, bin_size=bin_size, write=False, txids=taxids)

    cmd = "kraken2 --db kraken/uhgg --threads 20 --report k_correct.report refgenomes/"
    cmd = cmd + str(bin_size) + "mers.fasta > k_correct.kraken"
    subprocess.Popen(cmd, shell=True)

    preprocess.distribute_reads(kraken="k_correct.kraken", txids=taxids, fasta=True, bin_size=bin_size)

    cmd = "kraken2 --db kraken/uhgg --threads 20 --report sample.report --paired " + str(fastq[0])
    cmd = cmd + " " + str(fastq[1]) + " > sample.kraken"
    subprocess.Popen(cmd, shell=True)
