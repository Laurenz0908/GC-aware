def getReadClassifications(kraken,txids):

    """
    THIS IS NOT USED!!!!!!!!!!
    The function takes a Kraken read classification file and
    a vector of txids and extracts read IDs classified to the
    txids.
    :param kraken: .kraken output from classification
    :param paired: True if fastq contains paired end reads
    :param txids: vector with taxids
    :return: data frame containing
    """

    classification = {}
    for txid in txids:
        classification[txid] = []

    with open(kraken, 'r') as f:
        for line in f:
            line = line.split(sep="\t")

            if line[0] == "U" or int(line[2]) not in classification.keys():
                continue
            else:
                classification[int(line[2])].append(line[1])
    return classification