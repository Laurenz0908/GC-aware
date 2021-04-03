import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import os
import seaborn as sns
import qpsolvers as qps


def print_time():
    now = datetime.now()
    current_time = now.strftime('%H:%M:%S')
    print("Current Time =", current_time)


def bin_coverage(pileup, window_size, bases=None):
    """
    bin coverage of a genome into windows of certain length
    :param pileup: output of samtools pileup
    :param window_size: no. of bases in each window
    :param bases: bases of which the content should be calculated
    :return: np vectors with: relative ratios of $bases in the
    window (G+C by default), mean coverage of window, sd of
    coverage
    """
    if bases is None:
        bases = ['G', 'C']

    df = pd.read_csv(pileup, sep="\t")
    isgc = [1 if x in bases else 0 for x in df.iloc[:, 2]]
    ind = range(0, df.shape[0] - 1, window_size)

    gc = np.empty(len(ind), dtype=int)
    cov_mean = np.empty(len(ind))
    cov_sd = np.empty(len(ind))
    for i, j in zip(ind, range(0, len(ind))):
        gc[j] = np.mean(isgc[i: i + window_size - 1]) * 100
        cov_mean[j] = np.mean(df.iloc[i:i + window_size - 1, 3])
        cov_sd[j] = np.std(df.iloc[i:i + window_size - 1, 3])

    return gc, cov_mean, cov_sd


def bin_coverage_organisms(pileups, window_size=400, bases=None):
    """
    Bins coverage of all organisms into genome segments of defined
    lengths of bases
    :param pileups: paths to pileups of organisms
    :param window_size: bases contained in one bin
    :param bases: nucleotides of which the relative ratio should be
    calculated (default=G+C)
    :return: pd dataframe which each line representing one bin, columns
    contain relative ratio of $bases, mean coverage of bin, sd of
    coverage and organism
    """

    df = pd.DataFrame()
    for file in pileups:
        gc, cov_mean, cov_sd = bin_coverage(file, window_size=window_size,
                                            bases=bases)
        organism = int(file.split(sep='_')[0])
        temp = pd.DataFrame(
            {
                'GC_content': gc,
                'cov_mean': cov_mean,
                'cov_sd': cov_sd,
                'organism': organism
            }
        )
        df = pd.concat([df, temp])

    return df.sort_values(by=['organism', 'GC_content'])


def plot_bins(all_binned, mean=False, save=None):
    """
    plots the coverage of the bins of a metagenomic sample
    :param all_binned: pd dataframe as returned by
    bin_coverage_organisms()
    :param mean: True if bins should be
    :param save: file path for plot, if None it will be
    output in new window
    :return: Plot
    """
    organisms = all_binned['organism'].unique()
    rgb_values = sns.color_palette('Set2', len(organisms))
    color_map = dict(zip(organisms, rgb_values))

    if mean:
        grouped = all_binned.groupby(['organism', 'GC_content'])
        df_mean = grouped[['cov_mean']].aggregate(np.mean)

        #fig, ax = plt.subplots(figsize=(8, 6))
        plt.scatter(df_mean.index.get_level_values(1), df_mean['cov_mean'],
                    c=df_mean.index.get_level_values(0).map(color_map))

    else:
        plt.scatter(all_binned['GC_content'], all_binned['cov_mean'],
                    c=all_binned['organism'].map(color_map))


def trim_ends(all_binned, quantiles):
    """
    for each organism in the dataframe the function cuts off the bins
    on the extremes of the organisms GC distribution, this is done
    because they typically contain few data and are thus prone to
    outliers
    :param all_binned: dataframe as returned by "bin_coverage_organisms()"
    :param quantiles: 1D array of length 2 setting relative bounds for
    trimming
    :return: trimmed pd dataframe
    """

    all_binned_new = all_binned.copy()
    for organism in all_binned_new['organism'].unique():
        gc = all_binned_new[all_binned_new['organism'] == organism]['GC_content']
        quants = np.quantile(gc, quantiles)
        ind_to_drop = all_binned_new[((all_binned_new['GC_content'] < quants[0]) |
                                      (all_binned_new['GC_content'] > quants[1])) &
                                     (all_binned_new['organism'] == organism)].index
        all_binned_new.drop(ind_to_drop, inplace=True)

    return all_binned_new.sort_values(by=['organism', 'GC_content'],
                                      ignore_index=True)


def get_mean_coverages(all_binned):
    """
    Calculates mean coverages of organisms from data frame as returned by
    bin_coverage_organisms()
    :param all_binned: data frame with binned coverages of organisms
    :return: array with mean coverages (multiIndexed)
    """

    grouped = all_binned.groupby(['organism'])
    return grouped[['cov_mean']].aggregate(np.mean)


def get_overlaps_mapping(all_binned, redundant=True):
    """
    gets all overlaps in GC regions between organisms
    :param all_binned: data frame with binned GC regions as returned
    by bin_coverage_organisms
    :param redundant: True if (org1,org2) should be listed as well
    as (org2,org1)
    :return: Numpy array of arrays where each entry contains
    """

    orgs = all_binned['organism'].unique()
    overlaps = []
    for i in range(0, len(orgs)):
        overlaps.append([])
        gc_org1 = all_binned['GC_content'][all_binned['organism'] == orgs[i]].unique()
        for j in range(0, len(orgs)):
            if not redundant and i >= j:
                continue
            gc_org2 = all_binned['GC_content'][all_binned['organism'] == orgs[j]].unique()
            overlaps[i].append([x for x in gc_org1 if x in gc_org2])
    return overlaps


def get_weights_mapping(all_binned, org1, org2, gc):
    """
    computes the weights w_ijb of 2 organisms for the least squares function
    :param all_binned: data frame containing binned organisms and coverages
    :param org1, org2: the 2 organisms as strings
    :param gc: vector with the overlapping GC bins
    :return: vector with the weights (each element corresponding to one GC bin)
    """
    counts1 = all_binned[all_binned['organism'] == org1]['GC_content'].value_counts()
    counts2 = all_binned[all_binned['organism'] == org2]['GC_content'].value_counts()
    bins = [b for b in counts1.keys() if b in gc and b in counts2.keys()]
    o_1 = counts1[counts1.keys().isin(bins)]
    o_2 = counts2[counts2.keys().isin(bins)]
    return (o_1 / sum(o_1) + o_2 / sum(o_2)) / 2


def get_beta_mapping(all_binned, org1, org2, gc):
    """
    calculates the beta for two organisms for the LS matrix
    :param all_binned: data frame containing binned organisms and coverages
    :param org1: first organism as string (txid)
    :param org2: second organsim as string (txid)
    :param gc: vector of overlapping GC bins
    :return: beta_ij
    """
    x_1 = all_binned[(all_binned['organism'] == org1) & (all_binned['GC_content'].isin(gc))]
    grouped = x_1.groupby('GC_content')
    x_1 = grouped[['cov_mean']].aggregate(np.mean)
    x_2 = all_binned[(all_binned['organism'] == org2) & (all_binned['GC_content'].isin(gc))]
    grouped = x_2.groupby('GC_content')
    x_2 = grouped[['cov_mean']].aggregate(np.mean)
    w = get_weights_mapping(all_binned=all_binned, org1=org1, org2=org2, gc=gc)
    return sum(w * x_1['cov_mean'] * x_2['cov_mean'])


def get_alpha_mapping(all_binned, org, overlaps):
    """
    return alpha value for LS matrix for one organism
    :param all_binned: data frame containing binned organisms and coverages
    :param org: corresponding organism as string (txid)
    :param overlaps: list with all overlaps of org
    :return: alpha value (float)
    """
    inner_sum = np.zeros(len(overlaps))
    orgs = all_binned['organism'].unique()
    for i in range(0, len(overlaps)):
        org2 = orgs[i]
        if org2 == org:
            continue
        w = get_weights_mapping(all_binned=all_binned, org1=org,
                                org2=org2, gc=overlaps[i])
        x = all_binned[(all_binned['organism'] == org) & (all_binned['GC_content'].isin(overlaps[i]))]
        grouped = x.groupby('GC_content')
        xsq = np.square(grouped[['cov_mean']].aggregate(np.mean)['cov_mean'])
        inner_sum[i] = np.sum(w * xsq)

    return np.sum(inner_sum)


def get_matrix_mapping(all_binned):
    """
    returns a matrix to minimize the least squares problem
    via quadratic programming algorithm
    :param all_binned: data frame containing binned organisms and coverages
    :return: matrix with alpha and beta values (n x n for n organisms in data frame)
    """
    all_overlaps = get_overlaps_mapping(all_binned=all_binned)
    orgs = all_binned['organism'].unique()
    a = np.zeros(len(all_overlaps))
    for i in range(0, len(all_overlaps)):
        a[i] = get_alpha_mapping(all_binned=all_binned, org=orgs[i], overlaps=all_overlaps[i])

    b = np.zeros((len(all_overlaps), len(all_overlaps)))
    for row in range(0, b.shape[0]):
        for col in range(0, b.shape[0]):
            if col < row:
                b[row, col] = b[col, row]

            elif row == col:
                b[row, col] = a[row]

            else:
                b[row, col] = -get_beta_mapping(all_binned, org1=orgs[row], org2=orgs[col],
                                                gc=all_overlaps[row][col])
    return b


def get_corrected_abundances(all_binned, mapping=False):
    """
    takes a coverage data.frame and the matrix as input for the
    quadratic problem solver, minimizes the LS function and returns
    the corrected abundance vector
    :param all_binned: data frame containing binned organisms and coverages
    :param mapping: True if the version using read mapping should be used
    :return: np vector with organism abundances, ordered as in all_binned
    """
    if mapping:
        df_trim = trim_ends(all_binned=all_binned, quantiles=[0.025, 0.975])
        m = get_matrix_mapping(all_binned=df_trim)
        grouped = df_trim.groupby('organism')
        means = grouped[['cov_mean']].aggregate(np.mean)
        means = means['cov_mean'] / np.sum(means['cov_mean'])

    a = np.full((len(means),), 1.)

    return qps.solve_qp(P=m, q=np.array(means), A=a, b=np.array([1.]))


def trim_dist(dist, quantiles=None):
    """
    cuts of bins on extreme ends of gc dist
    :param dist: array with gc dist
    :param quantiles: quantiles within which values are kept
    :return: array with trimmed gc distribution
    """
    if quantiles is None:
        quantiles = [0.01, 0.99]
    limits = [quantiles[0] * sum(dist), quantiles[1] * sum(dist)]
    dist[(dist < limits[0]) | (dist > limits[1])] = 0
    return dist


def normalize_gc_dists(sample_dist, ref_dist, trim=True):
    """
    normalizes a read gc distribution with an expected distributiom from
    reference sequences
    :param sample_dist: path to the sample distribution (from distribute_reads() )
    :param ref_dist: path to the reference distribution (from get_refgenomes() )
    :param trim: cuts off reads at extreme ends of GC dist (often outliers)
    :return: matrix with the normalized distributions and reference distributions
    """
    sample_dist = np.loadtxt(s_dist, dtype=int, delimiter="\t")
    if trim:
        sample_dist = pd.DataFrame(sample_dist).apply(trim_dist, result_type='broadcast')
    ref_dist = np.loadtxt(r_dist, dtype=int, delimiter="\t")
    ref_dist = ref_dist + 1
    return np.array(sample_dist) / ref_dist, ref_dist - 1


def get_overlaps(norm_dist):
    """
    return list of list with overlaps of gc distributions of organisms
    :param norm_dist: matrix with normalized gc distributions of organisms
    :return: list of lists containing arrays with pairwise overlaps of organisms
    """
    overlaps = []
    for i in range(0, norm_dist.shape[1]):
        org1 = np.where(norm_dist[:, i] != 0)[0]
        overlaps.append([])
        for j in range(0, norm_dist.shape[1]):
            org2 = np.where(norm_dist[:, j] != 0)[0]
            overlaps[i].append([x for x in org1 if x in org2])

    return overlaps


def get_weights(ref_dist, org1, org2, all_overlaps):
    """
    returns weights of 2 organisms
    :param ref_dist: reference distributions
    :param org1: first org (index!)
    :param org2: second org (index!)
    :param all_overlaps: list of list as from get_overlaps()
    :return: array containing overlapping GC bins or the orgs
    """
    ind = all_overlaps[org1][org2]
    o1 = ref_dist[ind, org1]
    o2 = ref_dist[ind, org2]
    return 0.5 * o1 / np.sum(o1) + 0.5 * o2 / np.sum(o2)


def get_beta(norm_dist, ref_dist, org1, org2, all_overlaps):
    """
    returns beta_ij for LS matrix
    :param norm_dist: matrix with normalized gc distributions of organisms
    :param ref_dist: reference distributions
    :param org1: first org (index!)
    :param org2: second org (index!)
    :param all_overlaps: list of list as from get_overlaps()
    :return:
    """
    gc = all_overlaps[org1][org2]
    if len(gc) == 0:
        return 0
    w = get_weights(ref_dist=ref_dist, org1=org1, org2=org2, all_overlaps=all_overlaps)
    return np.sum(w * norm_dist[gc, org1] * norm_dist[gc, org2])


def get_alpha(norm_dist, ref_dist, org, all_overlaps):
    """
    returns alpha_i for LS matrix
    :param norm_dist: matrix with normalized gc distributions of organisms
    :param ref_dist: reference distributions
    :param org: organism index (integer)
    :param all_overlaps: list of list as from get_overlaps()
    :return: alpha_i
    """
    overlaps = all_overlaps[org]
    inner_sum = np.zeros(len(overlaps))
    for i in range(0, len(overlaps)):
        if i == org:
            inner_sum[i] = 0
            continue
        w = get_weights(ref_dist=ref_dist, org1=org, org2=i, all_overlaps=all_overlaps)
        xsq = np.square(norm_dist[overlaps[i], org])
        inner_sum[i] = np.sum(w * xsq)

    return np.sum(inner_sum)


def get_matrix(norm_dist, ref_dist):
    """
    return matrix for no_mapping pipeline to minimize LS function
    :param norm_dist: matrix with normalized gc distributions of organisms
    :param ref_dist: reference distributions
    :return: matrix
    """
    all_overlaps = get_overlaps(norm_dist=norm_dist)
    a = np.zeros(len(all_overlaps))
    for i in range(0, len(all_overlaps)):
        a[i] = get_alpha(norm_dist=norm_dist, ref_dist=ref_dist, org=i,
                         all_overlaps=all_overlaps)

    b = np.zeros((len(all_overlaps), len(all_overlaps)))
    for row in range(0, len(all_overlaps)):
        for col in range(0, len(all_overlaps)):

            if row == col:
                b[row, col] = a[row]

            elif row > col:
                b[row, col] = b[col, row]

            else:
                b[row, col] = -get_beta(norm_dist=norm_dist, ref_dist=ref_dist,
                                        org1=row, org2=col, all_overlaps=all_overlaps)
    return b


def corrected_abundances(sample_path, reference_path):
    """
    return non gc biased abundances via no mapping pipeline
    :param sample_path: path to sample distribution file
    :param reference_path: path to reference distribution file
    :return: abundance vector (ordered by txid of organism)
    """
    norm_dist, ref_dist = normalize_gc_dists(sample_path, reference_path, trim=True)
    m = get_matrix(norm_dist, ref_dist)
    means = np.apply_along_axis(np.mean, 0, norm_dist)
    means = means / np.sum(means)
    a = np.full((len(means),), 1.)
    print(m)
    return qps.solve_qp(P=m, q=np.array(means), A=a, b=np.array([1.]))


def plot_dist(dist):
    """
    plots the coverage of the bins of a metagenomic sample
    :param dist: gc distrubiton matrix
    :return: Plot
    """
    rgb_values = sns.color_palette('Set2', dist.shape[1])
    color_map = dict(zip(range(0, dist.shape[1]), rgb_values))

    x = np.tile(range(0, dist.shape[0]), dist.shape[1])
    y = dist.reshape(dist.shape[0] * dist.shape[1], order='F')
    col = list(np.repeat(range(0, dist.shape[1]), dist.shape[0]))
    plt.scatter(x, y, c=list(map(color_map.get, col)))


print_time()

path = '/home/laurenz/Documents/Uni/phd/hmp/'
os.chdir(path)
s_dist = "class_reads/sample_bin_100.dist"
r_dist = "refgenomes/references_bin_100.dist"

norm, refdist = normalize_gc_dists(s_dist, r_dist, trim=True)
norm[0][0] = 0
plot_dist(norm)
plt.show()

all_overlaps_ = get_overlaps(norm)
lambdas_nm = corrected_abundances(s_dist, r_dist)
plot_dist(norm * lambdas_nm)
plt.show()

print_time()
"""
path = '/home/laurenz/Documents/Uni/phd/hmp/mapped_bt/pileup'
os.chdir(path)
files = os.listdir(path=path)
pileups = [file for file in files if "pileup" in file]
df = bin_coverage_organisms(pileups=pileups)

df_trim = trim_ends(all_binned=df, quantiles=[0.025, 0.975])
lambdas_mapping = get_corrected_abundances(all_binned=df_trim, mapping=True)

# from R code:
# lambdas_mapping = np.array([0.25, 0.53, 0.22])

plot_bins(all_binned=df_trim, mean=True)
plt.show()
print(get_matrix_mapping(all_binned=df_trim))
print(get_mean_coverages(all_binned=df_trim))
lam = np.repeat(np.array(lambdas_mapping), np.array(df_trim['organism'].value_counts()))
df_trim['cov_mean'] = df_trim['cov_mean'] * lam
plot_bins(all_binned=df_trim, mean=True)
plt.show()
"""
