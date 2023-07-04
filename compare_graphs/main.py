import pickle as pkl
import numpy as np
import networkx as nx
from scipy.linalg import toeplitz
import pyemd
import os
import subprocess as sp

def gaussian(x, y, sigma=1.0):
    dist = np.linalg.norm(x - y, 2)
    return np.exp(-dist * dist / (2 * sigma * sigma))

def gaussian_emd(x, y, sigma=1.0, distance_scaling=1.0):
    ''' Gaussian kernel with squared distance in exponential term replaced by EMD
    Args:
      x, y: 1D pmf of two distributions with the same support
      sigma: standard deviation
    '''
    support_size = max(len(x), len(y))
    d_mat = toeplitz(range(support_size)).astype(np.float)
    distance_mat = d_mat / distance_scaling

    # convert histogram values x and y to float, and make them equal len
    x = x.astype(np.float)
    y = y.astype(np.float)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    emd = pyemd.emd(x, y, distance_mat)
    return np.exp(-emd * emd / (2 * sigma * sigma))


def disc(samples1, samples2, kernel, is_parallel=True, *args, **kwargs):
    ''' Discrepancy between 2 samples
    '''
    d = 0
    for s1 in samples1:
        for s2 in samples2:
            d += kernel(s1, s2, *args, **kwargs)

    d /= len(samples1) * len(samples2)
    return d


def compute_mmd(samples1, samples2, kernel, is_hist=True, *args, **kwargs):
    ''' MMD between two samples
    '''
    # normalize histograms into pmf
    if is_hist:
        samples1 = [s1 / np.sum(s1) for s1 in samples1]
        samples2 = [s2 / np.sum(s2) for s2 in samples2]
    # print('===============================')
    # print('s1: ', disc(samples1, samples1, kernel, *args, **kwargs))
    # print('--------------------------')
    # print('s2: ', disc(samples2, samples2, kernel, *args, **kwargs))
    # print('--------------------------')
    # print('cross: ', disc(samples1, samples2, kernel, *args, **kwargs))
    # print('===============================')
    return disc(samples1, samples1, kernel, *args, **kwargs) + \
        disc(samples2, samples2, kernel, *args, **kwargs) - \
        2 * disc(samples1, samples2, kernel, *args, **kwargs)


def degree_stats(graph_ref_list, graph_pred_list, is_parallel=False):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    '''
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    for i in range(len(graph_ref_list)):
        degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
        sample_ref.append(degree_temp)
    for i in range(len(graph_pred_list_remove_empty)):
        degree_temp = np.array(nx.degree_histogram(graph_pred_list_remove_empty[i]))
        sample_pred.append(degree_temp)

    print(len(sample_ref), len(sample_pred))
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)

    return mmd_dist


def clustering_worker(param):
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(
        clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
    return hist


def clustering_stats(graph_ref_list, graph_pred_list, bins=100, is_parallel=True):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    for i in range(len(graph_ref_list)):
        clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
        hist, _ = np.histogram(
            clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
        sample_ref.append(hist)

    for i in range(len(graph_pred_list_remove_empty)):
        clustering_coeffs_list = list(nx.clustering(graph_pred_list_remove_empty[i]).values())
        hist, _ = np.histogram(
            clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
        sample_pred.append(hist)

    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd,
                           sigma=1.0 / 10, distance_scaling=bins)
    return mmd_dist

COUNT_START_STR = 'orbit counts: '

def edge_list_reindexed(G):
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
    for (u, v) in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges

def orca(graph):
    tmp_fname = './eval/orca/tmp.txt'
    f = open(tmp_fname, 'w')
    f.write(str(graph.number_of_nodes()) + ' ' + str(graph.number_of_edges()) + '\n')
    for (u, v) in edge_list_reindexed(graph):
        f.write(str(u) + ' ' + str(v) + '\n')
    f.close()

    output = sp.check_output(['./eval/orca/orca', 'node', '4', './eval/orca/tmp.txt', 'std'])
    output = output.decode('utf8').strip()
    idx = output.find(COUNT_START_STR)
    idx += len(COUNT_START_STR) + 2 # + 2 to remove the newline character
    output = output[idx:]
    node_orbit_counts = np.array([list(map(int, node_cnts.strip().split(' ')))
                                  for node_cnts in output.strip('\n').split('\n')])

    try:
        os.remove(tmp_fname)
    except OSError:
        pass

    return node_orbit_counts


def orbit_stats_all(graph_ref_list, graph_pred_list):
    total_counts_ref = []
    total_counts_pred = []

    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    for G in graph_ref_list:
        try:
            orbit_counts = orca(G)
        except Exception as error:
            print("An exception occurred:", error) #print exception
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_ref.append(orbit_counts_graph)

    crash = 0
    for G in graph_pred_list:
        orbit_counts = orca(G)
        # try:
        #     orbit_counts = orca(G)
        # except:
        #     crash+=1
        #     continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_pred.append(orbit_counts_graph)
    print(crash)

    total_counts_ref = np.array(total_counts_ref)
    total_counts_pred = np.array(total_counts_pred)
    print(total_counts_ref, total_counts_pred)
    print(len(total_counts_ref), len(total_counts_pred))
    mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=gaussian,
                           is_hist=False, sigma=30.0)

    print('-------------------------')
    print(np.sum(total_counts_ref, axis=0) / len(total_counts_ref))
    print('...')
    print(np.sum(total_counts_pred, axis=0) / len(total_counts_pred))
    print('-------------------------')
    return mmd_dist

def load_graph_list(fname,is_real=True):
    with open(fname, "rb") as f:
        graph_list = pkl.load(f)
    for i in range(len(graph_list)):
        edges_with_selfloops = graph_list[i].selfloop_edges()
        if len(edges_with_selfloops)>0:
            graph_list[i].remove_edges_from(edges_with_selfloops)

        graph_list[i] = max(nx.connected_component_subgraphs(graph_list[i]), key=len)
        graph_list[i] = nx.convert_node_labels_to_integers(graph_list[i])

    return graph_list

if __name__ == '__main__':
    graph_pred_list = load_graph_list('./src/GraphRNN_RNN_community4_4_128_pred_8400_1.dat')
    graph_ref_list = load_graph_list('./src/GraphRNN_RNN_community4_4_128_train_0.dat')
    print("MMD degree: " + str(degree_stats(graph_ref_list, graph_pred_list)))
    print("MMD clustering: " + str(clustering_stats(graph_ref_list, graph_pred_list)))
    print("MMD orbit: " + str(orbit_stats_all(graph_ref_list, graph_pred_list)))


