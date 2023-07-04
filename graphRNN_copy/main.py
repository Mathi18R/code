# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.decomposition import PCA
import logging
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from time import gmtime, strftime
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from random import shuffle
import pickle
#from tensorboard_logger import configure, log_value
import scipy.misc
import time as tm
import os
import onnx
import onnx_tf
import hls4ml
from args import Args
from model import *


def print_graphs():
    G_list = pickle.load(open('./graphs/GraphRNN_RNN_companies090_4_128_pred_9000_1.dat', 'rb'))
    print(len(G_list))
    print_amount = min(10, len(G_list))
    for i in range(print_amount):
        plt.figure()
        pos = nx.circular_layout(G_list[i])
        # pos = nx.spring_layout(G_list[i],k=1/np.sqrt(G_list[i].number_of_nodes()),iterations=100)
        nx.draw_networkx(G_list[i], pos=pos, with_labels=True)
        graph_info = "Node count: " + str(nx.number_of_nodes(G_list[i])) + " Edge count: " + str(nx.number_of_edges(G_list[i]))
        plt.text(-0.2, 1.1,graph_info)
        plt.axis('off')
        plt.margins(0, 0)
        plt.draw()
        plt.savefig("./output2/companies090_9000_" + str(i) + ".png", bbox_inches = 'tight', pad_inches = 0.25)
        plt.clf()

def print_graphs_080():
    G_list = pickle.load(open('./graphs/GraphRNN_RNN_companies080_4_128_pred_9000_1.dat', 'rb'))
    print(len(G_list))
    print_amount = min(10, len(G_list))
    for i in range(print_amount):
        G_list[i].remove_node(40)
        plt.figure()
        pos = nx.circular_layout(G_list[i])
        # pos = nx.spring_layout(G_list[i],k=1/np.sqrt(G_list[i].number_of_nodes()),iterations=100)
        nx.draw_networkx(G_list[i], pos=pos, with_labels=True)
        graph_info = "Node count: " + str(nx.number_of_nodes(G_list[i])) + " Edge count: " + str(nx.number_of_edges(G_list[i]))
        plt.text(-0.2, 1.1,graph_info)
        plt.axis('off')
        plt.margins(0, 0)
        plt.draw()
        plt.savefig("./output2/companies080_9000_" + str(i) + ".png", bbox_inches = 'tight', pad_inches = 0.25)
        plt.clf()

def shave_graphs_080():
    G_list = pickle.load(open('./graphs/GraphRNN_RNN_companies080_4_128_pred_9000_1.dat', 'rb'))
    for i in range(len(G_list)):
        G_list[i].remove_node(40)
    output_dir = './output/'
    fname = 'GraphRNN_RNN_companies080_4_128_pred_9000_40_nodes'
    fname = str(output_dir) + str(fname) + '.dat'
    with open(fname, "wb") as f:
        pickle.dump(G_list, f)


def synth_GRU_model():
    device = torch.device("cpu")

    # Initialize the right model
    if 'GraphRNN_VAE_conditional' in args.note:
        rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                        hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
                        has_output=False).cuda()
        output = MLP_VAE_conditional_plain(h_size=args.hidden_size_rnn, embedding_size=args.embedding_size_output,
                                           y_size=args.max_prev_node).cuda()
    elif 'GraphRNN_MLP' in args.note:
        rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                        hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
                        has_output=False).cuda()
        output = MLP_plain(h_size=args.hidden_size_rnn, embedding_size=args.embedding_size_output,
                           y_size=args.max_prev_node).cuda()
    elif 'GraphRNN_RNN' in args.note:
        rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                        hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
                        has_output=True, output_size=args.hidden_size_rnn_output)
        output = GRU_plain(input_size=1, embedding_size=args.embedding_size_rnn_output,
                           hidden_size=args.hidden_size_rnn_output, num_layers=args.num_layers, has_input=True,
                           has_output=True, output_size=1)
    # Load the PyTorch model
    rnn.load_state_dict(
        torch.load('./3-5-2023_12_38_run/model_save/GraphRNN_RNN_grid_4_128_lstm_400.dat', map_location='cpu'))
    print("named parameters:")
    print(rnn.named_parameters())
    # Convert the PyTorch model to ONNX
    input_shape = (args.batch_size, 1, args.max_prev_node)
    print(rnn)
    rnn.eval()
    dummy_input = torch.randn(input_shape)
    print("check2")
    onnx_model = torch.onnx.export(rnn, dummy_input, 'gru_model.onnx', verbose=True)

    print("check3")
    ## Convert the ONNX model to TensorFlow format
    tf_model = onnx_tf.backend.prepare(onnx_model, device='CPU')

## Convert the TensorFlow model to hls4ml-compatible format
# hls_config = hls4ml.utils.config_from_keras_model(tf_model, granularity='name')
# hls_config['Model'] = 'GRU'
# hls_model = hls4ml.converters.convert_from_keras_model(tf_model, hls_config)
#
## Save the hls4ml model to disk
# hls_model.compile()
# hls_model.build(csim=True, synth=True)
# hls_model.save('./3-5-2023_12_38_run/output/gru_model_hls4ml')


def read_sparse_txt(file_path):
    edge_list = []
    with open(file_path, "r") as filestream:
        for line in filestream:
            current_line = line.split(",")
            edge_list.append([int(current_line[0]), int(current_line[1])])
    G = nx.from_edgelist(edge_list)
    print("Edges before: " + str(nx.number_of_edges(G)))
    # make an undirected copy of the digraph
    UG = G.to_undirected()
    # extract subgraphs
    G_list_output = nx.connected_component_subgraphs(UG)
    return list(G_list_output)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    shave_graphs_080()

    print("check0")

    #args = Args()

    #synth_GRU_model()

    # G_list = read_sparse_txt('./DD/DD_A.txt')
    # print(len(G_list))
    # edge_sum = 0
    # for i in range(len(G_list)):
    #     if i < 10:
    #         plt.figure()
    #         nx.draw_networkx(G_list[i])
    #         graph_info = "Node count: " + str(nx.number_of_nodes(G_list[i])) + "\t Edge count: " + str(nx.number_of_edges(G_list[i]))
    #         plt.text(-0.2, 1.1, graph_info)
    #         plt.draw()
    #         plt.savefig("./3-5-2023_12_38_run/output/output1_" + str(i) + ".png")
    #         plt.clf()
    #     edge_sum+=nx.number_of_edges(G_list[i])
    # print("Edges after: " + str(edge_sum))

