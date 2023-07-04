# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
# import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchsummary import summary
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
# from tensorboard_logger import configure, log_value
import scipy.misc
import time as tm
import os
import onnx
import onnx_tf
from onnx2keras import onnx_to_keras
# import tf2onnx
import hls4ml
from args import Args
from model import *
import tensorflow as tf
from pt2keras import Pt2Keras
from pytorch2keras import pytorch_to_keras
import yaml
import traceback


def get_pt_model():
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
        torch.load('./3-5-2023_12_38_run/model_save/GraphRNN_RNN_companies090_4_128_lstm_9000.dat', map_location='cpu'))

    output.load_state_dict(
        torch.load('./3-5-2023_12_38_run/model_save/GraphRNN_RNN_companies090_4_128_output_9000.dat',
                   map_location='cpu'))

    print("named parameters:")
    print(rnn.named_parameters())
    print(output.named_parameters())

    return rnn, output

def my_pytorch_to_onnx():
    rnn, output = get_pt_model()

    # Convert the PyTorch model to ONNX
    input_shape_rnn = (args.batch_size, 1, args.max_prev_node)
    input_shape_output = (args.batch_size, 1, 1)
    print(rnn)
    print(output)
    rnn.eval()
    output.eval()
    dummy_input_rnn = torch.randn(input_shape_rnn)
    dummy_input_output = torch.randn(input_shape_output)
    try:
        rnn_onnx_model = torch.onnx.export(rnn, dummy_input_rnn, 'rnn_model.onnx', opset_version=11, verbose=True,
                                           input_names=['input'], output_names=['output'])
    except Exception as error:
        # handle the exception
        # print("An exception occurred:", error)  # An exception occurred: division by zero
        print(traceback.format_exc())
    try:
        output_onnx_model = torch.onnx.export(output, dummy_input_output, 'output_model.onnx', opset_version=11,
                                              verbose=True, input_names=['input'], output_names=['output'])
    except Exception as error:
        # handle the exception
        # print("An exception occurred:", error) # An exception occurred: division by zero
        print(traceback.format_exc())

#RuntimeError: GRU with linear_before_reset is not supported in Tensorflow.
def my_onnx_to_tf():
    rnn_onnx_model_path = './rnn_model.onnx'
    rnn_onnx_model = onnx.load(rnn_onnx_model_path)
    rnn_tf_model = onnx_tf.backend.prepare(rnn_onnx_model, device='CPU')
    rnn_tf_model.save('./rnn_tf/rnn_model')

    output_onnx_model_path = './output_model.onnx'
    output_onnx_model = onnx.load(output_onnx_model_path)
    output_tf_model = onnx_tf.backend.prepare(output_onnx_model, device='CPU')
    output_tf_model.save('./output_tf/output_model')

###code remnants:
# Convert the ONNX model to TensorFlow format
# tf_model_path = './test'
# tf_model = tf2onnx.convert.from_model(onnx_model, opset=11, output_path=tf_model_path)

## Convert the TensorFlow model to hls4ml-compatible format
# hls_config = hls4ml.utils.config_from_keras_model(tf_model, granularity='name')
# hls_config['Model'] = 'GRU'
# hls_model = hls4ml.converters.convert_from_keras_model(tf_model, hls_config)
#
## Save the hls4ml model to disk
# hls_model.compile()
# hls_model.build(csim=True, synth=True)
# hls_model.save('./3-5-2023_12_38_run/output/gru_model_hls4ml')
###end code remnants

#KeyError: 'ConstantOfShape'
def my_onnx_to_keras():
    rnn_onnx_model_path = './rnn_model.onnx'
    rnn_onnx_model = onnx.load(rnn_onnx_model_path)

    ## Convert the ONNX model to Keras format
    rnn_k_model = onnx_to_keras(rnn_onnx_model, ['input'])
    rnn_k_model.summary()
    rnn_k_model.save('rnn_keras')

    output_onnx_model_path = './output_model.onnx'
    output_onnx_model = onnx.load(output_onnx_model_path)

    ## Convert the ONNX model to Keras format
    output_k_model = onnx_to_keras(output_onnx_model, ['input'])
    output_k_model.summary()
    output_k_model.save('output_keras')

#Exception: Unsupported layer GRU
def my_pytorch_to_hls4ml():
    rnn, output = get_pt_model()

    # Convert the PyTorch model to ONNX
    input_shape_rnn = (args.batch_size, 1, args.max_prev_node)
    input_shape_output = (args.batch_size, 1, args.hidden_size_rnn_output)
    print(rnn)
    print(output)
    rnn.eval()
    output.eval()
    dummy_input_rnn = torch.randn(input_shape_rnn)
    dummy_input_output = torch.randn(input_shape_output)

    config = hls4ml.utils.config_from_pytorch_model(rnn)

    print("-----------------------------------")
    print("Configuration")
    print(config)
    print("-----------------------------------")

    hls_model = hls4ml.converters.convert_from_pytorch_model(rnn,
                                                             input_shape=input_shape_rnn,
                                                             output_dir='C:/Users/matth/PycharmProjects/pytorch-onnx-tensorflow-hls4ml/hls4ml-output/',
                                                             project_name='GraphRNN',
                                                             backend='Quartus',
                                                             hls_config=config)
    print('model is converted')

#ValueError: Unsupported ONNX opset version: 13
def my_pytorch_to_keras_pt2keras():
    rnn, output = get_pt_model()

    input_shape_rnn = (args.batch_size, 1, args.max_prev_node)
    input_shape_output = (args.batch_size, 1, args.hidden_size_rnn_output)
    print(rnn)
    print(output)
    rnn.eval()
    output.eval()
    dummy_input_rnn = torch.randn(input_shape_rnn)
    dummy_input_output = torch.randn(input_shape_output)

    # Create pt2keras object
    converter = Pt2Keras()

    converter.set_logging_level(logging.DEBUG)

    # convert model
    keras_model: tf.keras.Model = converter.convert(rnn, input_shape_rnn)

    # Save the model
    keras_model.save('./hls4ml-output/output_model.h5')

#RuntimeError: Cannot insert a Tensor that requires grad as a constant. Consider making it a parameter or input, or detaching the gradient
def my_pytorch_to_keras_pytorch2keras():
    rnn, output = get_pt_model()

    input_shape_rnn = (args.batch_size, 1, args.max_prev_node)
    input_shape_output = (args.batch_size, 1, args.hidden_size_rnn_output)
    print(rnn)
    print(output)
    rnn.eval()
    output.eval()
    dummy_input_rnn = torch.randn(input_shape_rnn)
    dummy_input_output = torch.randn(input_shape_output)

    # Convert the model
    keras_model = pytorch_to_keras(rnn, dummy_input_rnn, input_shape_rnn, verbose=True)

    # Save the model
    keras_model.save('./hls4ml-output/output_model.h5')

#Process finished with exit code -1073741819 (0xC0000005)
def converter_test():
    # with open('onnx-config.yml', 'r') as config_file:
    #     config = yaml.safe_load(config_file)
    onnx_model_path = './rnn_model.onnx'
    onnx_model = onnx.load(onnx_model_path)
    config = hls4ml.utils.config_from_onnx_model(onnx_model)

    print(config)

    # hls_model = hls4ml.converters.convert_from_onnx_model(model=onnx_model, config=config)

    hls_model = hls4ml.converters.convert_from_onnx_model(onnx_model,
                                                          hls_config=config,
                                                          output_dir='C:/Users/matth/PycharmProjects/pytorch-onnx-tensorflow-hls4ml/',
                                                          part='xcvc1902-vsvd1760-2MP-e-S',
                                                          backend='Vivado',
                                                          clock_period=500)


if __name__ == '__main__':
    print("check0")

    args = Args()

    converter_test()

    print("finished")
