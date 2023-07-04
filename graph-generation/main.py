from train import *

if __name__ == '__main__':
    # All necessary arguments are defined in args.py
    print("networkx " + nx.__version__)
    args = Args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    print('CUDA', args.cuda)
    print('File name prefix',args.fname)
    # check if necessary directories exist
    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)
    if not os.path.isdir(args.graph_save_path):
        os.makedirs(args.graph_save_path)
    if not os.path.isdir(args.figure_save_path):
        os.makedirs(args.figure_save_path)
    if not os.path.isdir(args.timing_save_path):
        os.makedirs(args.timing_save_path)
    if not os.path.isdir(args.figure_prediction_save_path):
        os.makedirs(args.figure_prediction_save_path)
    if not os.path.isdir(args.nll_save_path):
        os.makedirs(args.nll_save_path)

    time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    # logging.basicConfig(filename='logs/train' + time + '.log', level=logging.DEBUG)
    if args.clean_tensorboard:
        if os.path.isdir("tensorboard"):
            shutil.rmtree("tensorboard")
    configure("tensorboard/run"+time, flush_secs=5)

    graphs = create_graphs.create(args)
    # split datasets
    #random.seed(123)
    #shuffle(graphs)
    #graphs_len = len(graphs)
    #graphs_test = graphs[int(0.8 * graphs_len):]
    #graphs_train = graphs[0:int(0.8*graphs_len)]
    #graphs_validate = graphs[0:int(0.2*graphs_len)]

    ## if use pre-saved graphs
    dir_input = "./input/"
    fname_graphs = dir_input + 'test090.dat'
    graphs = load_graph_list(fname_graphs, is_real=True)
    #graphs = read_sparse_txt('./dataset/PROTEINS_full/PROTEINS_full_A.txt')
    random.seed(123)
    shuffle(graphs)
    graphs_len = len(graphs)
    graphs_test = graphs[int(0.8 * graphs_len):]
    graphs_train = graphs[0:int(0.8 * graphs_len)]
    graphs_validate = graphs[int(0.2 * graphs_len):int(0.4 * graphs_len)]
    args.max_prev_node = 100

    graph_validate_len = 0
    for graph in graphs_validate:
        graph_validate_len += graph.number_of_nodes()
    graph_validate_len /= len(graphs_validate)
    print('graph_validate_len', graph_validate_len)

    graph_test_len = 0
    for graph in graphs_test:
        graph_test_len += graph.number_of_nodes()
    graph_test_len /= len(graphs_test)
    print('graph_test_len', graph_test_len)

    args.max_num_node = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
    max_num_edge = max([graphs[i].number_of_edges() for i in range(len(graphs))])
    min_num_edge = min([graphs[i].number_of_edges() for i in range(len(graphs))])

    # args.max_num_node = 2000
    # show graphs statistics
    print('total graph num: {}, training set: {}'.format(len(graphs),len(graphs_train)))
    print('max number node: {}'.format(args.max_num_node))
    print('max/min number edge: {}; {}'.format(max_num_edge,min_num_edge))
    print('max previous node: {}'.format(args.max_prev_node))

    # save ground truth graphs
    ## To get train and test set, after loading you need to manually slice
    save_graph_list(graphs, args.graph_save_path + args.fname_train + '0.dat')
    save_graph_list(graphs, args.graph_save_path + args.fname_test + '0.dat')
    print('train and test graphs saved at: ', args.graph_save_path + args.fname_test + '0.dat')

    ### comment when normal training, for graph completion only
    # p = 0.5
    # for graph in graphs_train:
    #     for node in list(graph.nodes()):
    #         # print('node',node)
    #         if np.random.rand()>p:
    #             graph.remove_node(node)
        # for edge in list(graph.edges()):
        #     # print('edge',edge)
        #     if np.random.rand()>p:
        #         graph.remove_edge(edge[0],edge[1])


    args.max_prev_node = 39
    args.max_num_node = 39
    ### dataset initialization
    if 'nobfs' in args.note:
        print('nobfs')
        dataset = Graph_sequence_sampler_pytorch_nobfs(graphs_train, max_num_node=args.max_num_node)
        args.max_prev_node = args.max_num_node-1
    if 'barabasi_noise' in args.graph_type:
        print('barabasi_noise')
        dataset = Graph_sequence_sampler_pytorch_canonical(graphs_train,max_prev_node=args.max_prev_node)
        args.max_prev_node = args.max_num_node - 1
    else:
        dataset = Graph_sequence_sampler_pytorch(graphs_train,max_prev_node=args.max_prev_node,max_num_node=args.max_num_node)
    sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(dataset) for i in range(len(dataset))],
                                                                     num_samples=args.batch_size*args.batch_ratio, replacement=True)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                               sampler=sample_strategy)

    print("args.max_num_node: " + str(args.max_num_node) + "\targs.max_prev_node: " + str(args.max_prev_node))
    ### model initialization
    ## Graph RNN VAE model
    # lstm = LSTM_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_lstm,
    #                   hidden_size=args.hidden_size, num_layers=args.num_layers).cuda()
    if 'GraphRNN_VAE_conditional' in args.note:
        rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                        hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
                        has_output=False).cuda()
        output = MLP_VAE_conditional_plain(h_size=args.hidden_size_rnn, embedding_size=args.embedding_size_output, y_size=args.max_prev_node).cuda()
    elif 'GraphRNN_MLP' in args.note:
        rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                        hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
                        has_output=False).cuda()
        output = MLP_plain(h_size=args.hidden_size_rnn, embedding_size=args.embedding_size_output, y_size=args.max_prev_node).cuda()
    elif 'GraphRNN_RNN' in args.note:
        rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                        hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
                        has_output=True, output_size=args.hidden_size_rnn_output).cuda()
        output = GRU_plain(input_size=1, embedding_size=args.embedding_size_rnn_output,
                           hidden_size=args.hidden_size_rnn_output, num_layers=args.num_layers, has_input=True,
                           has_output=True, output_size=1).cuda()


    #Graph_list_print = load_graph_list(r"C:\Users\matth\PycharmProjects\graph-generation\graphs\GraphRNN_RNN_grid_4_128_pred_400_1.dat")
    #print("printing...")
    #draw_graph(Graph_list_print[0], prefix="bababooey")

    ### start training
    #print("start training")
    #train(args, dataset_loader, rnn, output)

    ### graph completion
    # train_graph_completion(args,dataset_loader,rnn,output)

    ### nll evaluation
    #print("start nll eval")
    #train_nll(args, dataset_loader, dataset_loader, rnn, output, max_iter = 200, graph_validate_len=graph_validate_len,graph_test_len=graph_test_len)


    rnn.train()
    output.train()

    summary(rnn, (args.batch_size, 1, args.max_prev_node))
    summary(output, (args.batch_size, 1, args.hidden_size_rnn_output))

    for batch_idx, data in enumerate(dataset_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))
        # output.hidden = output.init_hidden(batch_size=x_unsorted.size(0)*x_unsorted.size(1))

        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)
        print('x shape: ' + str(x.data.shape))
        print('y shape: ' + str(y.data.shape))

        x = Variable(x).cuda()
        y = Variable(y).cuda()
        h = rnn(x, pack=True, input_len=y_len)
        print('h shape: ' + str(h.data.shape))