import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
import pickle

# This file converts .csv gathered from yahoo finance to financial correlation graphs per month
# The .csv files should be places in the same folder ( ./ )
# company_list_to_graph_list takes a list with the names of the .csv files and converts them to graphs
# output = company_list_to_graph_list(csv_name_list)
# output[i] -> selects the type of stock data in order of occurance in .csv file
# output[i][0] -> stock data type name as string e.g. 'high', 'low', 'volume'
# output[i][1][j] -> list of stock data type per month
# output[i][1][j][0] -> name of the month as a string in format YYYY-MM
# output[i][1][j][1] -> graph with stock data type [i] and in month [j]

def get_name_list(input_dir):
    file_list = listdir(input_dir)
    name_list = []
    for i in range(len(file_list)):
        if file_list[i][-4:] == '.csv':
            name_list.append(file_list[i][:-4])
    return name_list

def get_df_name(df):
    name = [x for x in globals() if globals()[x] is df][0]
    return name

def get_csv_list(name_list, input_dir='./'):
    csv_list = []
    for i in range(len(name_list)):
        csv_list.append((name_list[i], pd.read_csv(input_dir + name_list[i] + '.csv')))
    return csv_list

def add_column_with_date(df_s, df_d, column_s, column_d):
    df_t = df_s[['Date', column_s]]
    df_d = df_d.merge(df_t, on='Date', how='outer')
    df_d = df_d.rename(columns={column_s: column_d})
    return df_d

def get_type_df(company_list, type_name):
    type_df = pd.DataFrame(columns=['Date'])
    for i in range(len(company_list)):
        type_df = add_column_with_date(company_list[i][1], type_df, type_name, company_list[i][0])
    type_df = type_df.dropna()
    # type_df = dataFrame
    return type_df

def get_type_df_grp(company_list, type_name):
    type_df = get_type_df(company_list, type_name)
    type_df['Date'] = pd.to_datetime(type_df['Date'])
    g = type_df.groupby(pd.Grouper(key='Date', freq='M'))
    type_list_grp = [group for _, group in g]
    for i in range(len(type_list_grp)):
        type_list_grp[i] = type_list_grp[i].reset_index()
        type_list_grp[i] = type_list_grp[i].drop(columns=['index'])
    # graph_list_grp[month index]=dataFrame
    return type_list_grp

def get_type_list(company_list):
    type_name_list = company_list[0][1].columns.values.tolist()
    type_name_list.remove('Date')
    type_list = []
    for i in range(len(type_name_list)):
        type_list.append((type_name_list[i], get_type_df_grp(company_list, type_name_list[i])))
    # type_list[type index] = (type name, graph_list_grp)
    return type_list

def get_corr_df(type_df):
    type_df = type_df.set_index('Date')
    corr_mx = type_df.corr()
    # corr_mx = dataFrame
    return corr_mx

def get_corr_df_grp(type_df_grp):
    corr_mx_grp = []
    for i in range(len(type_df_grp)):
        corr_mx_grp.append(((type_df_grp[i]['Date'][0]).strftime('%Y-%m-%d')[:-3], get_corr_df(type_df_grp[i])))
    # corr_mx_grp[date index] = (date name, corr_mx)
    return corr_mx_grp

def get_corr_list(type_list):
    corr_list = []
    for i in range(len(type_list)):
        corr_list.append((type_list[i][0], get_corr_df_grp(type_list[i][1])))
    # corr_list[type index] = (type name, corr_mx_grp)
    return corr_list

def get_thres_mx(corr_mx, tau):
    corr_mx = corr_mx.to_numpy()
    i_max = len(corr_mx)
    j_max = len(corr_mx[0])
    thres_mx = np.zeros((i_max, j_max), dtype=int)
    for i in range(i_max):
        for j in range(j_max):
            if i == j:
                continue
            elif corr_mx[i][j] > tau:
                thres_mx[i][j] = 1
            # already 0 if not above threshold
    return thres_mx

def get_thres_mx_gr(corr_mx, tau):
    thres_mx_gr = []
    for i in range(len(corr_mx)):
        thres_mx_gr.append((corr_mx[i][0], get_thres_mx(corr_mx[i][1], tau)))
    # thres_mx_gr[month index] = (month name, thres_mx)
    return thres_mx_gr

def get_thres_list(corr_list, tau):
    thres_list = []
    for i in range(len(corr_list)):
        thres_list.append((corr_list[i][0], get_thres_mx_gr(corr_list[i][1], tau)))
    # thres_list[type index] = (type name, thres_mx_gr)
    return thres_list

def get_graph_list(corr_list, tau):
    # create mapping to map the node index to company name when creating the graphs
    mapping = {}
    company_name_list = corr_list[0][1][0][1].columns.values.tolist()
    for j in range(len(company_name_list)):
        mapping.update({j: company_name_list[j]})

    # threshold the correlation matrix
    thres_list = get_thres_list(corr_list, tau)

    # create the graphs
    graph_list = []
    for i in range(len(thres_list)):
        graph_gr = []
        for j in range(len(thres_list[i][1])):
            G = nx.from_numpy_matrix(thres_list[i][1][j][1], parallel_edges=False)
            G = nx.relabel_nodes(G, mapping)
            graph_gr.append((thres_list[i][1][j][0], G))
        graph_list.append((thres_list[i][0], graph_gr))
    return graph_list

def csv_list_to_graph_list(input_dir, tau):
    name_list = get_name_list(input_dir)
    company_list = get_csv_list(name_list, input_dir)
    type_list = get_type_list(company_list)
    corr_list = get_corr_list(type_list)
    graph_list = get_graph_list(corr_list, tau)
    return graph_list

def get_raw_graph_list(graph_list):
    raw_graph_list = []
    for i in range(len(graph_list)):
        for j in range(len(graph_list[i][1])):
            raw_graph_list.append(graph_list[i][1][j][1])
    return raw_graph_list

def get_pickle_file_from_raw_graph_list(raw_graph_list, output_dir, fname):
    fname = str(output_dir) + str(fname) + '.dat'
    with open(fname, "wb") as f:
        pickle.dump(raw_graph_list, f)

def get_pickle_file(graph_list, output_dir, fname):
    raw_graph_list = get_raw_graph_list(graph_list)
    get_pickle_file_from_raw_graph_list(raw_graph_list, output_dir, fname)

if __name__ == '__main__':
    my_graph_list = csv_list_to_graph_list('./stonks/', 0.8)
    get_pickle_file(my_graph_list, './output/', 'test080')
