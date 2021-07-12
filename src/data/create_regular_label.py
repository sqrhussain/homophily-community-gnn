import networkx as nx
from networkx.utils import powerlaw_sequence
import os
from src.data.create_stochastic_block_model import load_labels
import random
from torch_geometric.utils import homophily_ratio, from_networkx

def get_random_graph(n,m,seed=0):
    d = max((2*m)//n,1)

    if (n*d)%2!=0:
        d+=1

    print(f'average degree = {d}')

    return nx.random_regular_graph(d,n,seed=seed)

def generate_dregular(G,seed=0):
    GNoisy = get_random_graph(G.number_of_nodes(), G.number_of_edges(), seed)
    keys = [x[0] for x in G.in_degree()]
    G_mapping = dict(zip(range(len(G.nodes())),keys))
    G_rev_mapping = dict(zip(keys,range(len(G.nodes()))))
    GNoisy = nx.relabel_nodes(GNoisy,G_mapping)
    return GNoisy


def generate_modified_conf_model(G, seed=0):
    node_labels_dict = nx.get_node_attributes(G,'label')
    # print(type(node_labels_dict))
    # print(list(node_labels_dict.values()))
    unique_node_labels = set(node_labels_dict.values())
    same_label_subgraphs = {}
    for node_label in unique_node_labels:
        same_label_subgraphs[node_label] = nx.DiGraph()
    edges_to_remove = []
    for edge in G.edges:
        if node_labels_dict[edge[0]] == node_labels_dict[edge[1]]:
            node_label = G.nodes(data=True)[edge[0]]['label']
            same_label_subgraphs[node_label].add_edge(edge[0], edge[1])
            edges_to_remove.append((edge[0], edge[1]))
    G.remove_edges_from(edges_to_remove)
    for label in same_label_subgraphs:
        G.add_edges_from(generate_dregular(same_label_subgraphs[label], seed).edges)
    return G


def generate_regular_label(G,seed=0):
    node_labels_dict = nx.get_node_attributes(G,'label')
    
    unique_node_labels = list(set(node_labels_dict.values()))
    same_label_subgraphs = {}
    intended_homophily = homophily_ratio
    total_edges = 0
    labels_to_nodes = {label:[node for node in node_labels_dict if node_labels_dict[node] == label] for label in unique_node_labels}
    
    GNoisy = nx.DiGraph()
    
    for node_label in unique_node_labels:
        nodes = labels_to_nodes[node_label]
        same_label_subgraphs[node_label] = generate_dregular(G.subgraph(nodes),seed)
        GNoisy.add_edges_from(same_label_subgraphs[node_label].edges)
        total_edges += len(same_label_subgraphs[node_label].edges())
    print(f'intra_edges = {total_edges}')
    remaining_edges = int((total_edges *(1-intended_homophily))/intended_homophily)
    print(f'inter_edges = {remaining_edges}')
    inter_edges = []
    for i in range(remaining_edges):
        l1 = random.choice(unique_node_labels)
        u1 = random.choice(labels_to_nodes[l1])
        l2 = random.choice(unique_node_labels)
        while l2 == l1:
            l2 = random.choice(unique_node_labels)
        u2 = random.choice(labels_to_nodes[l2])
        inter_edges.append([u1,u2])
    GNoisy.add_edges_from(inter_edges)

    return GNoisy


def generate_multiple_regular_label_graphs(graph_path, content_path, output_prefix,output_suffix = '.cites',inits=10):
    fin = graph_path
    G=nx.read_edgelist(fin,create_using=nx.DiGraph())
    node_labels_dict,_ = load_labels(content_path)
    nx.set_node_attributes(G,node_labels_dict, name='label')
    for i in range(inits):
        print(f'{graph_path} ... {output_prefix}_{i}{output_suffix}')
        GNoisy = generate_regular_label(G,seed=i)
        fout = f'{output_prefix}_{i}{output_suffix}'
        nx.write_edgelist(GNoisy,fout)



if __name__ == '__main__':
    datasets = 'cora citeseer twitter chameleon squirrel webkb actor pubmed cora_full'.split()
    for dataset in datasets:

        if not os.path.exists(f'data/graphs/reglabel/{dataset}/'):
            os.mkdir(f'data/graphs/reglabel/{dataset}/')
        generate_multiple_regular_label_graphs(f'data/graphs/processed/{dataset}/{dataset}.cites',
                                f'data/graphs/processed/{dataset}/{dataset}.content',
                                f'data/graphs/reglabel/{dataset}/{dataset}_reglabel',inits=10)
