from src.evaluation.gnn_evaluation_module import eval_gnn
from src.models.gat_models import MonoGAT#, BiGAT, TriGAT
from src.models.rgcn_models import MonoRGCN, RGCN2
from src.models.appnp_model import MonoAPPNPModel
from src.models.multi_layered_model import MonoModel#, BiModel, TriModel
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, RGCNConv, SGConv, APPNP, ClusterGCNConv
from src.data.data_loader import GraphDataset
import warnings
import pandas as pd
import os
import argparse
import numpy as np
import pickle
import torch
from src.evaluation.network_split import NetworkSplitShchur
from src.data.create_modified_configuration_model import generate_modified_conf_model
from torch_geometric.utils import from_networkx, to_networkx
from community import best_partition
import networkx as nx

def parse_args():

    parser = argparse.ArgumentParser(description="Test accuracy for GCN/SAGE/GAT/RGCN/SGC/APPNP")
    parser.add_argument('--size',
                        type=int,
                        default=96,
                        help='Channel size. Default is 12.')
    parser.add_argument('--lr',
                        type=float,
                        default=0.01,
                        help='Learning rate. Default is 0.01.')
    parser.add_argument('--wd',
                        type=float,
                        default=0.01,
                        help='Regularization weight. Default is 0.01.')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.8,
                        help='Dropout probability. Default is 0.6.')
    parser.add_argument('--conf',
                        type=bool,
                        default=False,
                        help='Is configuration model evaluation. Default is False.')
    parser.add_argument('--shifting',
                        type=bool,
                        default=False,
                        help='Is shifting evaluation. Default is False.')
    parser.add_argument('--sbm',
                        type=bool,
                        default=False,
                        help='Is SBM evaluation. Default is False.')
    parser.add_argument('--sbm_label',
                        type=bool,
                        default=False,
                        help='Is SBM_label evaluation. Default is False.')
    parser.add_argument('--flipped',
                        type=bool,
                        default=False,
                        help='Evaluating with flipped edges? Default is False.')
    parser.add_argument('--removed_hubs',
                        type=bool,
                        default=False,
                        help='Evaluating with removed hubs? Default is False.')
    parser.add_argument('--added_2hop_edges',
                        type=bool,
                        default=False,
                        help='Evaluating with added 2-hop edges? Default is False.')
    parser.add_argument('--label_sbm',
                        type=bool,
                        default=False,
                        help='Evaluating with SBMs created from labels? Default is False.')

    parser.add_argument('--heads',
                        type=int,
                        default=4,
                        help='Attention heads. Default is 4.')
    parser.add_argument('--attention_dropout',
                        type=float,
                        default=0.4,
                        help='Attention dropout for GAT. Default is 0.4.')
    parser.add_argument('--dataset',
                        default="cora",
                        help='Dataset name. Default is cora.')
    parser.add_argument('--model',
                        default="gcn",
                        help='Model name. Default is GCN.')
    parser.add_argument('--splits',
                        type=int,
                        default=100,
                        help='Number of random train/validation/test splits. Default is 100.')
    parser.add_argument('--runs',
                        type=int,
                        default=20,
                        help='Number of random initializations of the model. Default is 20.')
    parser.add_argument('--conf_inits',
                        type=int,
                        default=10,
                        help='Number of configuration model runs. Default is 10.')
    parser.add_argument('--sbm_inits',
                        type=int,
                        default=10,
                        help='Number of SBM runs. Default is 10.')
    parser.add_argument('--directionality',
                        default='undirected',
                        help='Directionality: undirected/directed/reversed. Default is undirected.')
                        
    parser.add_argument('--train_examples',
                        type=int,
                        default=20,
                        help='Number of training examples per class. Default is 20.')
    parser.add_argument('--val_examples',
                        type=int,
                        default=30,
                        help='Number of validation examples per class. Default is 30.')
                        
    args = parser.parse_args()
    return args

name2conv = {'gcn': GCNConv, 'sage': SAGEConv, 'gat': GATConv, 'rgcn': RGCNConv, 'rgcn2':RGCN2, 'sgc':SGConv, 'appnp':APPNP, 'cgcn':ClusterGCNConv}

def eval_archs_gat(dataset, dataset_name, channel_size, dropout, lr, wd, heads,attention_dropout,runs,splits,train_examples,val_examples, models=[MonoGAT],isDirected = False):
    if isDirected:
        models = [MonoGAT]
    return eval_gnn(dataset, dataset_name, GATConv, channel_size, dropout, lr, wd, heads=heads, attention_dropout=attention_dropout,
                      models=models, num_runs=runs, num_splits=splits, test_score=True,
                      train_examples = train_examples, val_examples = val_examples)


def eval_archs_gcn(dataset, dataset_name, conv, channel_size, dropout, lr, wd, runs,splits,train_examples,val_examples, models=[MonoModel], isDirected=False):
    if isDirected:
        models = [MonoModel]
    return eval_gnn(dataset, dataset_name, conv, channel_size, dropout, lr, wd, heads=1,attention_dropout=0.3, # dummy values for heads and attention_dropout
                      models=models, num_runs=runs, num_splits=splits,test_score=True,
                      train_examples = train_examples, val_examples = val_examples)


def eval_archs_appnp(dataset, dataset_name, conv, channel_size, dropout, lr, wd, runs,splits,train_examples,val_examples, models=[MonoAPPNPModel]):
    return eval_gnn(dataset, dataset_name, conv, channel_size, dropout, lr, wd, heads=1,attention_dropout=0.3, # dummy values for heads and attention_dropout
                      models=models, num_runs=runs, num_splits=splits,test_score=True,
                      train_examples = train_examples, val_examples = val_examples)

def eval_archs_rgcn(dataset, dataset_name, conv, channel_size, dropout, lr, wd, runs,splits,train_examples,val_examples, models=[MonoRGCN]):
    return eval_gnn(dataset, dataset_name, conv, channel_size, dropout, lr, wd, heads=1,attention_dropout=0.3,  # dummy values for heads and attention_dropout
                      models=models, num_runs=runs, num_splits=splits,test_score=True,
                      train_examples = train_examples, val_examples = val_examples)



def eval(model, dataset, dataset_name, channel_size, dropout, lr, wd, heads, attention_dropout, runs, splits, train_examples, val_examples, isDirected):
    if model == 'gat':
        return eval_archs_gat(dataset, dataset_name, channel_size, dropout, lr, wd, heads, attention_dropout, splits=splits, runs=runs, train_examples = train_examples, val_examples = val_examples,isDirected=isDirected)
    elif model == 'rgcn' or model == 'rgcn2':
        return eval_archs_rgcn(dataset, dataset_name, name2conv[model], channel_size, dropout, lr, wd, splits=splits, runs=runs, train_examples = train_examples, val_examples = val_examples)
    elif model == 'appnp':
        return eval_archs_appnp(dataset, dataset_name, name2conv[model], channel_size, dropout, lr, wd, splits=splits, runs=runs, train_examples = train_examples, val_examples = val_examples)
    else:
        return eval_archs_gcn(dataset, dataset_name, name2conv[model], channel_size, dropout, lr, wd, splits=splits, runs=runs, train_examples = train_examples, val_examples = val_examples,isDirected=isDirected)

def eval_original(model, dataset_name, directionality, size, dropout, lr, wd, heads,attention_dropout,
        splits, runs, train_examples, val_examples):
    isDirected = (directionality != 'undirected')
    isReversed = (directionality == 'reversed')
    dataset = GraphDataset(f'data/tmp/{dataset_name}{("_" + directionality) if isDirected else ""}', dataset_name,
                           f'data/graphs/processed/{dataset_name}/{dataset_name}.cites',
                           f'data/graphs/processed/{dataset_name}/{dataset_name}.content',
                           directed=isDirected, reverse=isReversed)[0]
    df_cur = eval(model=model, dataset=dataset, dataset_name=dataset_name, channel_size=size, lr=lr, splits=splits, runs=runs,
                  dropout=dropout, wd=wd, heads=heads, attention_dropout=attention_dropout,
                  train_examples = train_examples, val_examples = val_examples,isDirected=isDirected)
    return df_cur

def eval_shuffled_features(model, dataset_name, directionality, size, dropout, lr, wd, heads,attention_dropout,
        splits, runs, train_examples, val_examples):
    isDirected = (directionality != 'undirected')
    isReversed = (directionality == 'reversed')
    dataset = GraphDataset(f'data/tmp/{dataset_name}{("_" + directionality) if isDirected else ""}', dataset_name,
                           f'data/graphs/processed/{dataset_name}/{dataset_name}.cites',
                           f'data/graphs/processed/{dataset_name}/{dataset_name}.content',
                           directed=isDirected, reverse=isReversed)[0]
    dataset.x = dataset.x[torch.randperm(dataset.x.size()[0])]
    df_cur = eval(model=model, dataset=dataset, dataset_name=dataset_name, channel_size=size, lr=lr, splits=splits, runs=runs,
                  dropout=dropout, wd=wd, heads=heads, attention_dropout=attention_dropout,
                  train_examples = train_examples, val_examples = val_examples,isDirected=isDirected)
    return df_cur

def eval_random_features(model, dataset_name, directionality, size, dropout, lr, wd, heads,attention_dropout,
        splits, runs, train_examples, val_examples):
    isDirected = (directionality != 'undirected')
    isReversed = (directionality == 'reversed')
    dataset = GraphDataset(f'data/tmp/{dataset_name}{("_" + directionality) if isDirected else ""}', dataset_name,
                           f'data/graphs/processed/{dataset_name}/{dataset_name}.cites',
                           f'data/graphs/processed/{dataset_name}/{dataset_name}.content',
                           directed=isDirected, reverse=isReversed)[0]
    dataset.x = torch.randint(0, 2, dataset.x.shape, dtype=torch.float)
    df_cur = eval(model=model, dataset=dataset, dataset_name=dataset_name, channel_size=size, lr=lr, splits=splits, runs=runs,
                  dropout=dropout, wd=wd, heads=heads, attention_dropout=attention_dropout,
                  train_examples = train_examples, val_examples = val_examples,isDirected=isDirected)
    return df_cur

def eval_cm_communities(model, dataset_name, directionality, size, dropout, lr, wd, heads,attention_dropout,
        splits, runs, train_examples, val_examples, inits):
    isDirected = (directionality != 'undirected')
    isReversed = (directionality == 'reversed')
    df_val = pd.DataFrame()
    for i in range(inits):
      dataset = GraphDataset(f'data/tmp/{dataset_name}-cm_communities-{i}', dataset_name,
                             f'data/graphs/cm_communities/{dataset_name}/{dataset_name}_cm_communities_{i}.cites',
                             f'data/graphs/processed/{dataset_name}/{dataset_name}.content',
                             directed=isDirected, reverse=isReversed)[0]

      # G = to_networkx(dataset)
      # G = nx.DiGraph(G)
      # node_communities = best_partition(nx.to_undirected(G))
      # nx.set_node_attributes(G,node_communities,'label')
      # # print(dataset.edge_index)
      # old_edges = dataset.edge_index
      # G = generate_modified_conf_model(G)
      # # dir_path = f'data/graphs/cm_communities/{dataset_name}'
      # # if not os.path.exists(dir_path):
      # #   os.mkdir(dir_path)
      # # nx.write_edgelist(G, f'{dir_path}/{dataset_name}_cm_communities_{i}.cites')
      # dataset.edge_index = torch.tensor(data=np.array(list(G.edges)).T,dtype=torch.long)
      # print((torch.tensor(data=np.array(list(G.edges)).T,dtype=torch.long)-old_edges).abs().sum())
      # print(dataset.edge_index)
      df_cur = eval(model=model, dataset=dataset, dataset_name=dataset_name, channel_size=size, lr=lr, splits=splits, runs=runs,
                    dropout=dropout, wd=wd, heads=heads, attention_dropout=attention_dropout,
                    train_examples = train_examples, val_examples = val_examples,isDirected=isDirected)
      df_cur['graph'] = i
      df_val = pd.concat([df_val, df_cur])
    
    return df_val


def eval_random(model, dataset_name, directionality, size, dropout, lr, wd, heads,attention_dropout,
        splits, runs, train_examples, val_examples, random_inits):
    isDirected = (directionality != 'undirected')
    isReversed = (directionality == 'reversed')
    df_val = pd.DataFrame()
    for i in range(random_inits):
        dataset = GraphDataset(f'data/tmp/{dataset_name}{("_" + directionality) if isDirected else ""}-random{i}', dataset_name,
                             f'data/graphs/random/{dataset_name}/{dataset_name}_{i}.cites',
                             f'data/graphs/processed/{dataset_name}/{dataset_name}.content',
                             directed=isDirected, reverse=isReversed)[0]
        df_cur = eval(model=model, dataset=dataset, dataset_name=dataset_name, channel_size=size, lr=lr, splits=splits, runs=runs,
                      dropout=dropout, wd=wd, heads=heads,attention_dropout=attention_dropout,
                      train_examples = train_examples, val_examples = val_examples,isDirected=isDirected)
        df_cur['random_num'] = i
        df_val = pd.concat([df_val, df_cur])
    return df_val

def eval_erdos(model, dataset_name, directionality, size, dropout, lr, wd, heads,attention_dropout,
        splits, runs, train_examples, val_examples, erdos_inits):
    isDirected = (directionality != 'undirected')
    isReversed = (directionality == 'reversed')
    df_val = pd.DataFrame()
    for i in range(erdos_inits):
        dataset = GraphDataset(f'data/tmp/{dataset_name}{("_" + directionality) if isDirected else ""}-erdos{i}', dataset_name,
                             f'data/graphs/erdos/{dataset_name}/{dataset_name}_{i}.cites',
                             f'data/graphs/processed/{dataset_name}/{dataset_name}.content',
                             directed=isDirected, reverse=isReversed)[0]
        df_cur = eval(model=model, dataset=dataset, dataset_name=dataset_name, channel_size=size, lr=lr, splits=splits, runs=runs,
                      dropout=dropout, wd=wd, heads=heads,attention_dropout=attention_dropout,
                      train_examples = train_examples, val_examples = val_examples,isDirected=isDirected)
        df_cur['erdos_num'] = i
        df_val = pd.concat([df_val, df_cur])
    return df_val


def eval_injected_edges(model, dataset_name, directionality, size, dropout, lr, wd, heads,attention_dropout,
        splits, runs, train_examples, val_examples, inits, num_edges, hubs_experiment):
    isDirected = (directionality != 'undirected')
    isReversed = (directionality == 'reversed')
    df_val = pd.DataFrame()
    last_edge = None
    for e in num_edges:
      for i in range(inits):
          dataset = GraphDataset(f'data/tmp/{dataset_name}{("_" + directionality) if isDirected else ""}-injected_{e}_{i}_{hubs_experiment}', dataset_name,
                             f'data/graphs/injected_edges/{dataset_name}/{dataset_name}_{hubs_experiment}_{e}_{i}.cites',
                             f'data/graphs/processed/{dataset_name}/{dataset_name}.content',
                             directed=isDirected, reverse=isReversed)[0]
          # print(f'data/graphs/injected_edges/{dataset_name}/{dataset_name}_{hubs_experiment}_{e}_{i}.cites')
          # print(dataset.edge_index.shape)
          # print(dataset.edge_index)
          # if last_edge is None:
          #   last_edge = dataset.edge_index
          #   continue
          # print((1-last_edge.eq(last_edge).double()).sum())
          # continue
          df_cur = eval(model=model, dataset=dataset, dataset_name=dataset_name, channel_size=size, lr=lr, splits=splits, runs=runs,
                        dropout=dropout, wd=wd, heads=heads,attention_dropout=attention_dropout,
                        train_examples = train_examples, val_examples = val_examples,isDirected=isDirected)
          df_cur['init_num'] = i
          df_cur['injected_edges'] = e
          df_val = pd.concat([df_val, df_cur])
    return df_val


def eval_injected_edges_degree_cat(model, dataset_name, directionality, size, dropout, lr, wd, heads,attention_dropout,
        splits, runs, train_examples, val_examples, inits, num_edges, percentile):
    isDirected = (directionality != 'undirected')
    isReversed = (directionality == 'reversed')
    df_val = pd.DataFrame()
    last_edge = None
    e = num_edges
    hubs_experiment = 'global_edges'
    for i in range(inits):
      for frm in range(0,100,percentile):
        to = frm + percentile
        dataset = GraphDataset(f'data/tmp/{dataset_name}{("_" + directionality) if isDirected else ""}-injected_{e}_{i}_{hubs_experiment}_{frm}_to_{to}', dataset_name,
                           f'data/graphs/injected_edges_degree_cat/{dataset_name}/{dataset_name}_{hubs_experiment}_{e}_{i}_{frm}_to_{to}.cites',
                           f'data/graphs/processed/{dataset_name}/{dataset_name}.content',
                           directed=isDirected, reverse=isReversed)[0]
        df_cur = eval(model=model, dataset=dataset, dataset_name=dataset_name, channel_size=size, lr=lr, splits=splits, runs=runs,
                      dropout=dropout, wd=wd, heads=heads,attention_dropout=attention_dropout,
                      train_examples = train_examples, val_examples = val_examples,isDirected=isDirected)
        df_cur['init_num'] = i
        df_cur['injected_edges'] = e
        df_cur['from'] = frm
        df_cur['to'] = to

        df_val = pd.concat([df_val, df_cur])
    return df_val


def eval_injected_edges_constant_nodes(model, dataset_name, directionality, size, dropout, lr, wd, heads,attention_dropout,
        splits, runs, train_examples, val_examples, inits, control_ratio, edges_per_node, percentile):
    isDirected = (directionality != 'undirected')
    isReversed = (directionality == 'reversed')
    df_val = pd.DataFrame()
    last_edge = None
    hubs_experiment = 'global_edges'
    for frm in range(0,100,percentile):
      for i in range(inits):
        for e in edges_per_node:
          to = frm + percentile
          dataset = GraphDataset(f'data/tmp/{dataset_name}{("_" + directionality) if isDirected else ""}-injected_{e}edges_{control_ratio}nodes_{i}_{hubs_experiment}_{frm}_to_{to}', dataset_name,
                             f'data/graphs/injected_edges_constant_nodes/{dataset_name}/{dataset_name}_global_edges{e}_nodes{control_ratio:.3f}_{i}_{frm}_to_{to}.cites',
                             f'data/graphs/processed/{dataset_name}/{dataset_name}.content',
                             directed=isDirected, reverse=isReversed)[0]
          df_cur = eval(model=model, dataset=dataset, dataset_name=dataset_name, channel_size=size, lr=lr, splits=splits, runs=runs,
                        dropout=dropout, wd=wd, heads=heads,attention_dropout=attention_dropout,
                        train_examples = train_examples, val_examples = val_examples,isDirected=isDirected)
          df_cur['init_num'] = i
          df_cur['edges_per_node'] = e
          df_cur['control_ratio'] = control_ratio
          df_cur['from'] = frm
          df_cur['to'] = to

          df_val = pd.concat([df_val, df_cur])
    return df_val


def eval_injected_edges_attack_target(model, dataset_name, directionality, size, dropout, lr, wd, heads,attention_dropout,
        splits, runs, train_examples, val_examples, inits, control_ratio, edges_per_node, percentile):
    isDirected = (directionality != 'undirected')
    isReversed = (directionality == 'reversed')
    df_val = pd.DataFrame()
    last_edge = None
    hubs_experiment = 'global_edges'
    for atkfrm in range(0,100,percentile):
      for tgtfrm in range(0,100,percentile):
        for i in range(inits):
          for e in edges_per_node:
            atkto = atkfrm + percentile
            tgtto = tgtfrm + percentile
            dataset = GraphDataset(f'data/tmp/{dataset_name}{("_" + directionality) if isDirected else ""}-injected_{e}edges_{control_ratio:.3f}nodes_{i}_{hubs_experiment}_atk{atkfrm}_{atkto}_tgt{tgtfrm}_{tgtto}', dataset_name,
                               f'data/graphs/injected_edges_attack_target/{dataset_name}/{dataset_name}_global_edges{e}_nodes{control_ratio:.3f}_{i}_atk{atkfrm}_{atkto}_tgt{tgtfrm}_{tgtto}.cites',
                               f'data/graphs/processed/{dataset_name}/{dataset_name}.content',
                               directed=isDirected, reverse=isReversed)[0]
            df_cur = eval(model=model, dataset=dataset, dataset_name=dataset_name, channel_size=size, lr=lr, splits=splits, runs=runs,
                          dropout=dropout, wd=wd, heads=heads,attention_dropout=attention_dropout,
                          train_examples = train_examples, val_examples = val_examples,isDirected=isDirected)
            df_cur['init_num'] = i
            df_cur['edges_per_node'] = e
            df_cur['control_ratio'] = control_ratio
            df_cur['atkfrm'] = atkfrm
            df_cur['atkto'] = atkto
            df_cur['tgtfrm'] = tgtfrm
            df_cur['tgtto'] = tgtto

            df_val = pd.concat([df_val, df_cur])
    return df_val

def eval_injected_edges_sbm(model, dataset_name, directionality, size, dropout, lr, wd, heads,attention_dropout,
        splits, runs, train_examples, val_examples, inits, num_edges, hubs_experiment):
    isDirected = (directionality != 'undirected')
    isReversed = (directionality == 'reversed')
    df_val = pd.DataFrame()
    last_edge = None
    for e in num_edges:
      for i in range(inits):
          dataset = GraphDataset(f'data/tmp/{dataset_name}{("_" + directionality) if isDirected else ""}-injected_sbm_{e}_{i}_{hubs_experiment}', dataset_name,
                             f'data/graphs/injected_edges_sbm/{dataset_name}/{dataset_name}_{hubs_experiment}_{e}_{i}.cites',
                             f'data/graphs/processed/{dataset_name}/{dataset_name}.content',
                             directed=isDirected, reverse=isReversed)[0]
          df_cur = eval(model=model, dataset=dataset, dataset_name=dataset_name, channel_size=size, lr=lr, splits=splits, runs=runs,
                        dropout=dropout, wd=wd, heads=heads,attention_dropout=attention_dropout,
                        train_examples = train_examples, val_examples = val_examples,isDirected=isDirected)
          df_cur['init_num'] = i
          df_cur['injected_edges'] = e
          df_val = pd.concat([df_val, df_cur])
    return df_val


def eval_label_sbm(model, dataset_name, directionality, size, dropout, lr, wd, heads,attention_dropout,
        splits, runs, train_examples, val_examples,hubs_experiment):
    isDirected = (directionality != 'undirected')
    isReversed = (directionality == 'reversed')
    dataset = GraphDataset(f'data/tmp/{dataset_name}{("_" + directionality) if isDirected else ""}-label_sbm_{hubs_experiment}', dataset_name,
                           f'data/graphs/label_sbm/{dataset_name}/{dataset_name}_{hubs_experiment}.cites',
                           f'data/graphs/processed/{dataset_name}/{dataset_name}.content',
                           directed=isDirected, reverse=isReversed)[0]
    df_cur = eval(model=model, dataset=dataset, dataset_name=dataset_name, channel_size=size, lr=lr, splits=splits, runs=runs,
                  dropout=dropout, wd=wd, heads=heads, attention_dropout=attention_dropout,
                  train_examples = train_examples, val_examples = val_examples,isDirected=isDirected)
    return df_cur

def eval_conf(model, dataset_name, directionality, size, dropout, lr, wd, heads,attention_dropout,
        splits, runs, train_examples, val_examples, conf_inits):
    isDirected = (directionality != 'undirected')
    isReversed = (directionality == 'reversed')
    df_val = pd.DataFrame()
    for i in range(conf_inits):
        dataset = GraphDataset(f'data/tmp/{dataset_name}{("_" + directionality) if isDirected else ""}-confmodel{i}', dataset_name,
                               f'data/graphs/confmodel/{dataset_name}/{dataset_name}_confmodel_{i}.cites',
                               f'data/graphs/processed/{dataset_name}/{dataset_name}.content',
                               directed=isDirected, reverse=isReversed)[0]
        df_cur = eval(model=model, dataset=dataset, dataset_name=dataset_name, channel_size=size, lr=lr, splits=splits, runs=runs,
                      dropout=dropout, wd=wd, heads=heads,attention_dropout=attention_dropout,
                      train_examples = train_examples, val_examples = val_examples,isDirected=isDirected)
        df_cur['confmodel_num'] = i
        df_val = pd.concat([df_val, df_cur])
    return df_val

def eval_shifting(model, dataset_name, directionality, size, dropout, lr, wd, heads,attention_dropout,
        splits, runs, train_examples, val_examples, shifting_inits):
    isDirected = (directionality != 'undirected')
    isReversed = (directionality == 'reversed')
    df_val = pd.DataFrame()

    for change in 'CL':
      for inc in [True, False]:
          for r in [0.16,0.32,0.64]: #[0.02,0.04,0.08]:
            for i in range(shifting_inits):
              output_prefix = f'data/graphs/shifting/{dataset_name}/{dataset_name}_shifting'
              output_suffix = '.cites'
              graph_path = f'{output_prefix}_{change}_{"inc" if inc else "dec"}_r{r:.2f}_{i}{output_suffix}'
              if not os.path.exists(graph_path):
                print(f'File not found: {graph_path}')
                continue
              dataset = GraphDataset(f'data/tmp/{dataset_name}_shifting_{change}_{"inc" if inc else "dec"}_r{r:.2f}_{i}{output_suffix}',
                                     dataset_name, graph_path,
                                     f'data/graphs/processed/{dataset_name}/{dataset_name}.content',
                                     directed=isDirected, reverse=isReversed)[0]
              df_cur = eval(model=model, dataset=dataset, dataset_name=dataset_name, channel_size=size, lr=lr, splits=splits, runs=runs,
                            dropout=dropout, wd=wd, heads=heads,attention_dropout=attention_dropout,
                            train_examples = train_examples, val_examples = val_examples,isDirected=isDirected)
              df_cur['graph_num'] = i
              df_cur['inc'] = inc
              df_cur['change'] = change
              df_cur['r'] = r
              df_val = pd.concat([df_val, df_cur])
    return df_val

def eval_sbm(model, dataset_name, directionality, size, dropout, lr, wd, heads,attention_dropout,
        splits, runs, train_examples, val_examples, sbm_inits):
    isDirected = (directionality != 'undirected')
    isReversed = (directionality == 'reversed')
    df_val = pd.DataFrame()
    for i in range(sbm_inits):
        print(f'data/graphs/processed/{dataset_name}/{dataset_name}.content')
        dataset = GraphDataset(f'data/tmp/{dataset_name}{("_" + directionality) if isDirected else ""}-sbm{i}', dataset_name,
                               f'data/graphs/sbm/{dataset_name}/{dataset_name}_sbm_{i}.cites',
                               f'data/graphs/processed/{dataset_name}/{dataset_name}.content',
                               directed=isDirected, reverse=isReversed)[0]
        df_cur = eval(model=model, dataset=dataset, dataset_name=dataset_name, channel_size=size, lr=lr, splits=splits, runs=runs,
                      dropout=dropout, wd=wd, heads=heads,attention_dropout=attention_dropout,
                      train_examples = train_examples, val_examples = val_examples,isDirected=isDirected)
        df_cur['sbm_num'] = i
        df_val = pd.concat([df_val, df_cur])
    return df_val


def eval_sbm_label(model, dataset_name, directionality, size, dropout, lr, wd, heads,attention_dropout,
        splits, runs, train_examples, val_examples, sbm_inits):
    isDirected = (directionality != 'undirected')
    isReversed = (directionality == 'reversed')
    df_val = pd.DataFrame()
    for i in range(sbm_inits):
        print(f'data/graphs/processed/{dataset_name}/{dataset_name}.content')
        dataset = GraphDataset(f'data/tmp/{dataset_name}{("_" + directionality) if isDirected else ""}-sbm_label{i}', dataset_name,
                               f'data/graphs/sbm_label/{dataset_name}/{dataset_name}_sbm_{i}.cites',
                               f'data/graphs/processed/{dataset_name}/{dataset_name}.content',
                               directed=isDirected, reverse=isReversed)[0]
        df_cur = eval(model=model, dataset=dataset, dataset_name=dataset_name, channel_size=size, lr=lr, splits=splits, runs=runs,
                      dropout=dropout, wd=wd, heads=heads,attention_dropout=attention_dropout,
                      train_examples = train_examples, val_examples = val_examples,isDirected=isDirected)
        df_cur['sbm_num'] = i
        df_val = pd.concat([df_val, df_cur])
    return df_val


def eval_modcm(model, dataset_name, directionality, size, dropout, lr, wd, heads,attention_dropout,
        splits, runs, train_examples, val_examples, modcm_inits):
    isDirected = (directionality != 'undirected')
    isReversed = (directionality == 'reversed')
    df_val = pd.DataFrame()
    for i in range(modcm_inits):
        print(f'data/graphs/processed/{dataset_name}/{dataset_name}.content')
        dataset = GraphDataset(f'data/tmp/{dataset_name}{("_" + directionality) if isDirected else ""}-modcm{i}', dataset_name,
                               f'data/graphs/modcm/{dataset_name}/{dataset_name}_modcm_{i}.cites',
                               f'data/graphs/processed/{dataset_name}/{dataset_name}.content',
                               directed=isDirected, reverse=isReversed)[0]
        df_cur = eval(model=model, dataset=dataset, dataset_name=dataset_name, channel_size=size, lr=lr, splits=splits, runs=runs,
                      dropout=dropout, wd=wd, heads=heads,attention_dropout=attention_dropout,
                      train_examples = train_examples, val_examples = val_examples,isDirected=isDirected)
        df_cur['modcm_num'] = i
        df_val = pd.concat([df_val, df_cur])
    return df_val


def eval_modsbm(model, dataset_name, directionality, size, dropout, lr, wd, heads,attention_dropout,
        splits, runs, train_examples, val_examples, modsbm_inits):
    isDirected = (directionality != 'undirected')
    isReversed = (directionality == 'reversed')
    df_val = pd.DataFrame()
    for i in range(modsbm_inits):
        print(f'data/graphs/processed/{dataset_name}/{dataset_name}.content')
        dataset = GraphDataset(f'data/tmp/{dataset_name}{("_" + directionality) if isDirected else ""}-modsbm{i}', dataset_name,
                               f'data/graphs/modsbm/{dataset_name}/{dataset_name}_modsbm_{i}.cites',
                               f'data/graphs/processed/{dataset_name}/{dataset_name}.content',
                               directed=isDirected, reverse=isReversed)[0]
        df_cur = eval(model=model, dataset=dataset, dataset_name=dataset_name, channel_size=size, lr=lr, splits=splits, runs=runs,
                      dropout=dropout, wd=wd, heads=heads,attention_dropout=attention_dropout,
                      train_examples = train_examples, val_examples = val_examples,isDirected=isDirected)
        df_cur['modsbm_num'] = i
        df_val = pd.concat([df_val, df_cur])
    return df_val


def eval_reglabel(model, dataset_name, directionality, size, dropout, lr, wd, heads,attention_dropout,
        splits, runs, train_examples, val_examples, reglabel_inits):
    isDirected = (directionality != 'undirected')
    isReversed = (directionality == 'reversed')
    df_val = pd.DataFrame()
    for i in range(reglabel_inits):
        print(f'data/graphs/processed/{dataset_name}/{dataset_name}.content')
        dataset = GraphDataset(f'data/tmp/{dataset_name}{("_" + directionality) if isDirected else ""}-reglabel{i}', dataset_name,
                               f'data/graphs/reglabel/{dataset_name}/{dataset_name}_reglabel_{i}.cites',
                               f'data/graphs/processed/{dataset_name}/{dataset_name}.content',
                               directed=isDirected, reverse=isReversed)[0]
        df_cur = eval(model=model, dataset=dataset, dataset_name=dataset_name, channel_size=size, lr=lr, splits=splits, runs=runs,
                      dropout=dropout, wd=wd, heads=heads,attention_dropout=attention_dropout,
                      train_examples = train_examples, val_examples = val_examples,isDirected=isDirected)
        df_cur['reglabel_num'] = i
        df_val = pd.concat([df_val, df_cur])
    return df_val


################## Synthetic part #####################################

def load_communities(path):
    with open(path, 'rb') as handle:
        ret = pickle.load(handle)
    return ret

def load_labels(path):
    label = {}
    with open(path, 'r') as handle:
        label = {}
        for line in handle:
            s = line.strip().split()
            label[s[0]] = s[-1]
    return label
def agg(x):
    return len(x.unique())

def calc_uncertainty(df_community,dataset_name,labeled=False,seed=0):
    
    if dataset_name == 'cora':
        df_community.label = df_community.label.apply(lambda x : ''.join([c for c in x if c.isupper()]))
    
    if labeled:
        df_community = df_community[df_community[f'labeled{seed}']]
    communities = df_community.community.unique()
    labels = df_community.label.unique()
    mtx = df_community.pivot_table(index='community', columns='label',values='node',aggfunc=agg).fillna(0) / len(df_community)
    
    def Pmarg(c):
        return len(df_community[df_community.community == c]) / len(df_community)
    
    def Pcond(l,c):
        return mtx.loc[c,l]/Pmarg(c)
    
    H = 0
    for c in communities:
        h = 0
        for l in labels:
            if Pcond(l,c) == 0:
                continue
            h += Pcond(l,c) * np.log2(1./Pcond(l,c))
        H += h * Pmarg(c)
    
    def Pl(l):
        return len(df_community[df_community.label == l]) / len(df_community)
    
    Hl = 0
    for l in labels:
        if Pl(l) == 0:
            continue
        Hl += Pl(l) * np.log2(1./Pl(l))
    
    IG = Hl-H
    return IG/Hl

def eval_sbm_swap(model, dataset_name, directionality, size, dropout, lr, wd, heads,attention_dropout,
        splits, runs, train_examples, val_examples, sbm_inits, is_sbm):
    step = 10
    isDirected = (directionality != 'undirected')
    isReversed = (directionality == 'reversed')
    df_val = pd.DataFrame()
    
    
    for i in range(sbm_inits if is_sbm else 1):
        print(f'data/graphs/processed/{dataset_name}/{dataset_name}.content')
        if is_sbm:
          dataset = GraphDataset(f'data/tmp/{dataset_name}{("_" + directionality) if isDirected else ""}-sbm{i}-', dataset_name,
                               f'data/graphs/sbm/{dataset_name}/{dataset_name}_sbm_{i}.cites',
                               f'data/graphs/processed/{dataset_name}/{dataset_name}.content',
                               directed=isDirected, reverse=isReversed)
        else:
          dataset = GraphDataset(f'data/tmp/{dataset_name}{("_" + directionality) if isDirected else ""}-', dataset_name,
                               f'data/graphs/processed/{dataset_name}/{dataset_name}.cites',
                               f'data/graphs/processed/{dataset_name}/{dataset_name}.content',
                               directed=isDirected, reverse=isReversed)

        data = dataset[0]
        
        community = load_communities(f'data/community_id_dicts/{dataset_name}/{dataset_name}_louvain.pickle')
        
        mapping = data.node_name_mapping
        label = load_labels(f'data/graphs/processed/{dataset_name}/{dataset_name}.content')
        df_community = pd.DataFrame({'dataset':dataset_name, 'node':node, 'community':community[node], 'label':label[node]} for node in community)
        df_community['node_id'] = df_community.node.apply(lambda x:mapping[x])

        for seed in range(splits):
            split = NetworkSplitShchur(dataset, train_examples_per_class=train_examples,early_examples_per_class=0,
                 val_examples_per_class=val_examples, split_seed=seed)
            df_community[f'labeled{seed}'] = df_community.node_id.apply(lambda x: (split.train_mask[x]).numpy())
        
        n = len(data.y)
        # select nodes at random
        shuffled = np.arange(n)
        np.random.shuffle(shuffled)
        row = shuffled[:int(n/2)]
        col = shuffled[int(n/2):int(n/2)*2]
        assert(len(row) == len(col))
        
        df_cur = eval(model=model, dataset=dataset, dataset_name=dataset_name, channel_size=size, lr=lr, splits=splits, runs=runs,
                      dropout=dropout, wd=wd, heads=heads,attention_dropout=attention_dropout,
                      train_examples = train_examples, val_examples = val_examples,isDirected=isDirected)
        if is_sbm:
          df_cur['sbm_num'] = i
        df_cur['ratio'] = 0
        df_cur['uncertainty'] = calc_uncertainty(df_community, dataset_name)
        ulc = [calc_uncertainty(df_community, dataset_name, True, seed) for seed in range(splits)]
        df_cur['uncertainty_known'] = [ulc]
        print(df_cur)
        df_val = pd.concat([df_val, df_cur])

        for ratio in range(0,100,step):
            frm = int(ratio/100 * len(row))
            to = int((ratio+step)/100 * len(row))
            U = row[frm:to]
            V = col[frm:to]
            for u,v in zip(U,V):
                tmp = data.x[v].detach().clone()
                data.x[v] = dataset[0].x[u]
                data.x[u] = tmp
                
                tmp = data.y[v].detach().clone()
                data.y[v] = dataset[0].y[u]
                data.y[u] = tmp
                
                tmp = df_community.loc[df_community.node_id == v, 'community'].values[0]
                df_community.loc[df_community.node_id == v, 'community'] = df_community.loc[df_community.node_id == u, 'community'].values[0]
                df_community.loc[df_community.node_id == u, 'community'] = tmp
                  
            df_cur = eval(model=model, dataset=dataset, dataset_name=dataset_name, channel_size=size, lr=lr, splits=splits, runs=runs,
                          dropout=dropout, wd=wd, heads=heads,attention_dropout=attention_dropout,
                          train_examples = train_examples, val_examples = val_examples,isDirected=isDirected)
            if is_sbm:
              df_cur['sbm_num'] = i
            df_cur['ratio'] = ratio+step
            df_cur['uncertainty'] = calc_uncertainty(df_community, dataset_name)
            ulc = [calc_uncertainty(df_community, dataset_name, True, seed) for seed in range(splits)]
            df_cur['uncertainty_known'] = [ulc]
            print(df_cur)
            df_val = pd.concat([df_val, df_cur])
    return df_val

################## END: Synthetic part #####################################



def eval_flipped(model, dataset_name, directionality, size, dropout, lr, wd, heads,attention_dropout,
        splits, runs, train_examples, val_examples, percentages=range(10,51,10)):
    print(percentages)
    isDirected = (directionality != 'undirected')
    isReversed = (directionality == 'reversed')
    df_val = pd.DataFrame()
    for i in percentages:
        print(f'data/graphs/processed/{dataset_name}/{dataset_name}.content')
        dataset = GraphDataset(f'data/tmp/{dataset_name}{("_" + directionality) if isDirected else ""}-flipped{i}', dataset_name,
                               f'data/graphs/flip_edges/{dataset_name}/{dataset_name}_{i}.cites',
                               f'data/graphs/processed/{dataset_name}/{dataset_name}.content',
                               directed=isDirected, reverse=isReversed)[0]
        df_cur = eval(model=model, dataset=dataset, dataset_name=dataset_name, channel_size=size, lr=lr, splits=splits, runs=runs,
                      dropout=dropout, wd=wd, heads=heads,attention_dropout=attention_dropout,
                      train_examples = train_examples, val_examples = val_examples,isDirected=isDirected)
        df_cur['percentage'] = i
        df_val = pd.concat([df_val, df_cur])
    return df_val

def eval_removed_hubs(model, dataset_name, directionality, size, dropout, lr, wd, heads,attention_dropout,
        splits, runs, train_examples, val_examples, percentages=[1,2,4,8]):
    print(percentages)
    isDirected = (directionality != 'undirected')
    isReversed = (directionality == 'reversed')
    df_val = pd.DataFrame()
    for i in percentages:
        print(f'data/graphs/processed/{dataset_name}/{dataset_name}.content')
        dataset = GraphDataset(f'data/tmp/{dataset_name}{("_" + directionality) if isDirected else ""}-removed-hubs{i}', dataset_name,
                               f'data/graphs/removed_hubs/{dataset_name}/{dataset_name}_{i:02}.cites',
                               f'data/graphs/processed/{dataset_name}/{dataset_name}.content',
                               directed=isDirected, reverse=isReversed)[0]
        df_cur = eval(model=model, dataset=dataset, dataset_name=dataset_name, channel_size=size, lr=lr, splits=splits, runs=runs,
                      dropout=dropout, wd=wd, heads=heads,attention_dropout=attention_dropout,
                      train_examples = train_examples, val_examples = val_examples,isDirected=isDirected)
        df_cur['percentage'] = i
        df_val = pd.concat([df_val, df_cur])
    return df_val

def eval_added_2hop_edges(model, dataset_name, directionality, size, dropout, lr, wd, heads,attention_dropout,
        splits, runs, train_examples, val_examples, percentages=[1,2,4,8,16,32,64,128,256,512]):
    print(percentages)
    isDirected = (directionality != 'undirected')
    isReversed = (directionality == 'reversed')
    df_val = pd.DataFrame()
    for i in percentages:
        print(f'data/graphs/processed/{dataset_name}/{dataset_name}.content')
        network_path = f'data/graphs/added_2hop_edges/{dataset_name}/{dataset_name}_{i:02}.cites'
        if not os.path.exists(network_path):
            continue
        dataset = GraphDataset(f'data/tmp/{dataset_name}{("_" + directionality) if isDirected else ""}-added-2hops{i}', dataset_name,
                               network_path,
                               f'data/graphs/processed/{dataset_name}/{dataset_name}.content',
                               directed=isDirected, reverse=isReversed)[0]
        df_cur = eval(model=model, dataset=dataset, dataset_name=dataset_name, channel_size=size, lr=lr, splits=splits, runs=runs,
                      dropout=dropout, wd=wd, heads=heads,attention_dropout=attention_dropout,
                      train_examples = train_examples, val_examples = val_examples,isDirected=isDirected)
        df_cur['percentage'] = i
        df_val = pd.concat([df_val, df_cur])
    return df_val
    
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    args = parse_args()
    if args.directionality not in {'undirected', 'reversed', 'directed'}:
        print("--directionality must be in {'undirected','reversed','directed'}")
        exit(1)
    isDirected = (args.directionality != 'undirected')
    isReversed = (args.directionality == 'reversed')

    

    # TODO find a better way to create names
    val_out = f'reports/results/test_acc/{args.model}_{args.dataset}{"_conf" if args.conf else ""}' \
              f'{"_sbm" if args.sbm else ""}{("_" + args.directionality) if isDirected else ""}.csv'

    if os.path.exists(val_out):
        df_val = pd.read_csv(val_out)
    else:
        df_val = pd.DataFrame(
            columns='conv arch ch dropout lr wd heads attention_dropout splits inits val_accs val_avg val_std'
                    ' test_accs test_avg test_std stopped elapsed'.split())
    if args.conf:
        df_cur = eval_conf(args.model, args.dataset, args.directionality, args.size, args.dropout, args.lr, args.wd,
                args.heads, args.attention_dropout,
                args.splits, args.runs, args.train_examples, args.val_examples, args.conf_inits)
    if args.shifting:
        df_cur = eval_shifting(args.model, args.dataset, args.directionality, args.size, args.dropout, args.lr, args.wd,
                args.heads, args.attention_dropout,
                args.splits, args.runs, args.train_examples, args.val_examples, args.shifting_inits)
    elif args.sbm:
        df_cur = eval_sbm(args.model, args.dataset, args.directionality, args.size, args.dropout, args.lr, args.wd,
                args.heads, args.attention_dropout,
                args.splits, args.runs, args.train_examples, args.val_examples, args.sbm_inits)
    elif args.sbm_label:
        df_cur = eval_sbm_label(args.model, args.dataset, args.directionality, args.size, args.dropout, args.lr, args.wd,
                args.heads, args.attention_dropout,
                args.splits, args.runs, args.train_examples, args.val_examples, args.sbm_inits)
    elif args.flipped:
        df_cur = eval_flipped(args.model, args.dataset, args.directionality, args.size, args.dropout, args.lr, args.wd,
                args.heads, args.attention_dropout,
                args.splits, args.runs, args.train_examples, args.val_examples)
    elif args.removed_hubs:
        df_cur = eval_removed_hubs(args.model, args.dataset, args.directionality, args.size, args.dropout, args.lr, args.wd,
                args.heads, args.attention_dropout,
                args.splits, args.runs, args.train_examples, args.val_examples)
    elif args.added_2hop_edges:
        df_cur = eval_added_2hop_edges(args.model, args.dataset, args.directionality, args.size, args.dropout, args.lr, args.wd,
                args.heads, args.attention_dropout,
                args.splits, args.runs, args.train_examples, args.val_examples)
    elif  args.label_sbm:
        df_cur = eval_label_sbm(args.model, args.dataset, args.directionality, args.size, args.dropout, args.lr, args.wd,
                args.heads, args.attention_dropout,
                args.splits, args.runs, args.train_examples, args.val_examples)
    elif  args.injected_edges:
        df_cur = eval_injected_edges(args.model, args.dataset, args.directionality, args.size, args.dropout, args.lr, args.wd,
                args.heads, args.attention_dropout,
                args.splits, args.runs, args.train_examples, args.val_examples, 5, range(1000,5001,1000), args.hubs_experiment)
    elif  args.injected_edges_degree_cat:
        df_cur = eval_injected_edges_degree_cat(args.model, args.dataset, args.directionality, args.size, args.dropout, args.lr, args.wd,
                args.heads, args.attention_dropout,
                args.splits, args.runs, args.train_examples, args.val_examples, 5, 500, 5)
    elif  args.injected_edges_sbm:
        df_cur = eval_injected_edges_sbm(args.model, args.dataset, args.directionality, args.size, args.dropout, args.lr, args.wd,
                args.heads, args.attention_dropout,
                args.splits, args.runs, args.train_examples, args.val_examples, 5, range(100,2001,100), args.hubs_experiment)
    else:
        df_cur = eval_original(args.model, args.dataset, args.directionality, args.size, args.dropout, args.lr, args.wd,
                args.heads, args.attention_dropout,
                args.splits, args.runs, args.train_examples, args.val_examples)
    df_val = pd.concat([df_val, df_cur])
    df_val.to_csv(val_out, index=False)
