from src.evaluation.gnn_evaluation_module import train_and_get_embeddings
from src.models.multi_layered_model import MonoModel
from torch_geometric.nn import GCNConv
from src.data.data_loader import GraphDataset



def eval_original(dataset_name, directionality='undirected', 
        train_examples=20, val_examples=30):
    isDirected = (directionality != 'undirected')
    isReversed = (directionality == 'reversed')
    dataset = GraphDataset(f'data/tmp/{dataset_name}{("_" + directionality) if isDirected else ""}', dataset_name,
                           f'data/graphs/processed/{dataset_name}/{dataset_name}.cites',
                           f'data/graphs/processed/{dataset_name}/{dataset_name}.content',
                           directed=isDirected, reverse=isReversed)
    emb = train_and_get_embeddings(dataset, channels=96, modelType=GCNConv, architecture=MonoModel,
              lr=0.01, wd=0.01, heads=1, dropout=0.8, attention_dropout=None,
              epochs=200,
              train_examples=train_examples, val_examples=val_examples,
              split_seed=0, init_seed=0,
              test_score=False, actual_predictions=False, add_complete_edges=False)
    return emb


    
def eval_modcm(dataset_name, directionality='undirected',
        train_examples=20, val_examples=30, modcm_inits=1):
    isDirected = (directionality != 'undirected')
    isReversed = (directionality == 'reversed')
    df_val = pd.DataFrame()
    for i in range(modcm_inits):
        print(f'data/graphs/processed/{dataset_name}/{dataset_name}.content')
        dataset = GraphDataset(f'data/tmp/{dataset_name}{("_" + directionality) if isDirected else ""}-modcm{i}', dataset_name,
                               f'data/graphs/modcm/{dataset_name}/{dataset_name}_modcm_{i}.cites',
                               f'data/graphs/processed/{dataset_name}/{dataset_name}.content',
                               directed=isDirected, reverse=isReversed)
        emb = train_and_get_embeddings(dataset, channels=96, modelType=GCNConv, architecture=MonoModel,
              lr=0.01, wd=0.01, heads=1, dropout=0.8, attention_dropout=None,
              epochs=200,
              train_examples=train_examples, val_examples=val_examples,
              split_seed=0, init_seed=0,
              test_score=False, actual_predictions=False, add_complete_edges=False)
    return emb