from random import sample
from random import seed
from random import random
import networkx as nx
from src.data.create_stochastic_block_model import *


def equals(a,b):
    return a==b

def nequals(a,b):
    return a!=b

def shifting(G, L, C, hl, k, inc, random_state=2021):
    '''
    Changes hl while keeping hc close to the original
    G: graph to change
    C: C[u] the C membership of node [u]
    L: L[u] the L membership of node [u]
    hl: |{e in E: L[e[0]]==L[e[1]]}|/|E| that is L membership homophily
    k: k edges to rewire
    inc: True to increase hl, and False to decrease it
    '''

    seed(random_state)
    eq = equals if inc else nequals
    attempts_limit = k*10


    triples = []
    attempts = 0
    
    # edges = set(edge_list)
    while len(triples) < k:
        if attempts > attempts_limit:
            print(f"Number of attempts reached {attempts_limit}. Found {len(triples)} edges to rewire.")
            break
        attempts+=1
        edges = [e for e in G.edges if not eq(C[e[0]],C[e[1]])] + [[e[1],e[0]] for e in G.edges if not eq(C[e[0]],C[e[1]])]
        if len(edges) == 0:
        	print(f"Could not find candidate edges to rewire.")
        	return None
        u,v = sample(edges,1)[0]
        nodes = set([w for w in G.nodes if eq(C[w],C[u])]) - set(G.neighbors(u))
        if len(nodes) == 0:
        	continue
        w = sample(nodes,1)[0]

        p = 1 if L[v]==L[w] else (1-hl if L[u]!=L[w] else hl)

        # print(f'u {u}\t{C[u]} {L[u]}')
        # print(f'v {v}\t{C[v]} {L[v]}')
        # print(f'w {w}\t{C[w]} {L[w]}')
        # print(f'{p:.2f}')
        # print('------')

        if random() < p:
            if G.has_edge(u,v):
                G.remove_edge(u,v)
            else:
                G.remove_edge(v,u)
            G.add_edge(u,w)
            triples.append([u,v,w])

    return G


def compute_homophily(G, prop):
    return len([e for e in G.edges if prop[e[0]]==prop[e[1]]])/G.number_of_edges()

def generate_multiple_shifting(graph_path, content_path, community_path, output_prefix,output_suffix = '.cites',inits=3):
    fin = graph_path
    G=nx.read_edgelist(fin,create_using=nx.DiGraph())
    L,_ = load_labels(content_path)
    C = load_communities(community_path,G)

    prop = {'C':C, 'L':L}
    for keep,change in zip('LC','CL'):
        p_keep = prop[keep]
        p_change = prop[change]
        for inc in [True, False]:
            for r in [0.16,0.32,0.64]:
                k = int(r*len(G.edges))
                for i in range(inits):
                    new_graph_path = f'{output_prefix}_{change}_{"inc" if inc else "dec"}_r{r:.2f}_{i}{output_suffix}'
                    print(f'{graph_path} ... {new_graph_path}')
                    G=nx.read_edgelist(fin,create_using=nx.DiGraph())
                    GOld = G.copy()
                    GNoisy = shifting(G, p_keep ,p_change, compute_homophily(G, p_keep), k, inc, random_state=i)
                    if GNoisy is None:
                    	print('Failed to rewire with given parameters.')
                    	continue
                    print(f'h{keep} = {compute_homophily(GOld,p_keep):.4f} to {compute_homophily(GNoisy,p_keep):.4f}')
                    print(f'h{change} = {compute_homophily(GOld,p_change):.4f} to {compute_homophily(GNoisy,p_change):.4f}')
                    nx.write_edgelist(GNoisy,new_graph_path)



if __name__ == "__main__":
    datasets = 'texas wisconsin cora citeseer actor pubmed cora_full squirrel'.split()
    for dataset in datasets:
        if not os.path.exists(f'data/graphs/shifting/'):
            os.mkdir(f'data/graphs/shifting/')
        if not os.path.exists(f'data/graphs/shifting/{dataset}/'):
            os.mkdir(f'data/graphs/shifting/{dataset}/')
        generate_multiple_shifting(f'data/graphs/processed/{dataset}/{dataset}.cites',
                                f'data/graphs/processed/{dataset}/{dataset}.content',
                                f'data/community_id_dicts/{dataset}/{dataset}_louvain.pickle',
                                f'data/graphs/shifting/{dataset}/{dataset}_shifting',inits=5)