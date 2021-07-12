import torch
import torch.nn.functional as F
from src.evaluation.network_split import NetworkSplitShchur
from sklearn.linear_model import LogisticRegression
from src.data.data_loader import EmbeddingData


def test_embedding(train_z, train_y, test_z, test_y, solver='lbfgs',
                   multi_class='ovr', seed=None, *args, **kwargs):
    r"""Evaluates latent space quality via a logistic(?) regression downstream
    task."""
    z = train_z.detach().cpu().numpy()
    y = train_y.detach().cpu().numpy()
    logreg = LogisticRegression(solver=solver, multi_class=multi_class, random_state=seed, n_jobs=1, *args, **kwargs)
    clf = logreg.fit(z, y)
    return clf.score(test_z.detach().cpu().numpy(),
                     test_y.detach().cpu().numpy())

class MLP(torch.nn.Module):
    def __init__(self,data):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(data.x.shape[1], 96)
        self.fc2 = torch.nn.Linear(96,int(data.y.max())+1)

    def forward(self,data):
        x = data.x
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x),dim=1)
        return x


def train_mlp(data,split,validation=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP(data).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    if validation:
        mask = split.val_mask
    else:
        mask = split.test_mask

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[split.train_mask], data.y[split.train_mask].type(torch.LongTensor).to(device))
        loss.backward()
        optimizer.step()
        
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = int(pred[mask].eq(data.y[mask].type(torch.LongTensor).to(device)).sum().item())
    acc = correct / int(mask.sum())
    return acc

    

def eval_method(data, num_splits=100,
                train_examples = 20, val_examples = 30):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = data[0].x
    z = z.to(device)
    vals = []
    for i in range(num_splits):
        split = NetworkSplitShchur(data, train_examples_per_class=train_examples,early_examples_per_class=0,
                 val_examples_per_class=val_examples, split_seed=i)
        val = train_mlp(data[0], split, validation=True)
        # val = test_embedding(z[split.train_mask], data[0].y[split.train_mask],
        #                      z[split.val_mask], data[0].y[split.val_mask], max_iter=100)
        vals.append(val)
    return vals


def test_method(data, num_splits=100,
                train_examples = 20, val_examples = 30,seed=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = data[0].x
    z = z.to(device)
    tests = []
    for i in range(num_splits):
        split = NetworkSplitShchur(data, train_examples_per_class=train_examples,early_examples_per_class=0,
                 val_examples_per_class=val_examples, split_seed=i)
        ts = train_mlp(data[0], split, validation=False)
        # ts = test_embedding(z[split.train_mask], data[0].y[split.train_mask],
        #                     z[split.test_mask], data[0].y[split.test_mask], max_iter=100,seed=seed)
        tests.append(ts)
    return tests


def report_test_acc_unsupervised_embedding(tmp,dataset,embfile,attrfile,
                        num_splits, train_examples, val_examples):
    tests = []
    emb = EmbeddingData(tmp, dataset,embfile,attrfile)
    print(f'started test {dataset}')
    test = test_method(emb, num_splits=num_splits, train_examples = train_examples, val_examples = train_examples)
    print(str(test) + '\n')
    tests = tests + test
    return tests