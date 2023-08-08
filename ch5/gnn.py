import torch
from torch_geometric.datasets import Planetoid, FacebookPagePage
import numpy as np
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

fb = FacebookPagePage(root=".")
cora = Planetoid(root=".", name="cora")
n_examples = fb.data.x.shape[0]

100*18000/n_examples
100*(20000-18001)/n_examples
100*(n_examples-20001)/n_examples

fb.data.train_mask = range(18000)
fb.data.val_mask = range(18001, 20000)
fb.data.test_mask = range(20001, 22470)

data = fb.data

class VanillaGNNLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear = torch.nn.Linear(dim_in, dim_out, bias=False)

    def forward(self, x, adjacency):
        x = self.linear(x)
        x = torch.sparse.mm(adjacency, x)        
        return x
    
class VanillaGNN(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out) -> None:
        super().__init__()
        self.gnn1 = VanillaGNNLayer(dim_in, dim_h)
        self.gnn2 = VanillaGNNLayer(dim_h, dim_out)
        
    def accuracy(self, y_pred, y_true):
        return torch.sum(y_pred==y_true)/len(y_true)
    
    def forward(self, x, adjacency):
        h = self.gnn1(x, adjacency)
        h = torch.relu(h)
        h = self.gnn2(h, adjacency)
        return F.log_softmax(h, dim=1)
    
    def fit(self, data, epochs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        self.train()
        for epoch in range(epochs+1):
            optimizer.zero_grad()
            out = self.forward(data.x, adjacency)
            
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            
            acc = self.accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            if epoch % 20 :
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
                val_acc = self.accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])
                msg = f"""
                Epoch: {epoch:>3} | Train loss: {loss} | Train Acc: {acc*100:>5.2f}% | Val loss: {val_loss} | Val Acc: {val_acc*100:>5.2f}% 
                """
                print(msg)
            
            

nn = VanillaGNN(fb.num_features, 16, fb.num_classes)

adjacency = to_dense_adj(fb.data.edge_index)[0]
adjacency += torch.eye(len(adjacency))

nn.fit(data, 50)

nn = VanillaGNN(cora.num_features, 16, cora.num_classes)
adjacency = to_dense_adj(cora.data.edge_index)[0]
adjacency += torch.eye(len(adjacency))
nn.fit(cora.data, epochs=30)