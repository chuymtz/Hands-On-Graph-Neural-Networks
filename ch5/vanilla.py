import pandas as pd
from torch_geometric.datasets import FacebookPagePage, Planetoid
import torch
from torch.nn import Linear
import torch.nn.functional as F

# dataset = FacebookPagePage(root=".")
dataset = Planetoid(root=".",name="cora")
data = dataset[0]

data.x.shape
data.y.shape


df_x = pd.DataFrame(data.x.numpy())
df_x['label'] = pd.DataFrame(data.y)
df_x.head()

# |> METRIC ----------------------------

def accuracy(y_pred, y_true):
    return torch.sum(y_pred==y_true)/len(y_true)


# |> ----------------------------

dataset.num_features == data.x.numpy().shape[1]
dataset.num_classes == len(data.y.unique())

dim_in = data.x.numpy().shape[1]
dim_h = 16
dim_out = len(data.y.unique())


class MLP(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out) -> None:
        super().__init__()
        self.linear1 = Linear(dim_in, dim_h)
        self.linear2 = Linear(dim_h, dim_out)
        
    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return F.log_softmax(x, dim=1)
        
    def fit(self, data, epochs):
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        
        self.train()
        for epoch in range(epochs+1):
            optimizer.zero_grad()
            out = self(data.x)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
                val_acc = accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])
                print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: {acc*100:>5.2f}% | Val Loss: {val_loss:.2f} | Val Acc: {val_acc*100:.2f}%')
            
    def test(self, data):
        self.eval()
        out = self(data.x)
        acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
        return acc

mlp = self = MLP(dim_in, dim_h, dim_out)
# self.forward(data.x)*2
mlp.fit(data, epochs=100)

acc = mlp.test(data)
print(f'MLP test accuracy: {acc*100:.2f}%')


fb = FacebookPagePage(root=".")[0]

dim_in = fb.x.numpy().shape[1]
dim_h = 16
dim_out = len(fb.y.unique())

fb.train_mask = range(18000)
fb.val_mask = range(18001, 20000)
fb.test_mask = range(20001, 22470)

mlp = MLP(dim_in, dim_h, dim_out)
# self.forward(data.x)*2
mlp.fit(fb, epochs=100)

acc = mlp.test(fb)
print(f'MLP test accuracy: {acc*100:.2f}%')



















