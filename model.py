from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 5) # -> 24x24
        self.pool1 = nn.MaxPool2d(2) # -> 12x12
        self.conv2 = nn.Conv2d(64, 128, 5) # -> 8x8
        self.dropout = nn.Dropout(p=0.4)
        self.dense = nn.Linear(128 * 8 * 8, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.view(x.size(0), -1) # Flatten
        return F.relu(self.dense(x))
    
def _average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size

def train(model, dataloader_train, is_distributed):
    device = device_("cuda" if cuda.is_available() else "cpu")
    model.train()
    for data, target in dataloader_train:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        if is_distributed and not cuda.is_available():
            # average gradients manually for multi-machine cpu case only
            _average_gradients(model)
        optimizer.step()