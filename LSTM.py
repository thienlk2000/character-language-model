import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def to_vec(s, char_to_idx):
    x = torch.zeros((len(s), len(char_to_idx)))
    idxs = [char_to_idx[c] for c in s]
    x[torch.arange(len(s)), idxs] = 1
    return x

def to_vec_target(s, char_to_idx):
    idxs = [char_to_idx[c] for c in s]
    
    return torch.tensor(idxs)

def train(model, output,data,optimizer, loss_fn, smooth_loss): # train 1 epoch
    h0 = (torch.zeros((1, batch_size, hidden_dim)),torch.zeros((1, batch_size, hidden_dim)))
    p = 0
    h_prev = h0
    n = 0
    while(p < len(data) -1):
        n += 1
        if (len(data) - p - 1) >= length: 
            seq = data[p:p+length]
            target = data[p+1:p+length+1]
        else:
            seq = data[p:len(data)-1]
            target = data[p+1:len(data)]

        
        x = to_vec(seq, char_to_idx)
        x = x.view((x.shape[0], batch_size, -1))
        y = to_vec_target(target, char_to_idx)
        o, h = model(x, h_prev)
        o = o.view(-1, hidden_dim)
        preds = output(o) #L, input_dim
        loss = loss_fn(preds, y)
        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if n % 100 == 0:
            result = sample(model, output, h_prev, char_to_idx[seq[0]], 500)
            text = ''.join(idx_to_char[s] for s in result)
            print(f'----\n {text} \n----')
            
            smooth_loss = 0.999*smooth_loss + 0.001*loss.item()
            print(f"Loss:{smooth_loss}")
        h_prev = (h[0].detach(),h[1].detach())
        p += length
    return smooth_loss
def sample(model,out,h_prev, id_first, num_length):
    result = [id_first]
    x = torch.zeros((1,batch_size, input_dim))
    x[:,:,id_first] = 1
    with torch.no_grad():
        for i in range(num_length):
            o, h_prev = model(x, h_prev)
            s = out(o).view(-1, input_dim).numpy()
            p = np.exp(s) / np.exp(s).sum(axis=1).reshape(-1,1)
            idx = np.random.choice(range(input_dim), p=p.ravel())
            result.append(idx)
            x = torch.zeros((1,batch_size, input_dim))
            x[:,:,idx] = 1
    return result


with open('input.txt', 'r') as f:
    data = f.read()

chars = list(set(data))
print("Data size:", len(data))
print("Volca size:", len(chars))
char_to_idx = {v: k for k, v in enumerate(chars)}
idx_to_char = {k:v for k,v in enumerate(chars)}

hidden_dim = 100
input_dim = len(chars)
length = 100
batch_size = 1

lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim)
output = nn.Linear(hidden_dim, len(chars))
loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam([
    {'params': lstm.parameters()},
    {'params': output.parameters()}
], lr=1e-3)
smooth_loss = -np.log(1.0/input_dim)*length 
while True:
    smooth_loss = train(lstm, output, data, optimizer, loss_fn,smooth_loss)