import numpy as np 
import pandas as pd 


from sklearn.model_selection import train_test_split

training_data = pd.read_csv('../Desktop/cil.csv',header = None)
training_data.replace(-1, 176, inplace = True)
training_data = np.array(training_data)
train_data = training_data[:round(0.8*len(training_data)), :]
val_data = training_data[round(0.8*len(training_data)):, :]

#train_test_split(training_data, test_size = 0.2, random_state = 42)

# +
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# +
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_groups):
        super().__init__()
        assert num_heads % num_groups == 0, "Number of heads must be divisible by the number of groups"

        """
        Multi-Head Attention has num_heads == num_groups
        Grouped-Query Attention has num_heads % num_grous == 0
        Multi-Query Attention has num_groups == 1 & num_heads == seq_len
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = d_model // num_heads 
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * num_groups * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x, past_kv=None):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k, v = self.kv_proj(x).chunk(2, dim=-1) #split the sequence into 2
        k = k.view(batch_size, seq_len, self.num_groups, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_groups, self.head_dim)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1) 
            
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        output = (attn_weights @ v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(output), (k, v)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, num_groups, dim_feedforward):
        super().__init__()
        self.self_attn = GroupedQueryAttention(d_model, num_heads, num_groups)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x, past_kv=None):
        attn_output, kv_cache = self.self_attn(x, past_kv)
        x = self.dropout1(attn_output) + x
        x = self.norm1(x)
        
        ff_output = self.linear2(F.relu(self.linear1(x)))
        x = self.dropout2(ff_output) + x
        x = self.norm2(x)
        
        return x, kv_cache

class AutoregressiveModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, num_groups, dim_feedforward):
        super().__init__()
        max_length = 99
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self._get_sinusoidal_encoding(d_model, max_length)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, num_groups, dim_feedforward)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def _get_sinusoidal_encoding(self, d_model, max_length):
        # Generates a sinusoidal positional encoding matrix
        pos_encoding = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.to(device)
    
    def forward(self, x, past_kvs=None):
        x = self.embedding(x)
        x += self.positional_encoding
        new_kvs = []
        
        for i, layer in enumerate(self.layers):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, kv_cache = layer(x, past_kv)
            new_kvs.append(kv_cache)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_kvs



# +
vocab_size = 177
d_model = 200 
num_layers = 3 
num_heads = 20 
num_groups = 2 
dim_feedforward = 400

model = AutoregressiveModel(vocab_size, d_model, num_layers, num_heads, num_groups, dim_feedforward).to(device)
CEL = nn.CrossEntropyLoss().to(device)

# +
import torch.utils.data as data

BATCH_SIZE = 128 

all_train_iterator = data.DataLoader(training_data,
                                    shuffle = True,
                                    batch_size = BATCH_SIZE)
train_iterator = data.DataLoader(train_data,
                                shuffle = True,
                                batch_size = BATCH_SIZE)
val_iterator = data.DataLoader(val_data,
                              shuffle = True,
                              batch_size = BATCH_SIZE)


# +
import torch.optim as optim

params = [
          {'params': model.embedding.parameters()}, #, 'lr': 5e-4 / 10},
          {'params': model.layers.parameters()}
         ]

optimizer = optim.AdamW(params, lr=1e-3, weight_decay = 1e-4)


# +

from tqdm.notebook import trange, tqdm

def train(model, iterator, optimizer, CEL, device):

    epoch_loss = 0
    epoch_accuracy = 0

    model.train()

    for x in tqdm(iterator, desc="Training", leave=False):

        x = x.to(device)
        
        optimizer.zero_grad()

        y = x[:, 1:]
        x = x[:, :-1]

        y_pred, _ = model(x)

        loss = CEL(y_pred.view(-1, y_pred.size(-1)), y.reshape(-1))

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)



def evaluate(model, iterator, optimizer, CEL, device):

    epoch_loss = 0
    epoch_accuracy = 0

    model.eval()

    for x in tqdm(iterator, desc="evaluating", leave=False):

        x = x.to(device)

        optimizer.zero_grad()

        y = x[:, 1:]
        x = x[:, :-1]

        y_pred, _ = model(x)

        loss = CEL(y_pred.view(-1, y_pred.size(-1)), y.reshape(-1))

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


# +
import matplotlib.pyplot as plt

EPOCHS = 20

best_val_loss = float('inf')

train_losses = []
val_losses = []

for epoch in trange(EPOCHS):

    train_loss = train(model, train_iterator, optimizer, CEL, device)
    val_loss = evaluate(model, val_iterator, optimizer, CEL, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'tut1-model.pt')

    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\t Val. Loss: {val_loss:.3f}')

plt.plot(range(1, EPOCHS + 1), train_losses, label='Train Loss')
plt.plot(range(1, EPOCHS + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train/Validation Loss')
plt.legend()
plt.show()
