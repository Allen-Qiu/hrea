# training the student 2 model

import numpy as np
import os, math, time
import torch
import pickle
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.data import random_split
from hrea_amazon_model import Student2

print(os.path.basename(__file__))

# 1. hyper-parameters
encoder_name = 'encoder.pth'
filename = 'encoder_parameters.pl'
infile = open(filename, 'rb')
parameters_dic = pickle.load(infile)
infile.close()

model_name = 'pkg_level1_bank50.model'
# review_file = 'amazon-yelp-review-100k.json'
epochs = 30
lr = 0.003
alpha = 1
batch_size = 400
bank2_size = 10         # level 2
num_workers = 12
embed_dim = parameters_dic['embedding_dim']
enc_dim = parameters_dic['encoder_enc_dim']
word_freq = parameters_dic['word_freq']
max_seq_length = parameters_dic['max_seq_length']
train_test_ratio = 0.9

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")

# 2. set the seed and GPU
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
print("Using device", device)

# 3. reading files, loading dataset
with open('vocab.pl', 'rb') as f:
    vocab = pickle.load(f)

with open('category.pl', 'rb') as f:
    category = pickle.load(f)

with open('dataset2.pl','rb') as f:
    dataset = pickle.load(f)

with open('bank1.out', 'rb') as f:
    bank1 = pickle.load(f)
    bank1 = torch.Tensor(bank1).to(device)

nouns = vocab.nouns_id
dic = vocab.get_idx_to_token()
print(f'nouns size: {len(nouns)}')
# vocab_size = len(dic)
# print(f'vocabulary size: {vocab_size}')

# 4. model: Hiercluster of level 2
daae = torch.load(encoder_name)  # pretrained encoder
encoder = daae.module.encoder
encoder = nn.DataParallel(encoder.to(device))
for param in encoder.parameters():  # set encoder untrainable
    param.requires_grad = False

model_l2 = Student2(encoder, bank1, (bank2_size, embed_dim), enc_dim).to(device)

# 5. 损失函数 loss = dist(Se,Sp) + Sp*Sn
pdist = torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)
def myloss(Se, Sp, Sn):
    a = torch.mean(pdist(Se, Sp))
    b = torch.mean(torch.sum(Sp * Sn, dim=1))
    loss = a + alpha*torch.maximum(torch.tensor(0), b)
    return loss

# 6. training
def get_dataloader(dataset, train_test_ratio):
    train_size = math.floor(len(dataset) * train_test_ratio)
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
    return train_loader, test_loader

def train_loop(train_dataloader, model_l2, loss_fn, optimizer):
    total_loss = 0
    num_batches = len(train_dataloader)
    
    for Xp, Xn, lp, ln, _, _ in train_dataloader: # p positive, n negative
        lp = torch.tensor(lp, dtype=torch.int32)
        ln = torch.tensor(ln, dtype=torch.int32)
        Sp, Se = model_l2(Xp.to(device), lp)
        Sn, _  = model_l2(Xn.to(device), ln)
        loss = loss_fn(Se, Sp, Sn)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss/num_batches

def test_loop(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X1, X2, s1, s2, _, _ in dataloader:
            Sp, Se = model(X1.to(device), s1)
            Sn, _  = model(X2.to(device), s2)
            test_loss += loss_fn(Se, Sp, Sn).item()

    test_loss /= num_batches
    return test_loss

optimizer = torch.optim.Adamax(model_l2.parameters(), lr=lr, betas=(0.5, 0.99))

for t in range(epochs):
    start_time = time.time()
    train_loader, test_loader = get_dataloader(dataset, train_test_ratio)
    model_l2.train()
    train_loss = train_loop(train_loader, model_l2, myloss, optimizer)
    run_time = time.time() - start_time
    model_l2.eval()
    valid_loss = test_loop(test_loader, model_l2, myloss)
    print(f"epoch {t+1:2d}/{epochs}, train loss: {train_loss:0.3f}, valid loss:{valid_loss:0.3f}, time:{run_time:0.1f}")

# 7. serialization
with open(f'pkg_level2_bank{bank2_size}.model', 'wb') as f:
    pickle.dump(model_l2, f)

with open('bank2.out', 'wb') as f:
    bank = model_l2.get_bank()
    bank = bank.cpu().detach().numpy()
    pickle.dump(bank, f)
