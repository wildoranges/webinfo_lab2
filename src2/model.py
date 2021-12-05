import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from data_process import get_word_vec, get_h_r_vec, load_and_process_data
from gensim.models import word2vec

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class TripletDataset(Dataset):
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return e_dict[self.x[index][0]], r_dict[self.x[index][1]], e_dict[self.y[index][0]]

class Word2Vec(nn.Module):

    def __init__(self, vector_size):
        super(Word2Vec, self).__init__()
        self.vector_size = vector_size
        self.model = nn.Sequential(
            nn.Linear(2*vector_size, vector_size),
            nn.ReLU()
        )
    
    def forward(self, vec_h, vec_r):
        input_vec = torch.hstack((vec_h, vec_r))
        return self.model(input_vec)

def train(model, train_dataloader, device, optimizer, n_epochs, criterion):
    model.train()
    for epoch in range(n_epochs):
        for i, (h, r, t) in enumerate(train_dataloader):
            optimizer.zero_grad()
            h = h.to(device)
            r = r.to(device)
            t = t.to(device)
            #vec_h = e_dict[h].to(device)
            #vec_r = r_dict[r].to(device)
            #vec_t = e_dict[t].to(device)
            output = model(h, r)
            loss = criterion(output, t)
            
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print('Train Epoch: {}/{} [{}/{}]\tLoss: {:.6f}'.format(
                    epoch, n_epochs, i * len(h), len(train_dataloader.dataset), loss.item()))

    torch.save(model, "../output/Model.pkl")



if __name__ == '__main__':
    word2vec_model = word2vec.Word2Vec.load("../output/test")
    e_dict, r_dict = get_h_r_vec(word2vec_model)
    
    # ft = open('../dataset/train.txt', 'r')
    # fw = open('../dataset/train_process.txt', 'w+')
    # for line in ft.readlines():
    #     h, r, t = line.strip().split('\t')
    #     if (h not in e_dict.keys()) or (r not in r_dict.keys()) or (t not in e_dict.keys()):
    #         continue
    #     fw.write(line)
    # ft.close()
    # fw.close()
    
    train_feature, train_label, test_feature, test_label = \
        load_and_process_data('../dataset/train_process.txt',\
             '../dataset/dev.txt')

    train_Dataset = TripletDataset(train_feature, train_label)
    test_Dataset = TripletDataset(test_feature, test_label)
    train_DataLoader = DataLoader(train_Dataset, batch_size=64)
    test_Dataset = DataLoader(test_Dataset, batch_size=64)

    model = Word2Vec(vector_size=100).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    optimizer2 = torch.optim.SGD(model.parameters(), lr=0.001)
    train(model=model, train_dataloader=train_DataLoader, device=device, optimizer=optimizer2, n_epochs=100, criterion=criterion)
