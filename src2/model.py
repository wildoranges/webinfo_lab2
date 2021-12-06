import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import random
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader, dataset
from data_process import get_word_vec, get_h_r_vec, load_and_process_data, remove_no_description_token
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
            nn.Tanh()
        )
    
    def forward(self, vec_h, vec_r):
        input_vec = torch.hstack((vec_h, vec_r))
        return F.normalize(self.model(input_vec))

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
            if i % 1000 == 0:
                print('Train Epoch: {}/{} [{}/{}]\tLoss: {:.6f}'.format(
                    epoch, n_epochs, i * len(h), len(train_dataloader.dataset), loss.item()))

    torch.save(model, "../output/model.pkl")

def test(test_dataloader, device, criterion):
    model = torch.load("../output/model.pkl")
    model.eval()
    total_loss = 0

    for i, (h, r, t) in enumerate(test_dataloader):
        h = h.to(device)
        r = r.to(device)
        t = t.to(device)
        loss = criterion(model(h, r), t)
        total_loss += loss
        if i % 10 == 0:
            print('Test Batch: [{}/{}]\t Loss: {}'.format(
                i * len(h), len(test_dataloader.dataset), loss.item()))

    print('Loss : {}\nAverage loss: {}'.format(total_loss, total_loss / len(test_dataloader.dataset)))
    
def process_test(test_data_path:str):
    f = open(test_data_path, "r")
    lines = f.readlines()
    test_data = []
    for line in lines:
        test_data.append(line.strip().split("\t")[:2])
    f.close()
    return test_data
    
def predict(e_dict, r_dict, test_dataloader, top=5, output_path="../output/result.txt", model_path="../output/model.pkl"):
    f = open(output_path, "w+")
    model = torch.load(model_path)
    model.eval()
    index = {}
    total_entity = set(e_dict.keys())
    total_relation = set(r_dict.keys())
    
    i = 0
    entity_matrix = None
    for k, v in e_dict.items():
        v = v.reshape((-1, 1))
        if i == 0:
            entity_matrix = v
        else:
            entity_matrix = np.hstack((entity_matrix, v))
        index[i] = k
        i += 1 
    
    
    for i, (h, r) in tqdm(enumerate(test_dataloader)):
        if (h not in total_entity) or (r not in total_relation):
            random_t_index = random.choices(range(len(index)), k=5)
            line = [str(index[i]) for i in random_t_index]
            line = "\t".join(line)
            f.write(line+"\n")
        else:
            h_vec = e_dict[h]
            r_vec = r_dict[r]
            output_vec = model(torch.Tensor(h_vec).reshape((-1, )).to(device), torch.Tensor(r_vec).reshape((-1, )).to(device))
            output_vec = output_vec.numpy().reshape((-1, 1))
            distance = np.sum((entity_matrix - output_vec)**2, axis=0)
            top_entity_index = list(distance.argsort()[:top])
            line = [str(index[i]) for i in top_entity_index]
            line = "\t".join(line)
            f.write(line+"\n")
            
    f.close()
            
        

if __name__ == '__main__':
    #word2vec_model = get_word_vec()
    word2vec_model = word2vec.Word2Vec.load("../output/word2vec.model")
    e_dict, r_dict = get_h_r_vec(word2vec_model)

    # ignore entities and relations without description in train set
    # remove_no_description_token(e_dict, r_dict, '../dataset/train.txt', '../dataset/train_process.txt')

    # ignore entities and relations without description in test set
    # remove_no_description_token(e_dict, r_dict, '../dataset/dev.txt', '../dataset/dev_process.txt')

    
    train_feature, train_label, test_feature, test_label = \
        load_and_process_data('../dataset/train_process.txt',\
             '../dataset/dev_process.txt')

    train_Dataset = TripletDataset(train_feature, train_label)
    test_Dataset = TripletDataset(test_feature, test_label)
    train_DataLoader = DataLoader(train_Dataset, batch_size=64)
    test_Dataloader = DataLoader(test_Dataset, batch_size=64)

    model = Word2Vec(vector_size=100).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
    optimizer2 = torch.optim.SGD(model.parameters(), lr=0.01)
    #train(model=model, train_dataloader=train_DataLoader, device=device, optimizer=optimizer, n_epochs=10, criterion=criterion)
    test(test_dataloader=test_Dataloader, device=device, criterion=criterion)
    #test_data = process_test("/run/media/cjb/Win/private/learning/new_private/webinfo/lab2/dataset/test.txt")
    #predict(e_dict, r_dict, test_data)