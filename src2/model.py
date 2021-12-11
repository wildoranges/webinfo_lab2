import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import random
from torch.nn.modules.activation import Sigmoid
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader, dataset
from data_process import get_ent_rel2id, get_word_vec, get_h_r_vec, load_and_process_data, remove_no_description_token
from gensim.models import word2vec

device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

class MyDataset(Dataset):
    
    def __init__(self, x, y, h2t):
        self.x = x
        self.y = y
        self.h2t = h2t
        self.pos_label = 0
        self.neg_label = 1
        self.negx = np.array(self.x)
        self.negy = np.array(self.y)
        for i in range(self.x.shape[0]):
            head = self.x[i][0]
            while True:
                j = random.randint(0, self.x.shape[0]-1)
                rand_tail = self.y[j][0]
                try:
                    if rand_tail not in self.h2t[head]:
                        self.negy[i][0] = rand_tail
                        break
                except KeyError:
                    self.negy[i][0] = rand_tail
                    break
        # self.pos_label = torch.LongTensor([0])
        # self.neg_label = torch.LongTensor([1])

    def __len__(self):
        return 2*self.x.shape[0]
    
    
    def __getitem__(self, index):
        if index % 2 == 0:
            return e_dict[self.x[index//2][0]], r_dict[self.x[index//2][1]], e_dict[self.y[index//2][0]], self.pos_label
        else:
            return e_dict[self.negx[index//2][0]], r_dict[self.negx[index//2][1]], e_dict[self.negy[index//2][0]], self.neg_label

    # def __getitem__(self, index):
    #     if index % 2 == 0:
    #         return e_dict[self.x[index//2][0]], r_dict[self.x[index//2][1]], e_dict[self.y[index//2][0]], self.pos_label
    #     else:
    #         head = self.x[index//2][0]
    #         while True:
    #             i = random.randint(0, self.x.shape[0]-1)
    #             rand_tail =  self.y[i][0]
    #             try:
    #                 if rand_tail not in self.h2t[head]:
    #                     return e_dict[head], r_dict[self.x[index//2][1]], e_dict[rand_tail], self.neg_label
    #             except KeyError:
    #                 return e_dict[head], r_dict[self.x[index//2][1]], e_dict[rand_tail], self.neg_label
                
class PosDataset(Dataset):
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.pos_label = 0
        # self.pos_label = torch.LongTensor([0])
        # self.neg_label = torch.LongTensor([1])

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return e_dict[self.x[index][0]], r_dict[self.x[index][1]], e_dict[self.y[index][0]], self.pos_label
        

class Vec2Tail(nn.Module):

    def __init__(self, vector_size, hidden_dim=128):
        super(Vec2Tail, self).__init__()

        self.vector_size = vector_size
        self.entity_num = len(e_dict)
        self.relation_num = len(r_dict)

        self.ent_embedding = nn.Embedding(self.entity_num, vector_size)
        self.rel_embedding = nn.Embedding(self.relation_num, vector_size)
        
        self.model = nn.Sequential(
            nn.Linear(3*vector_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        
        self.__init_embedding()

    def forward(self, vec_h, vec_r, vec_t):
        input_vec = torch.hstack((vec_h, vec_r))
        input_vec = torch.hstack((input_vec, vec_t))
        return self.model(input_vec)

    def __init_embedding(self):
        for key in e_dict.keys():
            self.ent_embedding.weight.data[ent2id[key]] = torch.from_numpy(e_dict[key])
        
        for key in r_dict.keys():
            self.rel_embedding.weight.data[rel2id[key]] = torch.from_numpy(r_dict[key])



def train(model, train_dataloader, device, optimizer, n_epochs, criterion):
    model.train()
    for epoch in range(n_epochs):
        for i, (h, r, t, label) in enumerate(train_dataloader):
            optimizer.zero_grad()
            h = h.to(device)
            r = r.to(device)
            t = t.to(device)
            label = label.to(device)
            #vec_h = e_dict[h].to(device)
            #vec_r = r_dict[r].to(device)
            #vec_t = e_dict[t].to(device)
            output = model(h, r, t)
            loss = criterion(output, label)
            
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
    total_entity = set(e_dict.keys())
    total_relation = set(r_dict.keys())
    
    index = list(e_dict.keys())
    
    for h, r in tqdm(test_dataloader):
        if (h not in total_entity) or (r not in total_relation):
            random_t_index = random.choices(range(len(index)), k=top)
            line = [str(index[i]) for i in random_t_index]
            line = "\t".join(line)
            f.write(line+"\n")
        else:
            h_vec = e_dict[h]
            r_vec = r_dict[r]
            result = []
            for tail in index:
                t_vec = e_dict[tail]
                output_vec = model(torch.Tensor(h_vec).reshape((1, -1)).to(device), torch.Tensor(r_vec).reshape((1, -1)).to(device), 
                                   torch.Tensor(t_vec).reshape((1, -1)).to(device))
                output_vec = output_vec.detach().cpu().reshape((-1, 1))
                output_vec = F.softmax(output_vec, dim=0).numpy()
                result.append(float(output_vec[1][0]))
            all_prob = np.array(result)
            top_entity_index = list(all_prob.argsort()[:top])
            line = [str(index[i]) for i in top_entity_index]
            line = "\t".join(line)
            f.write(line+"\n")
            
    f.close()
    
    
def get_predict_accuracy(predict_path:str, target_path:str):
    fp = open(predict_path, "r")
    fd = open(target_path, "r")
    fp_lines = fp.readlines()
    fd_lines = fd.readlines()
    total = len(fd_lines)
    if len(fp_lines) != total:
        print("lines not match")
        exit(1)
        
    acc = 0
    for i in range(total):
        fd_line = fd_lines[i]
        fp_line = fp_lines[i]
        target = fd_line.strip().split('\t')[-1]
        predict_res = fp_line.strip().split('\t')
        if target in predict_res:
            acc += 1
            
    accuracy = acc / total
    print("accuracy : {}".format(accuracy))
    return accuracy
            
def get_all_tail_for_head(filepath:str):
    f = open(filepath, "r")
    all_tails = {}
    for line in f.readlines():
        h, _, t = line.strip().split('\t')
        try:
            all_tails[h].add(t)
        except KeyError:
            all_tails[h] = set()
            all_tails[h].add(t)
    f.close()
    return all_tails

if __name__ == '__main__':
    #word2vec_model = get_word_vec()
    word2vec_model = word2vec.Word2Vec.load("../output/word2vec.model")
    e_dict, r_dict = get_h_r_vec(word2vec_model, norm=True)
    ent2id, rel2id = get_ent_rel2id(e_dict, r_dict)

    # ignore entities and relations without description in train set
    # remove_no_description_token(e_dict, r_dict, '../dataset/train.txt', '../dataset/train_process.txt')

    # ignore entities and relations without description in test set
    # remove_no_description_token(e_dict, r_dict, '../dataset/dev.txt', '../dataset/dev_process.txt')

    train_feature, train_label, test_feature, test_label = \
        load_and_process_data('../dataset/train_process.txt',\
             '../dataset/dev_process.txt')

    all_tails = get_all_tail_for_head('../dataset/train_process.txt')
    
    # train_Dataset = MyDataset(train_feature, train_label, all_tails)
    # train_Dataset = PosDataset(train_feature, train_label)
    # test_Dataset = MyDataset(test_feature, test_label, all_tails)
    # train_DataLoader = DataLoader(train_Dataset, batch_size=64)
    # test_Dataloader = DataLoader(test_Dataset, batch_size=64)

    model = Vec2Tail(vector_size=100).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
    optimizer2 = torch.optim.SGD(model.parameters(), lr=0.01)
    train(model=model, train_dataloader=train_DataLoader, device=device, optimizer=optimizer, n_epochs=10, criterion=criterion)
    # test(test_dataloader=test_Dataloader, device=device, criterion=criterion)
    test_data = process_test("../dataset/dev.txt")
    predict(e_dict, r_dict, test_data, output_path="../output/dev_result.txt")
    # get_predict_accuracy("../output/dev_result.txt", "../dataset/dev.txt")
