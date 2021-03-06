import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import random
from torch.nn.modules.activation import Sigmoid
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, dataset
from data_process import get_ent_rel2id, get_word_vec, get_h_r_vec, load_and_process_data, remove_no_description_token
from gensim.models import word2vec

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class MyDataset(Dataset):
    
    def __init__(self, x, y, h2t):
        self.x = x
        self.y = y
        self.h2t = h2t
        self.t2h = {}
        for _, rt in h2t.items():
            for _, ts in rt.items():
                for t in ts:
                    self.t2h[t] = {}
                
        for h, rt in h2t.items():
            for r, ts in rt.items():
                for t in ts:
                    try:
                        self.t2h[t][r].add(h)
                    except KeyError:
                        self.t2h[t][r] = set()
                        self.t2h[t][r].add(h)
        self.pos_label = 0
        self.neg_label = 1
        self.negx = np.array(self.x)
        self.negy = np.array(self.y)
        self.gen_neg_samples()
        
    def gen_neg_samples(self):
        for i in range(self.x.shape[0]):
            head = self.x[i][0]
            rel = self.x[i][1]
            tail = self.y[i][0]
            head_or_tail = random.randint(0, 1)
            if head_or_tail > 0: 
                #random tail
                while True:
                    j = random.randint(0, self.x.shape[0]-1)
                    rand_tail = self.y[j][0]
                    try:
                        if rand_tail not in self.h2t[head][rel]:
                            self.negy[i][0] = rand_tail
                            break
                    except KeyError:
                        self.negy[i][0] = rand_tail
                        break
            else:
                #rand head
                while True:
                    j = random.randint(0, self.x.shape[0]-1)
                    rand_head = self.x[j][0]
                    try:
                        if rand_head not in self.t2h[tail][rel]:
                            self.negx[i][0] = rand_head
                            break
                    except KeyError:
                        self.negx[i][0] = rand_head
                        break
                    

    def __len__(self):
        return self.x.shape[0]
    
    
    def __getitem__(self, index):

        return ent2id[self.x[index][0]], rel2id[self.x[index][1]], ent2id[self.y[index][0]], \
            ent2id[self.negx[index][0]], rel2id[self.negx[index][1]], ent2id[self.negy[index][0]]
        

class Vec2Tail(nn.Module):

    def __init__(self, vector_size, hidden_dim=128):
        super(Vec2Tail, self).__init__()

        self.vector_size = vector_size
        self.entity_num = len(e_dict)
        self.relation_num = len(r_dict)

        self.ent_embedding = nn.Embedding(self.entity_num, vector_size)
        self.rel_hyperplane_embedding = nn.Embedding(self.relation_num, vector_size)
        self.rel_norm_embedding = nn.Embedding(self.relation_num, vector_size)

        self.__init_embedding()

    def forward(self, h, r, t):
        # model() -> d(h+r, t)
        vec_h = self.ent_embedding(h)
        d_r = self.rel_hyperplane_embedding(r)
        norm_vec = self.rel_norm_embedding(r)
        vec_t = self.ent_embedding(t)
        vec_h = self.project(vec_h, norm_vec)
        vec_t = self.project(vec_t, norm_vec)
        return torch.sqrt(torch.sum((vec_h + d_r - vec_t) ** 2, dim=1))
        
    def __init_embedding(self):
        for key in e_dict.keys():
            self.ent_embedding.weight.data[ent2id[key]] = torch.from_numpy(e_dict[key])
        
        for key in r_dict.keys():
            self.rel_hyperplane_embedding.weight.data[rel2id[key]] = torch.from_numpy(r_dict[key])

        norm = self.ent_embedding.weight.detach().cpu().numpy()
        norm = norm / np.sqrt(np.sum(np.square(norm), axis=1, keepdims=True))
        self.ent_embedding.weight.data.copy_(torch.from_numpy(norm))

        norm = self.rel_hyperplane_embedding.weight.detach().cpu().numpy()
        norm = norm / np.sqrt(np.sum(np.square(norm), axis=1, keepdims=True))
        self.rel_hyperplane_embedding.weight.data.copy_(torch.from_numpy(norm))
        nn.init.xavier_uniform_(self.rel_norm_embedding.weight.data)
        
    def project(self, vec, norm_vec):
        norm_vec = F.normalize(norm_vec, p=2, dim=-1)
        
        return vec - (torch.sum(vec*norm_vec, dim=-1).reshape(-1, 1) * norm_vec)
        
    def predict(self, test_dataloader, top=5, seperator='\t', output_path="../output/result.txt"):
        self.eval()
        f = open(output_path, "w+")
        total_entity_set = set(ent2id.keys())
        total_relation_set = set(rel2id.keys())
        
        index = list(ent2id.keys())
        
        total_entity_id = torch.LongTensor([ent2id[ent] for ent in index]).to(device)
        
        for h, r in tqdm(test_dataloader):
            if (h not in total_entity_set) or (r not in total_relation_set):
                random_t_index = random.choices(range(len(index)), k=top)
                line = [str(index[i]) for i in random_t_index]
                line = seperator.join(line)
                f.write(line+"\n")
            else:
                h_id = torch.LongTensor([ent2id[h]]).to(device)
                r_id = torch.LongTensor([rel2id[r]]).to(device)
                norm_vec = self.rel_norm_embedding(r_id)
                vec_h = self.ent_embedding(h_id)
                d_r = self.rel_hyperplane_embedding(r_id)
                target = self.project(vec_h, norm_vec) + d_r
                target = target.reshape((1, -1))
                t_matrix = self.ent_embedding(total_entity_id)
                t_matrix = self.project(t_matrix, norm_vec)
                distances = torch.sum((t_matrix - target)**2, dim=1).argsort()[:top]
                line = [str(index[int(i)]) for i in distances]
                line = seperator.join(line)
                f.write(line+"\n")
        f.close()

def train(model, train_dataloader, device, optimizer, n_epochs, criterion):
    model.train()
    n_batches = len(train_dataloader.dataset) //  train_dataloader.batch_size + 1
    loss_list = []
    for epoch in range(n_epochs):
        total_loss = 0.0
        for i, (h, r, t, h_neg, r_neg, t_neg) in enumerate(train_dataloader):
            optimizer.zero_grad()
            h, r, t = h.to(device), r.to(device), t.to(device)
            h_neg, r_neg, t_neg = h_neg.to(device), r_neg.to(device), t_neg.to(device)
            out_pos = model(h, r, t)
            out_neg = model(h_neg, r_neg, t_neg)

            y = -torch.ones(h.shape[0]).to(device)

            loss = criterion(out_pos, out_neg, y)
            total_loss += loss
            loss.backward()
            optimizer.step()
        
        mean_train_loss = total_loss / n_batches
        loss_list.append(mean_train_loss.detach().cpu().numpy())
        print("epoch {}, loss = {}".format(epoch, mean_train_loss))
    
    figure = plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(loss_list)
    plt.title('Training Loss')
    # plt.show()
    figure.savefig('loss.png')

    torch.save(model, "../output/model.pkl")
    
    
def process_test(test_data_path:str):
    f = open(test_data_path, "r")
    lines = f.readlines()
    test_data = []
    for line in lines:
        test_data.append(line.strip().split("\t")[:2])
    f.close()
    return test_data
    
    
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
    lines = f.readlines()
    for line in lines:
        h, _, _ = line.strip().split('\t')
        all_tails[h] = {}
    for line in lines:
        h, r, t = line.strip().split('\t')
        try:
            all_tails[h][r].add(t)
        except KeyError:
            all_tails[h][r] = set()
            all_tails[h][r].add(t)
        
    f.close()
    return all_tails

if __name__ == '__main__':
    #word2vec_model = get_word_vec()
    word2vec_model = word2vec.Word2Vec.load("../output/word2vec.model")
    e_dict, r_dict = get_h_r_vec(word2vec_model, norm=False)
    ent2id, rel2id = get_ent_rel2id(e_dict, r_dict)

    # ignore entities and relations without description in train set
    # remove_no_description_token(e_dict, r_dict, '../dataset/train.txt', '../dataset/train_process.txt')

    # ignore entities and relations without description in test set
    # remove_no_description_token(e_dict, r_dict, '../dataset/dev.txt', '../dataset/dev_process.txt')

    train_feature, train_label, test_feature, test_label = \
        load_and_process_data('../dataset/train_process.txt',\
             '../dataset/dev_process.txt')

    all_tails = get_all_tail_for_head('../dataset/train_process.txt')
    
    train_Dataset = MyDataset(train_feature, train_label, all_tails)
    train_DataLoader = DataLoader(train_Dataset, batch_size=1000, shuffle=True)

    model = Vec2Tail(vector_size=100).to(device)
    #model = torch.load("../output/model.pkl")

    criterion = nn.MarginRankingLoss(margin=4.0, reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train(model=model, train_dataloader=train_DataLoader, device=device, optimizer=optimizer, n_epochs=25, criterion=criterion)
    test_data = process_test("../dataset/dev.txt")
    model.predict(test_data, output_path="../output/dev_result.txt")
    get_predict_accuracy("../output/dev_result.txt", "../dataset/dev.txt")
    pre_data = process_test("../dataset/test.txt")
    model.predict(pre_data, seperator=',', output_path="../output/test_result.txt")
