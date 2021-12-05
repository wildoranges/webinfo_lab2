from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T
from data_preprocess import get_max_seqlen, load_and_process_data, load_text

n_epochs = 100 
n_entities = 14541 
n_relations = 237
batch_size = 16
n_hidden = 100
max_seqlen = 200

device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

class Model(nn.Module):
    # to be modified
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.n_entities = n_entities
        self.n_relations = n_relations
        # use RNN to calculate semantic similarity
        self.encoder = nn.RNN(input_size=n_class, hidden_size=n_hidden)
        self.decoder = nn.RNN(input_size=n_class, hidden_size=n_hidden)
        self.fc = nn.Linear(n_hidden, n_class)

    def forward(self, enc_input_h, enc_input_r, enc_hidden, dec_input):
        # to be modified
        enc_input_h = enc_input_h.transpose(0, 1)
        enc_input_r = enc_input_r.transpose(0, 1)


        dec_input = dec_input.transpose(0, 1)
        #enc_hidden : [1, batch_size, hidden_size]
        _, enc_states = self.encoder(enc_input_h, enc_hidden)
        
        _, enc_states = self.encoder(enc_input_r, enc_states)

        # outputs: [maxlen + 1, batch_size, hidden_size]
        outputs, _ = self.decoder(dec_input, enc_states)
        outputs = self.fc(outputs)
        
        return outputs


class TripletDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        h = self.x[index][0]
        r = self.x[index][1]
        t = self.y[index][0]

        words_h = [word_dict[n] for n in entities_dict[h].split()]
        words_h += [word_dict['Pad']] * (max_seqlen - len(words_h))

        words_r = [word_dict[n] for n in relations_dict[r].split()]
        words_r += [word_dict['Pad']] * (max_seqlen - len(words_r))

        words_t = [word_dict[n] for n in entities_dict[t].split()]
        words_t += [word_dict['Pad']] * (max_seqlen - len(words_t))

        head_seq = np.eye(n_class)[words_h]
        rela_seq = np.eye(n_class)[words_r]
        #c = np.eye(n_class)[words_t]
        tail_seq = np.eye(n_class)[[word_dict['Start']] + words_t]


        return torch.FloatTensor(head_seq), torch.FloatTensor(rela_seq), torch.FloatTensor(tail_seq), \
            torch.FloatTensor(tail_seq + [word_dict['End']])



def train(model, train_dataloader, device, optimizer, criterion, n_epochs):
    model.train()
    for epoch in tqdm(range(n_epochs)):

        for i, (h, r, output_t, target_t) in enumerate(train_dataloader):
            optimizer.zero_grad()            
            hidden_state = torch.zeros(1, batch_size, n_hidden).to(device)
            h = h.to(device)
            r = r.to(device)
            output_t = output_t.to(device)
            target_t = target_t.to(device)
            output = model(h, r, hidden_state, output_t)
            output = output.transpose(0, 1)
            loss = criterion(output, target_t)

            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                    print('Train Epoch: {}/{} [{}/{}]\tLoss: {:.6f}'.format(
                        epoch, n_epochs, i * len(h), len(train_dataloader.dataset), loss.item()))

    torch.save(model, "../output/Model.pkl")


def test(model, test_dataloader, device, criterion):
    # TODO
    model.eval()
    num_correct = 0
    with torch.no_grad():
        for i, (feature, label) in enumerate(test_dataloader):
            label = label.to(device)
            feature[:,1] += n_entities            
            feature = feature.to(device)
    
    raise NotImplementedError


if __name__ == '__main__':
    train_feature, train_label, test_feature, test_label = \
        load_and_process_data('/data/wentaozhang/zwt/webinfo_lab2/dataset/train.txt',\
             '/data/wentaozhang/zwt/webinfo_lab2/dataset/dev.txt')
    
    entities_dict, relations_dict, word_dict, index_dict \
        = load_text('/data/wentaozhang/zwt/webinfo_lab2/dataset/entity_with_text.txt', \
            '/data/wentaozhang/zwt/webinfo_lab2/dataset/relation_with_text.txt')    
    # entities_dict : key: entity id . value: entity text
    # relations_dict : key: relation id. value: relation text


    word_dict['Pad'] = len(word_dict)
    word_dict['Start'] = len(word_dict)
    word_dict['End'] = len(word_dict)

    max_key, max_seqlen = get_max_seqlen(entities_dict, relations_dict)
    
    train_Dataset = TripletDataset(train_feature, train_label)
    test_Dataset = TripletDataset(test_feature, test_label)
    train_DataLoader = DataLoader(train_Dataset, batch_size=batch_size)
    test_Dataset = DataLoader(test_Dataset, batch_size=batch_size)

    n_class = len(word_dict)  # vocab list


    model = Model().to(device)
    #model = Model()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    train(model=model, train_dataloader=train_DataLoader, device=device, optimizer=optimizer, criterion=criterion, n_epochs=100)
