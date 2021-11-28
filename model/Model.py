import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import optimizer
from torch.utils.data import Dataset, DataLoader
from data_preprocess import load_and_process_data

n_epochs = 100 
n_entities = 14540 
n_relations = 237
device = torch.device('cuda:0')

class Model(nn.Module):

    def __init__(self, n_tokens, embedding_dim, hidden_dim) -> None:
        super(Model, self).__init__()
        self.n_tokens = n_tokens
        self.embedding_dim = embedding_dim
        self.embedding: nn.Embedding = nn.Embedding(num_embeddings=n_tokens, embedding_dim=embedding_dim)
        # nn.Embedding: 输入index， 输出该index的embedding. shape = n_tokens
        self.linear1 = nn.Linear(2 * embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, n_tokens)


    def forward(self, input_indices):
        input_vecs = self.embedding(input_indices) # len(input_indices) = 2 , shape = (batchsize, 2, embedding_dim)
        input_flatten = input_vecs.view(-1, 1, 2 * self.embedding_dim) # shape: (batchsize, 1, 2 * embedding_dim), flatten
        out = self.linear1(input_flatten)
        out = F.relu(out) 
        out = self.linear2(out)
        probs = F.softmax(out, dim=-1) # probs: 1 * n_tokens
        return probs


class TripletDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def train(model, train_dataloader, device, optimizer, criterion, n_epochs):
    model.train()
    for epoch in range(n_epochs):
        for i, (feature, label) in enumerate(train_dataloader):
            label = label.to(device)
            feature[:,1] += n_entities            
            feature = feature.to(device)

            optimizer.zero_grad()
            output = model(feature)
            label = label.to(torch.int64)
            label = F.one_hot(label, model.n_tokens).to(torch.float64)
            output = torch.flatten(output, start_dim=-2)
            label = torch.flatten(label, start_dim=-2)

            loss = criterion(output, label)

            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                    print('Train Epoch: {}/{} [{}/{}]\tLoss: {:.6f}'.format(
                        epoch, n_epochs, i * len(feature), len(train_dataloader.dataset), loss.item()))

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
    train_feature, train_label, test_feature, test_label = load_and_process_data('../dataset/train.txt', '../dataset/dev.txt')
    train_Dataset = TripletDataset(train_feature, train_label)
    test_Dataset = TripletDataset(test_feature, test_label)
    train_DataLoader = DataLoader(train_Dataset, batch_size=64)
    test_Dataset = DataLoader(test_Dataset, batch_size=64)
    model = Model(n_tokens=n_entities + n_relations, embedding_dim = 64, hidden_dim=32).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

    train(model=model, train_dataloader=train_DataLoader, device=device, optimizer=optimizer, criterion=criterion, n_epochs=100)
