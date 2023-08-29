import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import FwdNeuralNet

with open('intents.json','r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = [] # data and labels

ignore_words = ['?','.',',','!',':',';','-']

for intent in intents['intents']:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent['patterns']:
        word = tokenize(pattern)
        all_words.extend(word)
        xy.append((word, tag))

all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
# print(all_words)
# print(tags)


X_train = []
y_train = [] 

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    
    label = tags.index(tag)
    y_train.append(label) # CrossEntropyEncoding

X_train = np.array(X_train)
y_train = np.array(y_train)


class ChatDataset(Dataset):
    def __init__(self) :
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index) :
        return self.x_data[index], self.y_data[index]

    def __len__(self) :
        return self.n_samples
    
# Hyperparamaters
batch_size = 8
hidden_size = 10
input_size = len(all_words)
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000


# Creating dataset and loading data

dataset = ChatDataset()

train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


# Setting device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model Creation

model = FwdNeuralNet(input_size, hidden_size, output_size).to(device=device)


#Loss and Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training the model

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)
        words = torch.tensor(words, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        #1. Pass inputs to forward function
        outputs = model(words)

        #2. Calculate Loss
        loss = loss_fn(outputs, labels)

        #3. Zero Grad
        optimizer.zero_grad()

        #4. Backpropogation
        loss.backward()

        #5. Step to the next epoch
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1} / {num_epochs}, loss = {loss.item() : .4f}')

print(f'Final loss : {loss.item() : .4f}')

# Save model

data = {
    "model_state" : model.state_dict(),
    "input_size" : input_size,
    "hidden_size" : hidden_size,
    "output_size" : output_size,
    "all_words" : all_words,
    "tags" : tags,
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. File saved to {FILE}')
