import json 
import numpy as np
import torch
import torch.nn as nn
from nltk_utils import tokenize, stem, bag_of_words
from torch.utils.data import Dataset, DataLoader
from model import MyModel
with open('intents.json', 'r') as f:
	intents = json.load(f)

tags = []
all_words = []
xy = []

for intent in intents['intents']:
	tag = intent['tag']
	tags.append(tag)
	for pattern in intent['patterns']:
		w = tokenize(pattern)
		all_words.extend(w)
		xy.append((w, tag))

ignore = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
Y_train = []
for (pattern_sentece, tag) in xy:
	bag = bag_of_words(pattern_sentece, all_words)
	X_train.append(bag)
	label = tags.index(tag)
	Y_train.append(label)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

class ChatDataset(Dataset):
	def __init__(self):
		self.n_samples = len(X_train)
		self.x_data = X_train
		self.y_data = Y_train

	def __getitem__(self, index):
		return self.x_data[index], self.y_data[index]

	def __len__(self):
		return self.n_samples

batch_size = 8
hidden_size = 8 
output_size = len(tags)
input_size = len(X_train[0])
num_epochs = 1000
# print(input_size, len(all_words))
# print(output_size, tags)
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)
device = torch.device('cpu')
model = MyModel(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
	for words, labels in train_loader:
		words = words.to(device)
		labels = labels.to(device)
		outputs = model(words)
		loss = criterion(outputs, labels)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()



data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "Chatbot.pth"
torch.save(data, FILE)

print(f'Модель создана {FILE}')