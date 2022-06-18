import random
import json
import torch
from model import MyModel
from nltk_utils import bag_of_words, tokenize
device = torch.device('cpu')
FILE = "Chatbot.pth"
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]
model = MyModel(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

def neironka(phrase):
    phrase = tokenize(phrase)
    X = bag_of_words(phrase, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                r = random.choice(intent['responses'])
                return r
    else:
        return "Не поняла!"