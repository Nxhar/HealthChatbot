import json
import torch
from model import FwdNeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as F:
    intents = json.load(F)

FILE = 'data.pth'
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = FwdNeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)

model.eval()

bot_name = 'RNVR'
print('RNVR : Hello User! I am here to clarify your queries regarding first aid!')
print('RNVR : Type "quit" to quit the chatbot.')

while True:
    sentence = input('You  : ')
    if sentence.lower() == 'quit':
        break
    
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1,X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probabilities = torch.softmax(output, dim=1)
    prob = probabilities[0][predicted.item()]

    

    response = ''
    if prob.item()>0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                response = intent['responses']
    else:
        response = "I'm sorry, I'm not experienced enough to help you with that case... :( "
    
    print(f'RNVR : {response[0]}')

print('Thank you for utilizing our services. We hope we were of help to you!')


