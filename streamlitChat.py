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




import streamlit as st

st.title("RNVR HealthBot")

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all the history of messages

# messages are displayed as:
# messages : [{role: "user/assistant", content: "content to be written"}, .... ]


# Greet with a message and store in the session state
# with st.chat_message("assistant"):
#     st.markdown("Hello User! I am here to clarify your queries regarding first aid!")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# st.session_state.messages.append({'role':"assistant","content":"Hello User! I am here to clarify your queries regarding first aid!",})


# User input 
prompt = st.chat_input('Enter your query here!')

if prompt is not None:

    # Display user's message
    with st.chat_message('user'):
        st.markdown(prompt)

    # Add the prompt to the session state
    st.session_state.messages.append({'role':'user','content':prompt})

    # Tokenize and preprocess prompt
    sentence = prompt
    
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1,X.shape[0])
    X = torch.from_numpy(X)

    # Pass the processed input and process the output 
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probabilities = torch.softmax(output, dim=1)
    prob = probabilities[0][predicted.item()]

    # Generate appropriate response
    response = ''
    if prob.item()>0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                response = intent['responses']
    else:
        response = "I'm sorry, I'm not experienced enough to help you with that case... :( "
    
    with st.chat_message('assistant'):
        st.markdown(response[0])
    
    st.session_state.messages.append({'role':'assistant','content':response[0]})

    

    