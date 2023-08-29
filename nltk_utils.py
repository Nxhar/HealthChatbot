import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

# nltk.download('punkt')   # Run for the first time only. To use the word_tokenizer

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):

    tokenized_sentence = [stem(word.lower()) for word in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    
    return bag



# Testing the Tokenizer
# a = 'hello world, i am a godlike being of no premising whatsoever'
# print(a) 
# a = tokenize(a)
# print(a)
# a = [stem(w) for w in a]
# print(a)