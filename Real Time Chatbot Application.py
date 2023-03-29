# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 19:48:18 2023

@author: SAYAM KUMAR
"""

import random, json, pickle, nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

model = load_model('chatbot_model.h5')

def clean_text(sentence):
    tokenized_words = nltk.word_tokenize(sentence)
    tokenized_words = [lemmatizer.lemmatize(word) for word in tokenized_words]
    return tokenized_words

def bag_of_words(sentence):
    sentence_words = clean_text(sentence)
    bow = [0]*len(words)
    
    for word in sentence_words:
        for i, w in enumerate(words):
            if w == word:
                bow[i] = 1
            
    return np.array(bow)

def predict_classes(sentence):
    bow = bag_of_words(sentence) # Create bag of words
    results = model.predict(np.array([bow]))[0] # Predict the results using the pretrained model
    ERROR_THRESHOLD = 0.25 # To reduce uncertainty and obtain accurate results
    results = [[idx,res] for idx,res in enumerate(results) if res > ERROR_THRESHOLD] # Fetching index of output class and the resultant probability by enumerating the results
    results.sort(key=lambda x: x[1],reverse=True) # Sort by probability in reverse order
    return_list = []
    
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    
    return return_list

def generate_response(intents_list,intents_json):
    tag = intents_list[0]['intent']
    list_intents = intents_json['intents']
    
    for x in list_intents:
        if x['tag'] == tag:
            result = random.choice(x['responses'])
            break
    
    return result
print("Welcome! Bot is running!")

while True:
    message = input("")
    pred = predict_classes(message)
    res = generate_response(pred,intents)
    print(res)
    