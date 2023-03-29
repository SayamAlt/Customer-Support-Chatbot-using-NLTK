# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import random, nltk, json, pickle
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop


lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['!','.',',','?','#',';']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        list_words = word_tokenize(pattern)
        words.extend(list_words)
        documents.append((list_words,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

train_data = []
output_zeros = [0]*len(classes)

for document in documents:
    bow = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    
    for word in words:
        bow.append(1) if word in word_patterns else bow.append(0)
    
    output = list(output_zeros)
    output[classes.index(document[1])] = 1
    train_data.append([bow,output])

random.shuffle(train_data)

train_data = np.array(train_data)

X_train = list(train_data[:,0])
y_train = list(train_data[:,1])

model = Sequential()
model.add(Dense(units=128,activation='relu',input_shape=(len(X_train[0]),)))
model.add(Dropout(0.5))
model.add(Dense(units=64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]),activation='softmax'))
rmsprop = RMSprop(learning_rate=1e-3,momentum=0.9,weight_decay=1e-6,epsilon=1e-8)

model.compile(loss='categorical_crossentropy',optimizer=rmsprop,metrics=['accuracy'])

r = model.fit(np.array(X_train),np.array(y_train),epochs=200,batch_size=5,verbose=1)

model.save('chatbot_model.h5',r)