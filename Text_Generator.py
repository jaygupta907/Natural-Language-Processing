import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense, Embedding, LSTM, Dropout, Bidirectional, GlobalMaxPooling1D
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
import re
import numpy as np
import pandas as pd
import tensorflow as tf


with open("text.txt", "r", encoding="utf-8") as f:
    story_data = f.read()
print(story_data)


def clean_text(text):
    text = re.sub(r',', '', text)
    text = re.sub(r'\'', '',  text)
    text = re.sub(r'\"', '', text)
    text = re.sub(r'\(', '', text)
    text = re.sub(r'\)', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'“', '', text)
    text = re.sub(r'”', '', text)
    text = re.sub(r'’', '', text)
    text = re.sub(r'\.', '', text)
    text = re.sub(r';', '', text)
    text = re.sub(r':', '', text)
    text = re.sub(r'\-', '', text)

    return text


lower_data = story_data.lower()
split_data = lower_data.splitlines()
final = ''
for line in split_data:
    line = clean_text(line)
    final += '\n' + line

print(final)

# splitting again to get list of cleaned and splitted data ready to be processed
final_data = final.split('\n')
print(final_data)
# Instantiating the Tokenizer
max_vocab = 1000000
tokenizer = Tokenizer(num_words=max_vocab)
tokenizer.fit_on_texts(final_data)

# Getting the total number of words of the data.
word2idx = tokenizer.word_index
# Adding 1 to the vocab_size because the index starts from 1 not 0. This will make it uniform when using it further
vocab_size = len(word2idx) + 1


# We will turn the sentences to sequences line by line and create n_gram sequences

input_seq = []

for line in final_data:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_seq = token_list[:i+1]
        input_seq.append(n_gram_seq)

print(input_seq)

# Getting the maximum length of sequence for padding purpose
max_seq_length = max(len(x) for x in input_seq)
print(max_seq_length)

# Padding the sequences and converting them to array
input_seq = np.array(pad_sequences(
    input_seq, maxlen=max_seq_length, padding='pre'))
print(input_seq)

# Taking xs and labels to train the model.

# xs contains every word in sentence except the last one because we are using this value to predict the y value
xs = input_seq[:, :-1]
# labels contains only the last word of the sentence which will help in hot encoding the y value in next step
labels = input_seq[:, -1]
print("xs: ", xs)
print("labels:", labels)


# one-hot encoding the labels according to the vocab size

# The matrix is square matrix of the size of vocab_size. Each row will denote a label and it will have
# a single +ve value(i.e 1) for that label and other values will be zero.

ys = to_categorical(labels, num_classes=vocab_size)
print(ys)


# using the functional APIs of keras to define the model

# using 1 less value becasuse we are preserving the last value for predicted word
i = Input(shape=(max_seq_length - 1, ))
x = Embedding(vocab_size, 124)(i)
x = Dropout(0.2)(x)
x = LSTM(520, return_sequences=True)(x)
x = Bidirectional(layer=LSTM(340, return_sequences=True))(x)
x = GlobalMaxPooling1D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(vocab_size, activation='softmax')(x)

model = Model(i, x)

model.compile(optimizer=Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


r = model.fit(xs, ys, epochs=100)

# Evaluating the model on accuracy
plt.plot(r.history['accuracy'])


def predict_words(seed, no_words):
    for i in range(no_words):
        token_list = tokenizer.texts_to_sequences([seed])[0]
        token_list = pad_sequences(
            [token_list], maxlen=max_seq_length-1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=1)

        new_word = ''

        for word, index in tokenizer.word_index.items():
            if predicted == index:
                new_word = word
                break
        seed += " " + new_word
    print(seed)

# predicting or generating the poem with the seed text


seed_text = ''
next_words = 100

predict_words(seed_text, next_words)

# saving the model

model.save('poem_generator.h5')  # Will create a HDF5 file of the model
