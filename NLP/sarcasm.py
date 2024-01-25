import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Embedding, LSTM

!wget --no-check-certificate \
  https://raw.githubusercontent.com/lusiaulia/learn/main/NLP/Sarcasm_Headlines_Dataset.json\
  -O /tmp/Sarcasm_Headlines_Dataset.json 

df = pd.read_json('/tmp/Sarcasm_Headlines_Dataset.json', lines=True)

sentences = []
labels = []

for item in range(len(df)):
  sentences.append(df.headline[item])
  labels.append(df.is_sarcastic[item])

vocab_size = 1000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = round(len(sentences)*0.8)

training_sentences = sentences[:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[:training_size]
testing_labels = labels[training_size:]

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
model.add(LSTM(32, return_sequences = True))
model.add(Flatten())
model.add(Dense(24, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

num_epochs = 20
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

class MyCallbacks(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs is None:
                logs = {}
            if (logs.get("accuracy") is not None and logs.get('val_accuracy') is not None and logs.get(
                    'accuracy') > 0.87 and logs.get('val_accuracy') > 0.83):
                print("\n Desired accuracy > 87% and validation_accuracy > 83%, so cancelling training!")
                self.model.stop_training = True

callbacks = MyCallbacks()

history = model.fit(training_padded,
          training_labels,
          epochs=num_epochs,
          callbacks = callbacks,
          validation_data=(testing_padded, testing_labels),
          verbose=1)
