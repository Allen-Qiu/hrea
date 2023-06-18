"""
predicting ratings of reviews in yelp
"""

import numpy as np
import tensorflow as tf
import json

from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Average, Dense, Input, GlobalAveragePooling1D

# 0. hyperparameters ----------------
seq_len = 512
batch_size = 20
epoch = 20
local = "bert-base-uncased"
lr = 3e-5

# 1. dataset ---------
fin = open('yelp-reviews.json')
lines = fin.readlines()
reviews = list()
stars = list()

for line in lines:
    rdict = json.loads(line)
    reviews.append(rdict['text'])
    stars.append(rdict['stars']-1)
# steps = max([len(line) for line in reviews])

np.random.seed(42)
shuffle_indices = np.random.permutation(np.arange(len(stars)))

x_shuffled = np.array(reviews)[shuffle_indices]
y_shuffled = np.array(stars)[shuffle_indices]

# Split train/test set
x_train, x_test = x_shuffled[:-1000], x_shuffled[-1000:]
y_train, y_test = y_shuffled[:-1000], y_shuffled[-1000:]

N_CLASSES = len(set(y_train))
print(N_CLASSES)
print(len(reviews))

# 2. prepare inputs for bert ----------
tokenizer = BertTokenizer.from_pretrained(local)

encodings = tokenizer(x_train.tolist(), truncation=True, padding='max_length', max_length=seq_len)
X_train = [np.array(encodings["input_ids"]),
           np.array(encodings["token_type_ids"]),
           np.array(encodings["attention_mask"])]

encodings = tokenizer(x_test.tolist(), truncation=True, padding='max_length', max_length=seq_len)
X_test = [np.array(encodings["input_ids"]),
          np.array(encodings["token_type_ids"]),
          np.array(encodings["attention_mask"])]

# 3. build model ---------------
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    input_ids = Input(shape=(seq_len,), dtype=tf.int32, name='input_ids')
    input_type = Input(shape=(seq_len,), dtype=tf.int32, name='token_type_ids')
    input_mask = Input(shape=(seq_len,), dtype=tf.int32, name='attention_mask')
    inputs = [input_ids, input_type, input_mask]

    bert = TFBertModel.from_pretrained(local)
    bert_outputs = bert(inputs)
    last_hidden_states = bert_outputs.last_hidden_state
    avg = GlobalAveragePooling1D()(last_hidden_states)
    # avg = last_hidden_states[:, 0, :]
    # output = Dense(1)(avg)
    output = Dense(N_CLASSES, activation="softmax")(avg)
    # opt = tf.keras.optimizers.Adam()
    op = tf.keras.optimizers.Adamax(learning_rate=lr)
    model = Model(inputs=inputs, outputs=output)
    rmse = tf.keras.metrics.RootMeanSquaredError()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    # loss_fn = tf.keras.losses.CategoricalCrossentropy()
    # loss_fn = tf.keras.losses.BinaryCrossentropy()
    # model.compile(loss='mse', optimizer=opt, metrics=['mae',rmse])
    model.compile(loss=loss_fn, optimizer=op, metrics=['mae','accuracy'])

# 3. fine-tunning --------------------
model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, verbose=2, validation_split=0.1)
loss = model.evaluate(X_test,y_test,verbose=2)
print(loss)
# res = model.predict(X_test)
# print(res.shape)
# print(res[0])
# for i in range(len(y_test)):
#     print("%s, %s" % (res[i], y_test[i]))
