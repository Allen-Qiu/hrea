# mulclass on amazon dataset
# bert

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import GlobalAveragePooling1D
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, BertTokenizer, TFBertModel

from tensorflow.keras.layers import Dense, Input, Embedding, Bidirectional,LSTM, \
    Dropout,BatchNormalization, Dot, Lambda, Reshape
from tensorflow.keras.initializers import Constant
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop, Adamax, Nadam
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
import json

print('amazon-mulclass-bert')

# 0. hyparameters
vocab_size          = 10000
ratio               = 0.1
word_embedding_dims = 50
cate_embedding_dims = 300
encode_dims         = 200         # size of encoded category and review
units               = 300
lr                  = 5e-5
batch_size          = 20
epoch_size          = 15
verbose             = 2
seq_len             = 512

review_file = 'wecr-reviews-10c.json'
checkpoint  = "bert-base-uncased"
# checkpoint = "../../transformers/model/bert-base-uncased"
# checkpoint = "../../transformers/model/distilbert-base-uncased"
print(review_file)

# 1. read data
np.random.seed(10)
fin   = open(review_file)
lines = fin.readlines()

labels    = list()
review_list = list()

for line in lines:
    dic = json.loads(line)
    labels.append(dic['class'])
    review_list.append(dic['review'])
fin.close()

size = len(labels)
s = set(labels)
label_idx = {}
for i, item in enumerate(s):
    label_idx[item] = i

y = []
for item in labels:
    y.append(label_idx[item])
y = np.array(y)
output_units = len(s)
print(output_units)

# 2. building dataset
shuffle_indices = np.random.permutation(np.arange(size))
review_shuffled = np.array(review_list)[shuffle_indices]
y_shuffled = y[shuffle_indices]

test_size = np.floor(ratio * size).astype(int)
review_train, review_test = review_shuffled[:-test_size], review_shuffled[-test_size:]

y_cate = tf.keras.utils.to_categorical(y_shuffled, num_classes = output_units)
y_train, y_test =  y_cate[:-test_size], y_cate[-test_size:]

# 2. build dataset for bert
tokenizer = BertTokenizer.from_pretrained(checkpoint)
encodings = tokenizer(review_train.tolist(), truncation=True, padding='max_length', max_length=seq_len)
X_train = [np.array(encodings["input_ids"]),
           np.array(encodings["token_type_ids"]),
           np.array(encodings["attention_mask"])]

encodings = tokenizer(review_test.tolist(), truncation=True, padding='max_length', max_length=seq_len)
X_test = [np.array(encodings["input_ids"]),
          np.array(encodings["token_type_ids"]),
          np.array(encodings["attention_mask"])]

checkpoint_filepath = '/tmp/amazon-bert'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

# 3. building model
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))
with strategy.scope():
    input_ids  = Input(shape=(seq_len,), dtype=tf.int32, name='input_ids')
    input_type = Input(shape=(seq_len,), dtype=tf.int32, name='token_type_ids')
    input_mask = Input(shape=(seq_len,), dtype=tf.int32, name='attention_mask')
    inputs = [input_ids, input_type, input_mask]

    bert = TFBertModel.from_pretrained(checkpoint)
    bert_outputs = bert(inputs)
    last_hidden_states = bert_outputs.last_hidden_state
    # avg = last_hidden_states[:, 0, :]
    avg = GlobalAveragePooling1D()(last_hidden_states)
    output = Dense(output_units, activation="softmax")(avg)
    model = Model(inputs=inputs, outputs=output)
    op = Adamax(learning_rate=lr)
    m = tf.keras.metrics.SparseCategoricalCrossentropy()
    m2 = tf.keras.metrics.AUC()
    # loss = SparseCategoricalCrossentropy()
    loss = CategoricalCrossentropy()
    model.compile(loss=loss, optimizer=op, metrics=['accuracy',m2])

model.fit(X_train, y_train, epochs=epoch_size,
          batch_size=batch_size,
          validation_split=0.1,
          callbacks=[checkpoint_callback],
          verbose=verbose)
model.load_weights(checkpoint_filepath)
model.evaluate(X_test, y_test,verbose=2)

# print('Test Accuracy: %s' % acc)
print("batch_size=%s"%(batch_size))
print("optimazation= %s" % op.get_config())


