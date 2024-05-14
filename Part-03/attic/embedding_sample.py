import tensorflow as tf
from tensorflow.keras import layers

target_sentence = [14, 12,  2, 15,  0] # <SOS> Wait. <EOS> <pad>
embedding_dim = 3

embedding_layer = layers.Embedding(max(target_sentence) + 1, embedding_dim)

for s in target_sentence:
    result = embedding_layer(tf.constant(s)).numpy()
    print("{:>10} => {}".format(s, result))
