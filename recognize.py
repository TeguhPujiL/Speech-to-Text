import tensorflow as tfs
tfs.logging.set_verbosity(tfs.logging.ERROR)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sugartensor as tf
import numpy as np
import librosa
from model import *
import data
import sys

# mengatur log level untuk debug
tf.sg_verbosity(10)

# hyper parameters
batch_size = 1     # batch size

# inputs
# panjang kata
voca_size = data.voca_size

# menginput mfcc feature pada file audio
x = tf.placeholder(dtype=tf.sg_floatx, shape=(batch_size, None, 20))

# panjang sequence kecuali zero-padding
seq_len = tf.not_equal(x.sg_sum(axis=2), 0.).sg_int().sg_sum(axis=1)

# encode audio feature
logit = get_logit(x, voca_size=voca_size)

# ctc decoding
decoded, _ = tf.nn.ctc_beam_search_decoder(logit.sg_transpose(perm=[1, 0, 2]), seq_len, merge_repeated=False)

# to dense tensor
y = tf.sparse_to_dense(decoded[0].indices, decoded[0].dense_shape, decoded[0].values) + 1

# regcognize audio file

# perintah untuk menginput path file audio
tf.sg_arg_def(file=('', 'speech wave file to recognize.'))

# load audio file
file = sys.argv[1]
wav, sr = librosa.load(file, mono=True, sr=16000)

# mendapatkan mfcc feature
mfcc = np.transpose(np.expand_dims(librosa.feature.mfcc(wav, 16000), axis=0), [0, 2, 1])

# run network
with tf.Session() as sess:

    # init variables
    tf.sg_init(sess)

    # restore parameters
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('asset/train'))

    # run session
    label = sess.run(y, feed_dict={x: mfcc})

    # print label
    data.print_index(label)
