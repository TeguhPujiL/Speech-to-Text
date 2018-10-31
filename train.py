import tensorflow as tfs
tfs.logging.set_verbosity(tfs.logging.ERROR)
import os
import sugartensor as tf
from data import SpeechCorpus, voca_size
from model import *
from tqdm import tqdm
tqdm.monitor_interval = 0

# mengatur log level untuk debug
tf.sg_verbosity(10)

# hyper parameters
batch_size = 16    # total batch size

# menginput corpus ke tensorflow
data = SpeechCorpus(batch_size=batch_size * tf.sg_gpus())

# menginput mfcc feature dari file audio
inputs = tf.split(data.mfcc, tf.sg_gpus(), axis=0)

# mengambil label
labels = tf.split(data.label, tf.sg_gpus(), axis=0)

# panjang sequence kecuali zero-padding
seq_len = []
for input_ in inputs:
    seq_len.append(tf.not_equal(input_.sg_sum(axis=2), 0.).sg_int().sg_sum(axis=1))

# pemprosesan parallel untuk mengambil loss
@tf.sg_parallel
def get_loss(opt):
    # encode audio feature
    logit = get_logit(opt.input[opt.gpu_index], voca_size=voca_size)
    # CTC loss
    return logit.sg_ctc(target=opt.target[opt.gpu_index], seq_len=opt.seq_len[opt.gpu_index])

# train
tf.sg_train(lr=0.0001, loss=get_loss(input=inputs, target=labels, seq_len=seq_len),
            ep_size=data.num_batch, max_ep=100)
