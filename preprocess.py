import tensorflow as tfs
tfs.logging.set_verbosity(tfs.logging.ERROR)
import numpy as np
import pandas as pd
import glob
import csv
import librosa
import scikits.audiolab
import data
import os
import subprocess

# path data
_data_path = "asset/Dataset/"

# prosedur process dataset wav
def process_wav(csv_file):

    # membuat csv writer
    writer = csv.writer(csv_file, delimiter=',')

    # membaca label info
    df = pd.read_table(_data_path + 'wav/speaker-info.txt', usecols=['ID'],
                       index_col=False, delim_whitespace=True)

    # membaca id file
    file_ids = []
    for d in [_data_path + 'wav/txt/p%d/' % uid for uid in df.ID.values]:
        file_ids.extend([f[-12:-4] for f in sorted(glob.glob(d + '*.txt'))])

    for i, f in enumerate(file_ids):

        # membaca filename pada data wav
        wave_file = _data_path + 'wav/wav48/%s/' % f[:4] + f + '.wav'
        fn = wave_file.split('/')[-1]
        target_filename = 'asset/Dataset/preprocess/mfcc/' + fn + '.npy'
        if os.path.exists( target_filename ):
            continue
        # print info
        print("dataset wav preprocessing (%d / %d) - '%s']" % (i, len(file_ids), wave_file))

        # load wav file
        wave, sr = librosa.load(wave_file, mono=True, sr=None)

        # re-sample frekuensi ( 48K -> 16K )
        wave = wave[::3]

        # mendapatkan mfcc feature
        mfcc = librosa.feature.mfcc(wave, sr=16000)

        # mendapatkan label index
        label = data.str2index(open(_data_path + 'wav/txt/%s/' % f[:4] + f + '.txt').read())

        # menyimpan hasil
        if len(label) < mfcc.shape[1]:
            # menyimpan meta info
            writer.writerow([fn] + label)

            # menyimpan mfcc
            np.save(target_filename, mfcc, allow_pickle=False)

# prosedur process dataset flac
def process_flac(csv_file, category):

    parent_path = _data_path + 'flac/' + category + '/'
    labels, wave_files = [], []

    # membuat csv writer
    writer = csv.writer(csv_file, delimiter=',')

    # membaca direktori list dari speaker
    speaker_list = glob.glob(parent_path + '*')
    for spk in speaker_list:

        # membaca direktori list dari chapter
        chapter_list = glob.glob(spk + '/*/')
        for chap in chapter_list:

            # membaca list file pada text label
            txt_list = glob.glob(chap + '/*.txt')
            for txt in txt_list:
                with open(txt, 'rt') as f:
                    records = f.readlines()
                    for record in records:
                        # memparsing record
                        field = record.split('-')  # split by '-'
                        speaker = field[0]
                        chapter = field[1]
                        field = field[2].split()  # split field[2] by ' '
                        utterance = field[0]  # kolom pertama pada id ucapan

                        # file name flac
                        wave_file = parent_path + '%s/%s/%s-%s-%s.flac' % \
                                                  (speaker, chapter,  speaker, chapter, utterance)
                        wave_files.append(wave_file)

                        # label index
                        labels.append(data.str2index(' '.join(field[1:])))  # kolom terakhir menjadi text label

    # menyimpan hasil
    for i, (wave_file, label) in enumerate(zip(wave_files, labels)):
        fn = wave_file.split('/')[-1]
        target_filename = 'asset/Dataset/preprocess/mfcc/' + fn + '.npy'
        if os.path.exists( target_filename ):
            continue
        # print info
        print("dataset flac preprocessing (%d / %d) - '%s']" % (i, len(wave_files), wave_file))

        # load flac file
        wave, sr = librosa.load(wave_file, mono=True, sr=None)

        # mendapatkan mfcc feature
        mfcc = librosa.feature.mfcc(wave, sr=16000)

        # menyimpan hasil
        if len(label) < mfcc.shape[1]:
            # menyimpan meta info
            writer.writerow([fn] + label)

            # menyimpan mfcc
            np.save(target_filename, mfcc, allow_pickle=False)

# membuat direktori
if not os.path.exists('asset/Dataset/preprocess'):
    os.makedirs('asset/Dataset/preprocess')
if not os.path.exists('asset/Dataset/preprocess/meta'):
    os.makedirs('asset/Dataset/preprocess/meta')
if not os.path.exists('asset/Dataset/preprocess/mfcc'):
    os.makedirs('asset/Dataset/preprocess/mfcc')

# menjalankan preprocessing

# dataset wav untuk training
csv_f = open('asset/Dataset/preprocess/meta/train.csv', 'w')
process_wav(csv_f)
csv_f.close()

# dataset flac untuk training
csv_f = open('asset/Dataset/preprocess/meta/train.csv', 'a+')
process_flac(csv_f,'train')
csv_f.close()

# dataset flac untuk validasi
csv_f = open('asset/Dataset/preprocess/meta/valid.csv', 'w')
process_flac(csv_f, 'valid')
csv_f.close()

# dataset flac untuk testing
csv_f = open('asset/Dataset/preprocess/meta/test.csv', 'w')
process_flac(csv_f, 'test')
csv_f.close()
