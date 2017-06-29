# encoding=utf-8

import numpy as np
import os
import re
import gensim, logging
import datetime
from keras.preprocessing.text import text_to_word_sequence
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()
'''
    加载消极观点数据和积极观点数据
    两种观点数据分别放在两个文件中
'''
def load_data_and_labels(data_dir=None,files=None,data=None,type='zh',splitable=False):
    if(data!=None):
        neg_examples=data[0]
        pos_examples=data[1]
    else:
        neg_file = files[0]
        pos_file = files[1]
        neg_examples = list(open(os.path.join(data_dir,neg_file)).readlines())
        pos_examples = list(open(os.path.join(data_dir,pos_file)).readlines())

    if splitable==False:
        # 中文
        if(type=='zh'):
            neg_examples = [line.strip() for line in neg_examples]
            pos_examples = [line.strip() for line in pos_examples]
        else:
            #英文
            neg_examples = [clean_str(line) for line in neg_examples if(len(clean_str(line))>1)]
            pos_examples = [clean_str(line) for line in pos_examples if(len(clean_str(line))>1)]
    else:
        if(type=='zh'):
            # 中文
            neg_examples = [line.strip().split() for line in neg_examples]
            pos_examples = [line.strip().split() for line in pos_examples]
        # neg_examples = [text_to_word_sequence(line) for line in neg_examples]
        # pos_examples = [text_to_word_sequence(line) for line in pos_examples]
        else:
            # 英文
            neg_examples = [clean_str(line).split() for line in neg_examples if(len(clean_str(line))>1)]
            pos_examples = [clean_str(line).split() for line in pos_examples if(len(clean_str(line))>1)]
    x_texts = pos_examples + neg_examples

    neg_labels = [[1,0] for _ in neg_examples]
    pos_labels = [[0,1] for _ in pos_examples]

    y_labels = np.concatenate([pos_labels,neg_labels], axis=0)

    # 加载数据中所有的词汇
    # vocab = []
    # for text in x_texts:
    #     vocab.extend(set(text))
    # vocab = set(vocab)
    return x_texts,y_labels,neg_examples,pos_examples

"""
    过滤掉一些长度过长或过短的句子
"""
def load_data_by_length(texts, accept_length):
    texts_filte = []
    for t in texts:
        if len(t) in accept_length:
            texts_filte.append(t)

    return texts_filte
'''
    训练时使用批量梯度下降法，每次迭代的batch数据数量
'''
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
        基于数据的batch迭代器
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

"""
    加载提前训练好的word2vec数据，将其针对当前数据
    保存当前数据中存在的词汇
    当初始化提前训练好的w2c数据时，目标数据中有可能某些词不再
    训练好的w2c数据中，故采用随机初始化的方式填充
"""
def load_bin_vec(fname, vocab, ksize=100):
    time_str = datetime.datetime.now().isoformat()
    print("{}:开始筛选w2v数据词汇...".format(time_str))

    word_vecs = {}
    model = gensim.models.Word2Vec.load_word2vec_format(fname,binary=True)
    for word in vocab:
        try:
            word_vecs[word] = model[word]
        except:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, ksize).astype(np.float32)

    # 提前训练好的w2v数据中的词汇
    # w2v_vocab = list(model.vocab.keys())
    # for word in vocab:
    #     if word in w2v_vocab:
    #         word_vecs[word] = model[word]
    #     else:
    #         word_vecs[word] = np.random.uniform(-1.0, 1.0, ksize)
    #
    time_str = datetime.datetime.now().isoformat()
    print("{}:筛选w2v数据词汇结束...".format(time_str))

    return word_vecs

"""
    获取词嵌入层的参数W
    得到一个矩阵，W[i]代表下标为i的词的词向量
    或者采用随机方式初始化W
    word_vecs {word:word_vector}
    vocab_ids_map {word:id}
"""
def get_W(word_vecs,vocab_ids_map, k=100, is_rand=False):
    time_str = datetime.datetime.now().isoformat()
    print("{}:生成嵌入层参数W...".format(time_str))

    vocab_size = len(word_vecs)
    W = np.random.uniform(-1.0,1.0,size=[vocab_size,k]).astype(np.float32)
    if is_rand==False:
        print("非随机初始化...")
        for i,word in enumerate(word_vecs):
            id = vocab_ids_map[word]
            W[id] = word_vecs[word]

    time_str = datetime.datetime.now().isoformat()
    print("{}:生成嵌入层参数W完毕".format(time_str))
    return W