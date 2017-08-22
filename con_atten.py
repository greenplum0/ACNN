# -*- coding: utf-8 -*-
# @auther tim
# @date 2016.10.30
import cPickle
from sys import path

import sys

path.append("..")
from keras.callbacks import Callback

from keras.engine import Layer

from keras.models import Model
from keras.layers import Embedding, Input, Convolution1D, MaxPooling1D,  Dense, TimeDistributed, merge, Flatten, Dropout, \
    Maximum
from keras.layers.merge import Concatenate
from sklearn import model_selection
from keras import backend as K

from utils.data_loader import *
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os

os.environ['KERAS_BACKEND']='theano'

# 参数定义
embedding_size = 300  # 定义每一个词需要映射到词向量的维度n
nb_filter = 100  # 卷积神经网络积极核函数的个数
filter_lengths = [3,4,5]  # 卷积神经网络中卷积核时空域长度
# filter_lengths = [3]
hidden_dims = 100
MAX_SEQUENCE_LENGTH=51
# 文件地址
target = "mr"
data_prefix = '../data_process/'+target
train_files = ["rt-polarity.neg", "rt-polarity.pos"]

# 加载数据
print ('数据加载中。。。')
x_texts, y_labels, neg_examples, pos_examples = \
    load_data_and_labels(data_dir=data_prefix, files=train_files,type='en', splitable=False)


# 分词并对句子进行序列映射
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_texts)
sequences = tokenizer.texts_to_sequences(x_texts)


# 词——下标字典
word_index = tokenizer.word_index

# 最长的句子长度
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print MAX_SEQUENCE_LENGTH

# labels = to_categorical(y_labels, nb_classes=2)
print('数据的格式{}'.format(data.shape))
print('标签的格式{}'.format(y_labels.shape))


f = open('../w2v/'+target+'/word_vectors.save', 'rb')
word_vectors = cPickle.load(f)
f.close()

#回调函数在每次验证集有提高是对测试进行测试
class LossHistory(Callback):
    def __init__(self):
        self.best_acc = 0;
    def on_epoch_end(self, epoch, logs={}):
        acc = logs.get('val_acc')
        if(acc>self.best_acc):
            self.best_acc = acc
    def on_train_end(self,logs={}):
        print("BestTestAcc:{}".format(self.best_acc))
        with open("./result", 'a') as f:
            f.write("BestTestAcc:{0}\n".format(self.best_acc))


class AttenLayer(Layer):
    def __init__(self, **kwargs):
        super(AttenLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert  len(input_shape)==3
        self.W = self.add_weight(
            name='atten_W',
            shape=(input_shape[-1],),
            initializer='normal',
            trainable=True
        )
        super(AttenLayer, self).build(input_shape)

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))

        ai = K.exp(eij)
        weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')

        weighted_input = x*weights.dimshuffle(0,1,'x')
        return weighted_input.sum(axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def get_model(type='atten',inner='inner'):
    # 获取嵌入层的向量矩阵
    def get_embedding_weight(word_vectors, word_index, embedding_size):
        embedding_matrix = np.random.uniform(-0.25, 0.25, size=[len(word_index) + 1, embedding_size]).astype(np.float32)
        for word, i in word_index.items():
            embedding_vector = word_vectors.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        embedding_matrix[0]=np.zeros(embedding_size, dtype='float32')
        return embedding_matrix

    print("得到词向量")
    embedding_matrix = get_embedding_weight(word_vectors, word_index, embedding_size)

    main_input = Input(shape=(MAX_SEQUENCE_LENGTH,),
                       dtype='int32',
                       name='main_input')
    embedding_layer = Embedding(
        embedding_matrix.shape[0],
        embedding_matrix.shape[1],
        input_length=MAX_SEQUENCE_LENGTH,
        weights=[embedding_matrix],
        trainable=True
        )
    embedding_sequences = embedding_layer(main_input)

    if(inner=='inner'):
        padding='valid'
    else:
        padding='same'
    conv_att_features = []
    for filter_length in filter_lengths:
        convolution_layer = Convolution1D(
            filters=nb_filter,
            kernel_size=filter_length,
            padding =padding,
            activation='relu',
            name='convLayer'+str(filter_length)
        )
        conv_out = convolution_layer(embedding_sequences)

        ###attenton#########
        if(type=='atten' and inner=='inner'):
            att_inpt = TimeDistributed(Dense(nb_filter))(conv_out)
            att_out = AttenLayer(name='AttenLayer'+str(filter_length))(att_inpt)
            conv_att_features.append(att_out)
        elif(type=='max'):
            out = MaxPooling1D(
                name='maxPooling'+ str(filter_length),
                pool_size=(MAX_SEQUENCE_LENGTH - filter_length + 1)
            )(conv_out)
            conv_att_features.append(out)
        else:
            conv_att_features.append(conv_out)


    if(len(filter_lengths)>1):
        X = Concatenate()(conv_att_features)
    else:
        X = conv_att_features[0]

    if(type=='max'):
        X = Flatten()(X)
    if(inner=='outer'):
        X = TimeDistributed(Dense(len(filter_lengths)*nb_filter),name='DenseTimeDistributed')(X)
        X = AttenLayer(name='AttenLayer')(X)

    # X = Dropout(0.5)(X)


    # x = Dense(output_dim=hidden_dims, W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01))(attention_features)
    x = Dense(units=hidden_dims,activation='relu')(X)

    # 添加最后的全连接层
    predictions = Dense(2, activation='softmax')(x)

    # 建立模型
    print('正在初始化模型....')
    model = Model(main_input, predictions)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adadelta',
        metrics=['accuracy']
    )
    return model


if __name__=="__main__":
    mode = sys.argv[1]
    inner = sys.argv[2]
    # 随机化数据
    print('随机化数据...')
    np.random.seed(200)
    shuffle_indices = np.random.permutation(np.arange(len(y_labels)))
    x_dataset = data[shuffle_indices]
    y_dataset = y_labels[shuffle_indices]
    print('params  mode:{0},inner:{1},n_filters:{2},filter_lengths:{3},Max_Sent_length:{4}'.format(mode,inner,str(nb_filter),str(filter_lengths),str(MAX_SEQUENCE_LENGTH)))

    # 10折交叉实验
    kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=100)

    cv=1
    for train_index, test_index in kf.split(x_dataset):
        x_train = x_dataset[train_index]
        y_train = y_dataset[train_index]
        x_test = x_dataset[test_index]
        y_test = y_dataset[test_index]

        model = get_model(type=mode,inner=inner)
        # 训练
        print('开始训练模型...')
        print('this is the {} times for training...'.format(cv))
        lossH = LossHistory()
        history = model.fit(
            x_train,
            y_train,
            verbose=2,
            batch_size=50,
            epochs=15,
            validation_data=(x_test,y_test),
            callbacks=[lossH]
        )
        cv=cv+1


