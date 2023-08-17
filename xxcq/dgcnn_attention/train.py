# -*- coding: utf-8 -*-
# @Time        : 2022/11/2 14:24
# @Author      : tianyunzqs
# @Description :

import os
import sys
import json
import random
import jieba
import ahocorasick
import numpy as np
from tqdm import tqdm
from gensim.models import Word2Vec
import tensorflow as tf
import keras.backend as K
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.optimizers import Nadam
from keras.callbacks import Callback
from bert4keras.optimizers import extend_with_exponential_moving_average
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
from dgcnn_attention.dgcnn_attention_model import create_model, ExponentialMovingAverage

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf.Session(config=config)
K.set_session(session)

char_size = 256
maxlen = 2054  # 512

word2vec = Word2Vec.load(os.path.join(project_path, 'vector/word2vec_baike'))
id2word = {i + 1: j for i, j in enumerate(word2vec.wv.index2word)}
word2id = {j: i for i, j in id2word.items()}
word2vec = word2vec.wv.syn0
word_size = word2vec.shape[1]
word2vec = np.concatenate([np.zeros((1, word_size)), word2vec])


def sent2vec(S):
    """S格式：[[w1, w2]]
    """
    V = []
    for s in S:
        V.append([])
        for w in s:
            for _ in w:
                V[-1].append(word2id.get(w, 0))
    V = seq_padding(V)
    V = word2vec[V]
    return V


def load_data(path):
    dataset = []
    _char2id, _id2char = dict(), dict()
    _predicate2id, _id2predicate = dict(), dict()
    _predicates = dict()  # 格式：{predicate: [(subject, predicate, object)]}
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for d in data:
        for s in d["text"]:
            if s not in _char2id:
                _char2id[s] = len(_char2id) + 2  # 1是unk，0是padding
                _id2char[len(_id2char) + 2] = s
        spo_list = []
        for dd in d["spo_list"]:
            spo_list.append(tuple(dd))
            if dd[1] not in _predicates:
                _predicates[dd[1]] = []
            _predicates[dd[1]].append(tuple(dd))
            if dd[1] not in _predicate2id:
                _predicate2id[dd[1]] = len(_predicate2id)
                _id2predicate[len(_id2predicate)] = dd[1]
        dataset.append({'text': d["text"], 'spo_list': spo_list})
    return dataset, _char2id, _id2char, _predicate2id, _id2predicate, _predicates


train_data, char2id, id2char, predicate2id, id2predicate, predicates = load_data(
    os.path.join(project_path, 'data/train_triples.json'))
dev_data, _, _, _, _, _ = load_data(os.path.join(project_path, 'data/dev_triples.json'))
num_classes = len(id2predicate)


def random_generate(d, spo_list_key):
    r = np.random.random()
    if r > 0.5:
        return d
    else:
        k = np.random.randint(len(d[spo_list_key]))
        spi = d[spo_list_key][k]
        k = np.random.randint(len(predicates[spi[1]]))
        spo = predicates[spi[1]][k]
        F = lambda s: s.replace(spi[0], spo[0]).replace(spi[2], spo[2])
        text = F(d['text'])
        spo_list = [(F(sp[0]), sp[1], F(sp[2])) for sp in d[spo_list_key]]
        return {'text': text, spo_list_key: spo_list}


def seq_padding(X, padding=0):
    _max_len = max([len(x) for x in X])
    return np.array([
        np.concatenate([x, [padding] * (_max_len - len(x))]) if len(x) < _max_len else x for x in X
    ])


class ACUnicode:
    """稍微封装一下，弄个支持unicode的AC自动机
    """
    def __init__(self):
        self.ac = ahocorasick.Automaton()

    def add_word(self, k, v):
        # k = k.encode('utf-8')
        return self.ac.add_word(k, v)

    def make_automaton(self):
        return self.ac.make_automaton()

    def iter(self, s):
        # s = s.encode('utf-8')
        return self.ac.iter(s)


class SPOSearcher:
    def __init__(self, data):
        self.s_ac = ACUnicode()
        self.o_ac = ACUnicode()
        self.sp2o = {}
        self.spo_total = {}
        for i, d in tqdm(enumerate(data), desc=u'构建三元组搜索器'):
            for s, p, o in d['spo_list']:
                self.s_ac.add_word(s, s)
                self.o_ac.add_word(o, o)
                if (s, o) not in self.sp2o:
                    self.sp2o[(s, o)] = set()
                if (s, p, o) not in self.spo_total:
                    self.spo_total[(s, p, o)] = set()
                self.sp2o[(s, o)].add(p)
                self.spo_total[(s, p, o)].add(i)
        self.s_ac.make_automaton()
        self.o_ac.make_automaton()

    def extract_items(self, text_in, text_idx=None):
        R = set()
        for s in self.s_ac.iter(text_in):
            for o in self.o_ac.iter(text_in):
                if (s[1], o[1]) in self.sp2o:
                    for p in self.sp2o[(s[1], o[1])]:
                        if text_idx is None:
                            R.add((s[1], p, o[1]))
                        elif self.spo_total[(s[1], p, o[1])] - {text_idx}:
                            R.add((s[1], p, o[1]))
        return list(R)


spoer = SPOSearcher(train_data)


class TyDataGenerator:
    def __init__(self, data, batch_size=4):  # 64
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            T1, T2, S1, S2, K1, K2, O1, O2, PRES, PREO = [], [], [], [], [], [], [], [], [], []
            for i in idxs:
                spo_list_key = 'spo_list'  # if np.random.random() > 0.5 else 'spo_list_with_pred'
                d = random_generate(self.data[i], spo_list_key)
                text = d['text'][:maxlen]
                text_words = jieba.lcut(text)
                text = ''.join(text_words)
                items = {}
                for sp in d[spo_list_key]:
                    subjectid = text.find(sp[0])
                    objectid = text.find(sp[2])
                    if subjectid != -1 and objectid != -1:
                        key = (subjectid, subjectid+len(sp[0]))
                        if key not in items:
                            items[key] = []
                        items[key].append((objectid, objectid+len(sp[2]), predicate2id[sp[1]]))
                pre_items = {}
                for sp in spoer.extract_items(text, i):
                    subjectid = text.find(sp[0])
                    objectid = text.find(sp[2])
                    if subjectid != -1 and objectid != -1:
                        key = (subjectid, subjectid+len(sp[0]))
                        if key not in pre_items:
                            pre_items[key] = []
                        pre_items[key].append((objectid, objectid+len(sp[2]), predicate2id[sp[1]]))
                if items:
                    T1.append([char2id.get(c, 1) for c in text])  # 1是unk，0是padding
                    T2.append(text_words)
                    s1, s2 = np.zeros(len(text)), np.zeros(len(text))
                    for j in items:
                        s1[j[0]] = 1
                        s2[j[1]-1] = 1
                    pres = np.zeros((len(text), 2))
                    for j in pre_items:
                        pres[j[0], 0] = 1
                        pres[j[1]-1, 1] = 1
                    k1, k2 = np.array(list(items.keys())).T
                    k1 = random.choice(k1)
                    k2 = random.choice(k2[k2 >= k1])
                    o1, o2 = np.zeros((len(text), num_classes)), np.zeros((len(text), num_classes))
                    for j in items.get((k1, k2), []):
                        o1[j[0], j[2]] = 1
                        o2[j[1]-1, j[2]] = 1
                    preo = np.zeros((len(text), num_classes, 2))
                    for j in pre_items.get((k1, k2), []):
                        preo[j[0], j[2], 0] = 1
                        preo[j[1]-1, j[2], 1] = 1
                    preo = preo.reshape((len(text), -1))
                    S1.append(s1)
                    S2.append(s2)
                    K1.append([k1])
                    K2.append([k2-1])
                    O1.append(o1)
                    O2.append(o2)
                    PRES.append(pres)
                    PREO.append(preo)
                    if len(T1) == self.batch_size or i == idxs[-1]:
                        T1 = seq_padding(T1)
                        T2 = sent2vec(T2)
                        S1 = seq_padding(S1)
                        S2 = seq_padding(S2)
                        O1 = seq_padding(O1, np.zeros(num_classes))
                        O2 = seq_padding(O2, np.zeros(num_classes))
                        K1, K2 = np.array(K1), np.array(K2)
                        PRES = seq_padding(PRES, np.zeros(2))
                        PREO = seq_padding(PREO, np.zeros(num_classes * 2))
                        yield [T1, T2, S1, S2, K1, K2, O1, O2, PRES, PREO], None
                        T1, T2, S1, S2, K1, K2, O1, O2, PRES, PREO = [], [], [], [], [], [], [], [], [], []


subject_model, object_model, train_model = create_model(word_size, num_classes, maxlen, char_size, char2id)

AdamEMA = extend_with_exponential_moving_average(Adam, name='AdamEMA')
optimizer = AdamEMA(lr=0.008)
train_model.compile(optimizer=optimizer)

# train_model.compile(optimizer=Adam(1e-3))
# train_model.compile(optimizer=Nadam(lr=0.008, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004))
# train_model.compile(optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True))
train_model.summary()

# EMAer = ExponentialMovingAverage(train_model)
# EMAer.inject()


def extract_items(text_in):
    # print(text_in)
    result_spo = set()

    text_words = jieba.lcut(text_in.lower())
    text_in = ''.join(text_words)
    pre_items = {}
    for sp in spoer.extract_items(text_in):
        subjectid = text_in.find(sp[0])
        objectid = text_in.find(sp[2])
        if subjectid != -1 and objectid != -1:
            key = (subjectid, subjectid+len(sp[0]))
            if key not in pre_items:
                pre_items[key] = []
            pre_items[key].append((objectid, objectid+len(sp[2]), predicate2id[sp[1]]))
    _pres = np.zeros((len(text_in), 2))
    for j in pre_items:
        _pres[j[0], 0] = 1
        _pres[j[1]-1, 1] = 1
    _pres = np.expand_dims(_pres, 0)

    _t1 = [char2id.get(c, 1) for c in text_in]
    _t1 = np.array([_t1])
    _t2 = sent2vec([text_words])
    _k1, _k2 = subject_model.predict([_t1, _t2, _pres])
    _k1, _k2 = _k1[0, :, 0], _k2[0, :, 0]
    _k1, _k2 = np.where(_k1 > 0.4)[0], np.where(_k2 > 0.35)[0]
    _subjects, _PREO = [], []
    for i in _k1:
        j = _k2[_k2 >= i]
        if len(j) > 0:
            j = j[0]
            _subject = text_in[i: j+1]
            _subjects.append((_subject, i, j))
            _preo = np.zeros((len(text_in), num_classes, 2))
            for _ in pre_items.get((i, j+1), []):
                _preo[_[0], _[2], 0] = 1
                _preo[_[1]-1, _[2], 1] = 1
            _preo = _preo.reshape((len(text_in), -1))
            _PREO.append(_preo)
    if _subjects:
        _PRES = np.repeat(_pres, len(_subjects), 0)
        _PREO = np.array(_PREO)
        _t1 = np.repeat(_t1, len(_subjects), 0)
        _t2 = np.repeat(_t2, len(_subjects), 0)
        _k1, _k2 = np.array([_s[1:] for _s in _subjects]).T.reshape((2, -1, 1))
        _o1, _o2 = object_model.predict([_t1, _t2, _k1, _k2, _PRES, _PREO])
        for i, _subject in enumerate(_subjects):
            _oo1, _oo2 = np.where(_o1[i] > 0.5), np.where(_o2[i] > 0.4)
            for _ooo1, _c1 in zip(*_oo1):
                for _ooo2, _c2 in zip(*_oo2):
                    if _ooo1 <= _ooo2 and _c1 == _c2:
                        _object = text_in[_ooo1: _ooo2+1]
                        _predicate = id2predicate[_c1]
                        result_spo.add((_subject[0], _predicate, _object))
                        break
    # print(result_spo)
    return list(result_spo)


class Evaluate(Callback):
    def __init__(self):
        self.F1 = []
        self.best = [0., 0., 0., 0]
        self.passed = 0
        self.stage = 0

    def on_batch_begin(self, batch, logs=None):
        """第一个epoch用来warmup，不warmup有不收敛的可能。
        """
        if self.passed < self.params['steps']:
            lr = (self.passed + 1.) / self.params['steps'] * 1e-3
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
        # EMAer.apply_ema_weights()
        optimizer.apply_ema_weights()
        f1, precision, recall = self.evaluate()
        self.F1.append(f1)
        if f1 > self.best[0]:
            self.best = [f1, precision, recall, epoch+1]
            # 保存模型
            train_model.save_weights('best_model.weights')
            # train_model.save('best_model.h5')
            # with open('best_model.json', 'w') as file:
            #     file.write(train_model.to_json())
            # subject_model.save("best_subject_model.h5")
            # object_model.save("best_object_model.h5")

        print('f1: %.4f, precision: %.4f, recall: %.4f, '
              'best f1: %.4f, best precision: %.4f, best recall: %.4f, best epoch: %d\n' %
              (f1, precision, recall, self.best[0], self.best[1], self.best[2], self.best[3]))
        # EMAer.reset_old_weights()
        optimizer.reset_old_weights()
        # if epoch + 1 == 50 or (
        #     self.stage == 0 and epoch > 10 and
        #     (f1 < 0.5 or np.argmax(self.F1) < len(self.F1) - 8)
        # ):
        #     self.stage = 1
        #     train_model.load_weights('best_model.weights')
        #     optimizer.initialize()
        #     K.set_value(self.model.optimizer.lr, 1e-4)
        #     K.set_value(self.model.optimizer.iterations, 0)
        #     opt_weights = K.batch_get_value(self.model.optimizer.weights)
        #     opt_weights = [w * 0. for w in opt_weights]
        #     K.batch_set_value(zip(self.model.optimizer.weights, opt_weights))

    @staticmethod
    def evaluate():
        orders = ['subject', 'predicate', 'object']
        A, B, C = 1e-10, 1e-10, 1e-10
        with open('dev_pred.json', 'w') as fw:
            for d in tqdm(iter(dev_data)):
                pred_spo = set(extract_items(d['text']))
                true_spo = set(d['spo_list'])
                A += len(pred_spo & true_spo)
                B += len(pred_spo)
                C += len(true_spo)
                s = json.dumps({
                    'text': d['text'],
                    'spo_list': [
                        dict(zip(orders, spo)) for spo in true_spo
                    ],
                    'spo_list_pred': [
                        dict(zip(orders, spo)) for spo in pred_spo
                    ],
                    'new': [
                        dict(zip(orders, spo)) for spo in pred_spo - true_spo
                    ],
                    'lack': [
                        dict(zip(orders, spo)) for spo in true_spo - pred_spo
                    ]
                }, ensure_ascii=False, indent=4)
                fw.write(s + '\n')

        return 2 * A / (B + C), A / B, A / C


def test(test_data):
    """输出测试结果
    """
    orders = ['em1Text', 'label', 'em2Text']
    with open('false.json', 'w',  encoding='utf-8') as fw:
        for d in tqdm(iter(test_data)):
            pred_spo = set(extract_items(d['text']))
            s = json.dumps({
                'sentText': d['text'],
                'relationMentions': [
                    dict(zip(orders, spo + ('', ''))) for spo in pred_spo
                ]
            }, ensure_ascii=False)
            fw.write(s + '\n')


train_D = TyDataGenerator(train_data)
evaluator = Evaluate()


if __name__ == '__main__':
    # with open("test_triples.json", 'r', encoding='utf-8') as f:
    #     data = json.load(f)
    # test(data)
    do_train = False
    if(do_train):
        train_model.fit_generator(train_D.__iter__(),
                                 steps_per_epoch=len(train_D),
                                  epochs=10,
                                  callbacks=[evaluator]
                                  )
    else:
        train_model.load_weights('./best_model.weights')
        with open("test_triples.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        test(data)
