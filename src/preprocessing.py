# encoding:utf-8

# import sys
# reload(sys)
# sys.setdefaultencoding('utf8')

import os
import codecs
import collections
from six.moves import cPickle
import numpy as np


class TextLoader:
    def __init__(self, data_dir, batch_size, seq_length, mini_frq=3):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.mini_frq = mini_frq

        input_file = os.path.join(data_dir, "input_zh_size1.txt")
        vocab_file = os.path.join(data_dir, "vocab.zh.txt")

        self.preprocess(input_file, vocab_file)
        self.create_batches()
        self.reset_batch_pointer()

    def build_vocab(self, sentences):
        word_counts = collections.Counter()  # 计数器
        if not isinstance(sentences, list):
            sentences = [sentences]
        for sent in sentences:
            word_counts.update(sent)
        # most_common()返回按出现次数排序的单词 eg.:{('我',100),('的',34)}，mini_frq规定了最小的频次，低于此频次不进入词汇表
        vocabulary_inv = ['<START>', '<UNK>', '<END>'] + [x[0] for x in word_counts.most_common() if
                                                          x[1] >= self.mini_frq]
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}  # 依据vocab_inv生成索引字典
        return [vocabulary, vocabulary_inv]

    def preprocess(self, input_file, vocab_file):  # 提取input_file当中的所有词汇，然后以二进制存储在vocab_file中(vocab.zh.txt)
        with codecs.open(input_file, 'r', 'utf-8') as f:
            lines = f.readlines()
            if lines[0][:1] == codecs.BOM_UTF8:
                lines[0] = lines[0][1:]
            lines = [line.strip().split() for line in lines]  # 按行处理
        self.vocab, self.words = self.build_vocab(
            lines)  # vocab:带索引的词汇表(字典) words:不带索引(列表) 为什么还要返回words？猜测是列表进行某些操作好用一些，后续还会从词汇表提取东西，有了索引后就没那么好处理了
        self.vocab_size = len(self.words)

        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.words, f)  # 序列化词汇表

        raw_data = [[0] * self.seq_length + [self.vocab.get(w, 1) for w in line] + [2] * self.seq_length for line in
                    lines]  # 文本的索引数据(0和2是什么意思?)
        self.raw_data = raw_data

    def create_batches(self):
        # each element in xdata list is the context which contains fives words(display as indices),
        # and the corresponding element with the same index in ydata is the exact word we are trying to evaluate
        xdata, ydata = list(), list()
        for row in self.raw_data:
            for ind in range(self.seq_length, len(row)):
                xdata.append(row[ind - self.seq_length:ind])
                ydata.append([row[ind]])
        self.num_batches = int(len(xdata) / self.batch_size)

        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        xdata = np.array(xdata[:self.num_batches * self.batch_size])
        ydata = np.array(ydata[:self.num_batches * self.batch_size])

        self.x_batches = np.split(xdata, self.num_batches, 0)
        self.y_batches = np.split(ydata, self.num_batches, 0)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
