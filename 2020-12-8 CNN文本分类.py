# 构建计算图——LSTM模型
#    embedding
#    LSTM
#    fc
#    train_op
# 训练流程代码
# 数据集封装
#    api: next_batch(batch_size)
# 词表封装:
#    api: sentence2id(text_sentence): 句子转换id
# 类别的封装：
#    api: category2id(text_category).

import tensorflow as tf
import os
import sys
import numpy as np
import math

tf.logging.set_verbosity(tf.logging.INFO)
def get_default_params():
    return tf.contrib.training.HParams(
        num_embedding_size=16,
        num_timesteps=50,
        # CNN过滤器个数，通道数
        num_filters=64,
        # 卷积核大小
        num_kernel_size=3,
        num_fc_nodes=32,
        batch_size=100,
        learning_rate=0.001,
        num_word_threshold=10,
    )

hps = get_default_params()

train_file = r'C:\Users\ext_renqq\Desktop\文本分类数据/cnews.train.seg.txt'
val_file = r'C:\Users\ext_renqq\Desktop\文本分类数据/cnews.val.seg.txt'
test_file = r'C:\Users\ext_renqq\Desktop\文本分类数据/cnews.test.seg.txt'
vocab_file = r'C:\Users\ext_renqq\Desktop\文本分类数据/cnews.vocab.txt'
category_file = r'C:\Users\ext_renqq\Desktop\文本分类数据/cnews.category.txt'
output_folder = r'C:\Users\ext_renqq\Desktop\文本分类数据/run_text_rnn'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)


class Vocab(object):
    def __init__(self, filename, num_word_threshold):
        # 词典
        self._word_to_id = {}
        # <UNK> 的id（初始值）
        self._unk = -1
        # 频率下限
        self._num_word_threshold = num_word_threshold
        # 将词典读出来存到dict里
        self._read_dict(filename)

    def _read_dict(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            word, frequency = line.strip('\r\n').split('\t')
            word = word
            frequency = int(frequency)
            # 低于下限不要
            if frequency < self._num_word_threshold:
                continue
            idx = len(self._word_to_id)
            if word == '<UNK>':
                # 刷新UNK的id
                self._unk = idx
            self._word_to_id[word] = idx

    def word_to_id(self, word):
        # 如果没有word返回UNK
        return self._word_to_id.get(word, self._unk)

    @property
    def unk(self):
        return self._unk

    def size(self):
        return len(self._word_to_id)

    def sentence_to_id(self, sentence):
        # 把分词字典里面的id取出来变成字典
        word_ids = [self.word_to_id(cur_word) \
                    for cur_word in sentence.split()]
        return word_ids


class CategoryDict(object):
    def __init__(self, filename):
        # 读取类别并存入字典，给每个一个id
        self._category_to_id = {}
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            category = line.strip('\r\n')
            idx = len(self._category_to_id)
            self._category_to_id[category] = idx

    def size(self):
        return len(self._category_to_id)

    def category_to_id(self, category):
        # 传入类别返回id
        if category not in self._category_to_id:
            raise Exception("%s is not in our category list" % category)
        return self._category_to_id[category]


# 建立词典
vocab = Vocab(vocab_file, hps.num_word_threshold)
vocab_size = vocab.size()
# 打日志
tf.logging.info('vocab_size: %d' % vocab_size)
# 建立类别词典
category_vocab = CategoryDict(category_file)
num_classes = category_vocab.size()
tf.logging.info('num_classes: %d' % num_classes)
test_str = '体育'
tf.logging.info(
    'label: %s, id: %d' % (
        test_str,
        category_vocab.category_to_id(test_str)))


class TextDataSet(object):
    def __init__(self, filename, vocab, category_vocab, num_timesteps):
        self._vocab = vocab
        self._category_vocab = category_vocab
        self._num_timesteps = num_timesteps
        # matrix
        self._inputs = []
        # vector
        self._outputs = []
        self._indicator = 0
        self._parse_file(filename)

    def _parse_file(self, filename):
        tf.logging.info('Loading data from %s', filename)
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            label, content = line.strip('\r\n').split('\t')
            # id_label = self._category_vocab.category_to_id(label)
            # id_words = self._vocab.sentence_to_id(content)
            # id_words = id_words[0: self._num_timesteps]
            # padding_num = self._num_timesteps - len(id_words)
            # id_words = id_words + [
            #     self._vocab.unk for i in range(padding_num)]
            # self._inputs.append(id_words)
            # self._outputs.append(id_label)
            # 对于训练集的数据进行了切割，加个判断
            if label in self._category_vocab._category_to_id:
                # 将传入的类别转化为对应的id
                id_label = self._category_vocab.category_to_id(label)
                # 将传入的句子转化为一个词一个词对应的id
                id_words = self._vocab.sentence_to_id(content)
                # 控制句长在50,超过截断
                id_words = id_words[0: self._num_timesteps]
                padding_num = self._num_timesteps - len(id_words)
                id_words = id_words + [
                    self._vocab.unk for i in range(padding_num)]
                # 将句子的向量放到输入list
                self._inputs.append(id_words)
                # 将类别的向量放到输出的list
                self._outputs.append(id_label)
        self._inputs = np.asarray(self._inputs, dtype=np.int32)
        self._outputs = np.asarray(self._outputs, dtype=np.int32)
        self._random_shuffle()

    def _random_shuffle(self):
        p = np.random.permutation(len(self._inputs))
        self._inputs = self._inputs[p]
        self._outputs = self._outputs[p]

    def next_batch(self, batch_size):
        end_indicator = self._indicator + batch_size
        # 当获取的指针超过数据集大小时，归0
        if end_indicator > len(self._inputs):
            self._random_shuffle()
            self._indicator = 0
            end_indicator = batch_size
        # 还归0超过就报错
        if end_indicator > len(self._inputs):
            raise Exception("batch_size: %d is too large" % batch_size)

        batch_inputs = self._inputs[self._indicator: end_indicator]
        batch_outputs = self._outputs[self._indicator: end_indicator]
        self._indicator = end_indicator
        return batch_inputs, batch_outputs

train_dataset = TextDataSet(
    train_file, vocab, category_vocab, hps.num_timesteps)
val_dataset = TextDataSet(
    val_file, vocab, category_vocab, hps.num_timesteps)
test_dataset = TextDataSet(
    test_file, vocab, category_vocab, hps.num_timesteps)

print(train_dataset.next_batch(2))
print(val_dataset.next_batch(2))
print(test_dataset.next_batch(2))


def create_model(hps, vocab_size, num_classes):
    num_timesteps = hps.num_timesteps
    batch_size = hps.batch_size

    inputs = tf.placeholder(tf.int32, (batch_size, num_timesteps))
    outputs = tf.placeholder(tf.int32, (batch_size,))
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    global_step = tf.Variable(
        tf.zeros([], tf.int64), name='global_step', trainable=False)
    # embedding的值初始化到-1到1
    embedding_initializer = tf.random_uniform_initializer(-1.0, 1.0)
    with tf.variable_scope(
            'embedding', initializer=embedding_initializer):
        embeddings = tf.get_variable(
            'embedding',
            [vocab_size, hps.num_embedding_size],
            tf.float32)
        # [1, 10, 7] -> [embeddings[1], embeddings[10], embeddings[7]]
        # tf.nn.embedding_lookup函数的用法主要是选取一个张量里面索引对应的元素
        embed_inputs = tf.nn.embedding_lookup(embeddings, inputs)

    scale = 1.0 / math.sqrt(hps.num_embedding_size + hps.num_filters) / 3.0
    cnn_init = tf.random_uniform_initializer(-scale, scale)
    with tf.variable_scope('cnn', initializer=cnn_init):
        # embed_inputs: [batch_size, timesteps, embed_size]
        # conv1d: [batch_size, timesteps, num_filters]
        conv1d = tf.layers.conv1d(
            embed_inputs,
            hps.num_filters,
            hps.num_kernel_size,
            activation=tf.nn.relu,
        )
        # 取出最大值axis=0轴可以理解为行增长方向（向下）,axis=1轴可以理解为列增长方向(向右）
        global_maxpooling = tf.reduce_max(conv1d, axis=[1])

    """
    scale = 1.0 / math.sqrt(hps.num_embedding_size + hps.num_lstm_nodes[-1]) / 3.0
    lstm_init = tf.random_uniform_initializer(-scale, scale)
    with tf.variable_scope('lstm_nn', initializer = lstm_init):
        cells = []
        for i in range(hps.num_lstm_layers):
            cell = tf.contrib.rnn.BasicLSTMCell(
                hps.num_lstm_nodes[i],
                state_is_tuple = True)
            cell = tf.contrib.rnn.DropoutWrapper(
                cell,
                output_keep_prob = keep_prob)
            cells.append(cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells)

        initial_state = cell.zero_state(batch_size, tf.float32)
        # rnn_outputs: [batch_size, num_timesteps, lstm_outputs[-1]]
        rnn_outputs, _ = tf.nn.dynamic_rnn(
            cell, embed_inputs, initial_state = initial_state)
        last = rnn_outputs[:, -1, :]
    """

    fc_init = tf.uniform_unit_scaling_initializer(factor=1.0)
    with tf.variable_scope('fc', initializer=fc_init):
        fc1 = tf.layers.dense(global_maxpooling,
                              hps.num_fc_nodes,
                              activation=tf.nn.relu,
                              name='fc1')
        fc1_dropout = tf.contrib.layers.dropout(fc1, keep_prob)
        logits = tf.layers.dense(fc1_dropout,
                                 num_classes,
                                 name='fc2')

    with tf.name_scope('metrics'):
        softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=outputs)
        loss = tf.reduce_mean(softmax_loss)
        # [0, 1, 5, 4, 2] -> argmax: 2
        y_pred = tf.argmax(tf.nn.softmax(logits),
                           1,
                           output_type=tf.int32)
        correct_pred = tf.equal(outputs, y_pred)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    with tf.name_scope('train_op'):
        train_op = tf.train.AdamOptimizer(hps.learning_rate).minimize(
            loss, global_step=global_step)
        """
        tvars = tf.trainable_variables()
        for var in tvars:
            tf.logging.info('variable name: %s' % (var.name))
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(loss, tvars), hps.clip_lstm_grads)
        optimizer = tf.train.AdamOptimizer(hps.learning_rate)
        train_op = optimizer.apply_gradients(
            zip(grads, tvars), global_step = global_step)
        """

    return ((inputs, outputs, keep_prob),
            (loss, accuracy),
            (train_op, global_step))


placeholders, metrics, others = create_model(
    hps, vocab_size, num_classes)

inputs, outputs, keep_prob = placeholders
loss, accuracy = metrics
train_op, global_step = others

init_op = tf.global_variables_initializer()
train_keep_prob_value = 0.8
test_keep_prob_value = 1.0

num_train_steps = 10000

# Train: 100%
# Valid: 95.7%
# Test:  95.3%
with tf.Session() as sess:
    sess.run(init_op)
    for i in range(num_train_steps):
        batch_inputs, batch_labels = train_dataset.next_batch(
            hps.batch_size)
        outputs_val = sess.run([loss, accuracy, train_op, global_step],
                               feed_dict = {
                                   inputs: batch_inputs,
                                   outputs: batch_labels,
                                   keep_prob: train_keep_prob_value,
                               })
        loss_val, accuracy_val, _, global_step_val = outputs_val
        if global_step_val % 20 == 0:
            tf.logging.info("Step: %5d, loss: %3.3f, accuracy: %3.3f"
                            % (global_step_val, loss_val, accuracy_val))
