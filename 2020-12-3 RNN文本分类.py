'''
文本分类数据集：
https://pan.baidu.com/s/1hugrfRu 密码：qfud
参考网址：
https://blog.csdn.net/qq_36047533/article/details/88094385
https://blog.csdn.net/weixin_40931845/article/details/83865877
'''

# 分词
# 词语 -> id
#   matrix -> [|V|, embed_size]
#   词语A -> id(5)
#   词表

# label -> id

import sys
import os
import jieba # pip install jieba

# input files
train_file = '../../text_classification_data/cnews.train.txt'
val_file = '../../text_classification_data/cnews.val.txt'
test_file = '../../text_classification_data/cnews.test.txt'

# output files
seg_train_file = '../../text_classification_data/cnews.train.seg.txt'
seg_val_file = '../../text_classification_data/cnews.val.seg.txt'
seg_test_file = '../../text_classification_data/cnews.test.seg.txt'

vocab_file = '../../text_classification_data/cnews.vocab.txt'
category_file = '../../text_classification_data/cnews.category.txt'

with open(val_file, 'r') as f:
    lines = f.readlines()

label, content = lines[0].decode('utf-8').strip('\r\n').split('\t')
word_iter = jieba.cut(content)


def generate_seg_file(input_file, output_seg_file):
    """Segment the sentences in each line in input_file"""
    with open(input_file, 'r') as f:
        lines = f.readlines()
    with open(output_seg_file, 'w') as f:
        for line in lines:
            label, content = line.decode('utf-8').strip('\r\n').split('\t')
            word_iter = jieba.cut(content)
            word_content = ''
            for word in word_iter:
                word = word.strip(' ')
                if word != '':
                    word_content += word + ' '
            out_line = '%s\t%s\n' % (label, word_content.strip(' '))
            f.write(out_line.encode('utf-8'))

generate_seg_file(train_file, seg_train_file)
generate_seg_file(val_file, seg_val_file)
generate_seg_file(test_file, seg_test_file)


def generate_vocab_file(input_seg_file, output_vocab_file):
    with open(input_seg_file, 'r') as f:
        lines = f.readlines()
    word_dict = {}
    for line in lines:
        label, content = line.strip('\r\n').decode('utf-8').split('\t')
        for word in content.split():
            word_dict.setdefault(word, 0)
            word_dict[word] += 1
    # [(word, frequency), ..., ()]
    sorted_word_dict = sorted(
        word_dict.items(), key=lambda d: d[1], reverse=True)
    with open(output_vocab_file, 'w') as f:
        f.write('<UNK>\t10000000\n')
        for item in sorted_word_dict:
            f.write('%s\t%d\n' % (item[0].encode('utf-8'), item[1]))


generate_vocab_file(seg_train_file, vocab_file)


def generate_category_dict(input_file, category_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    category_dict = {}
    for line in lines:
        label, content = line.strip('\r\n').decode('utf-8').split('\t')
        category_dict.setdefault(label, 0)
        category_dict[label] += 1
    category_number = len(category_dict)
    with open(category_file, 'w') as f:
        for category in category_dict:
            line = '%s\n' % category.encode('utf-8')
            print
            '%s\t%d' % (
                category.encode('utf-8'), category_dict[category])
            f.write(line)


generate_category_dict(train_file, category_file)

