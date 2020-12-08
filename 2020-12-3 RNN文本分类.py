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
train_file = r'C:\Users\ext_renqq\Desktop\文本分类数据\cnews.train.txt'
val_file = r'C:\Users\ext_renqq\Desktop\文本分类数据\cnews.val.txt'
test_file = r'C:\Users\ext_renqq\Desktop\文本分类数据\cnews.test.txt'

# output files
seg_train_file = r'C:\Users\ext_renqq\Desktop\文本分类数据\cnews.train.seg.txt'
seg_val_file = r'C:\Users\ext_renqq\Desktop\文本分类数据\cnews.val.seg.txt'
seg_test_file = r'C:\Users\ext_renqq\Desktop\文本分类数据\cnews.test.seg.txt'

vocab_file = r'C:\Users\ext_renqq\Desktop\文本分类数据\cnews.vocab.txt'
category_file = r'C:\Users\ext_renqq\Desktop\文本分类数据\cnews.category.txt'

with open(val_file, 'rb') as f:
    lines = f.readlines()
    # for line in lines:
    #print(lines[0].decode('utf-8'), lines[0].decode('utf-8').strip('\r\n'), lines[0].decode('utf-8').strip('\r\n').split('\t'))


label, content = lines[0].decode('utf-8').strip('\r\n').split('\t')
word_iter = jieba.cut(content)
# print(word_iter)


def generate_seg_file(input_file, output_seg_file):
    """Segment the sentences in each line in input_file"""
    with open(input_file, 'rb') as f:
        lines = f.readlines()
    with open(output_seg_file, 'w', encoding='utf-8') as f:
        for line in lines:
            label, content = line.decode('utf-8').strip('\r\n').split('\t')
            word_iter = jieba.cut(content)
            word_content = ''
            for word in word_iter:
                # print('word', word)
                word = word.strip(' ')
                # print('word', word)
                if word != '':
                    word_content += word + ' '
            out_line = '%s\t%s\n' % (label, word_content.strip(' '))
            f.write(out_line)

generate_seg_file(train_file, seg_train_file)
generate_seg_file(val_file, seg_val_file)
generate_seg_file(test_file, seg_test_file)


def generate_vocab_file(input_seg_file, output_vocab_file):
    with open(input_seg_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    word_dict = {}
    for line in lines:
        label, content = line.strip('\r\n').split('\t')
        for word in content.split():
            word_dict.setdefault(word, 0)
            word_dict[word] += 1
    # [(word, frequency), ..., ()]
    # print('111', word_dict)
    sorted_word_dict = sorted(
        word_dict.items(), key=lambda d: d[1], reverse=True)
    # print('111',sorted_word_dict)
    with open(output_vocab_file, 'w', encoding='utf-8') as f:
        f.write('<UNK>\t10000000\n')
        for item in sorted_word_dict:
            # print('item', item, item[1])
            # print(type(item))
            f.write('%s\t%d\n' % (item[0], item[1]))


generate_vocab_file(seg_train_file, vocab_file)


def generate_category_dict(input_file, category_file):
    with open(input_file, 'rb') as f:
        lines = f.readlines()
    category_dict = {}
    for line in lines:
        # print(line.decode('utf-8'))
        label, content = line.decode('utf-8').strip('\r\n').split('\t')
        category_dict.setdefault(label, 0)
        category_dict[label] += 1
    category_number = len(category_dict)
    print('category_dict', category_dict)
    with open(category_file, 'w', encoding='utf-8') as f:
        for category in category_dict:
            line = '%s\n' % category
            print('%s\t%d' % (
                category, category_dict[category]))
            f.write(line)


generate_category_dict(train_file, category_file)

