#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv

import argparse
#set argparser
parser = argparse.ArgumentParser()
parser.add_argument("--dataPath")
parser.add_argument("--modelPath")
parser.add_argument("--outputPath")
parser.add_argument("--resultfile")
arguments = parser.parse_args()
dataPath = arguments.dataPath
print("dataPath: ",dataPath)  
modelPath = arguments.modelPath
print("modelPath: ",modelPath)  
outputPath = arguments.outputPath
print("outputPath: ",outputPath) 
resultfile = arguments.resultfile
print("resultfile: ",resultfile)

fo = open(resultfile, "r+")
result = fo.read()
# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the positive data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", result, "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    x_raw = data_helpers.load_AI100_data(dataPath)
    #y_test = np.argmax(y_test, axis=1)
    y_test = [1, 0]
else:
    x_s = ["经工商登记机关核准，本公司的经营范围包括：许可经营项目：面向“三农”发放贷款、提供融资性担保、开展金融机构业务代理以及其他业务。公司主要业务包括：面向“三农”发放小额贷款业务、应付款保函业务以及提供融资性担保业务。", "公司主营业务系为住宅、商业、写字楼、工业园区等业态项目提供专业的物业服务与经营服务。"]
    x_raw = data_helpers.load_AI100_data_list(x_s)
    y_test = [1, 0]

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])
        with open(outputPath, 'w') as f:
            for i, pre in enumerate(all_predictions):
                f.write(str(i+1))
                f.write(',')
                f.write(str(int(pre) + 1))
                f.write('\n')

# # Print accuracy if y_test is defined
# if y_test is not None:
#     correct_predictions = float(sum(all_predictions == y_test))
#     print("Total number of test examples: {}".format(len(y_test)))
#     print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
#
# # Save the evaluation to a csv
# predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
# out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
# print("Saving evaluation to {0}".format(out_path))
# with open(out_path, 'w') as f:
#     csv.writer(f).writerows(predictions_human_readable)