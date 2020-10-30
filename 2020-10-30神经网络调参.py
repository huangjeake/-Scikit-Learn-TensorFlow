
import tensorflow as tf
# tf.nn.sigmoid
# tf.nn.relu
# tf.glorot_uniform_initializer()
# tf.truncated_normal_initializer()
# tf.keras.initializers.he_normal()
# tf.train.AdamOptimizer
# tf.train.MomentumOptimizer
# tf.train.GradientDescentOptimizer
# saver = tf.train.Saver()
# saver.save()
# saver.restore()

'''
Tensorboard实战：
    # Create a summary to monitor cost tensor
    tf.summary.scalar("loss", loss)
    # Create a summary to monitor accuracy tensor
    tf.summary.scalar("accuracy", acc)
    # Create summaries to visualize weights
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var)
    # Summarize all gradients
    for grad, var in grads:
        tf.summary.histogram(var.name + '/gradient', grad)
    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(logs_path,
                                            graph=tf.get_default_graph())
    _, c, summary = sess.run([apply_grads, loss, merged_summary_op],
                                     feed_dict={x: batch_xs, y: batch_ys})
            # Write logs at every iteration
            summary_writer.add_summary(summary, epoch * total_batch + i)
    终端输入 tensorboard --logdir=./log
fine_tune实战:
    保存模型：1.saver = tf.train.Saver() saver.save(sess, 模型路径) 2.joblib
    模型加载：1.saver.restore(sess, 模型路径)2.joblib
    冻结神经网络层： 卷积层trainable设置为False，不进行训练，保存原始参数
                        
activation-initializer-optimizer：
    tf.nn.sigmoid# s激活函数
    tf.nn.relu# 修正线性单元
    tf.glorot_uniform_initializer
    tf.truncated_normal_initializer
    tf.keras.initializers.he_normal
    tf.train.AdamOptimizer
    tf.train.MomentumOptimizer
    tf.train.GradientDescentOptimizer
        
图像增强实战：

批量归一化：
    

'''