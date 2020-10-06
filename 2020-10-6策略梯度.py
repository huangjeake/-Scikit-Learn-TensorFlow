'''
策略梯度PG算法：
    Policy Gradient 算法就是对策略函数进行建模，然后用梯度下降更新网络的参数。但是在强化学习中并没有实际的损失函数，而 PG 算法
    的目的是最大化累计奖励的期望值，所以可以将损失函数设为：l o s s = − E [ log ⁡ [ π ( a ∣ s ) ] ⋅ Q ( s , a ) ] loss=
    -E[\log{[\pi(a|s)]}\cdot Q(s,a)]loss=−E[log[π(a∣s)]⋅Q(s,a)]，可以理解为如果一个动作的奖励值较大，则下次选取该动作的可能
    性增加的幅度也大，反之选取该动作的可能性增加的幅度小。

时间差分算法TD：
    时序差分算法是一种无模型的强化学习算法。它继承了动态规划(Dynamic Programming)和蒙特卡罗方法(Monte Carlo Methods)的优点，
    从而对状态值(state value)和策略(optimal policy)进行预测。
    态规划的backup操作是基于当前状态和下一个状态的reward，蒙特卡罗方法的backup是基于一个完整的episode的reward，而时序差分算法
    的backup是基于当前状态和下一个状态的reward

深度Q神经网络：
    重播存储器是可选的，但强烈推荐。没有它，可能需要非常 相关的连续经验训练评论家DQN。这将引入更多的误差，并减小训 练算法的收敛速
    度。通过使用重播内存，我们确保提供给训练算法的 记忆可以不相关

tf.multinomial(logits, num_samples) :
第一个参数logits可以是一个数组，每个元素的值表示对应index的选择概率。
假设logits有两个元素，即[0.6,0.4],这表示的意思是取 0 的概率是0.6， 取 1 的概率是0.4
第二个参数num_samples表示抽样的个数。
tf.multinomial(tf.log([[0.01]]),3) 不管重复运行多少次结果都是 [0,0,0]
tf.multinomial(tf.log([[0.1, 0.6]]),3) 结果可能 [0,0,0]，也可能是[0,1,1],当然也有其他可能。



'''
# PG算法
import tensorflow as tf
import numpy as np
import gym

# reset_graph()

n_inputs = 4
n_hidden = 4
n_outputs = 1

learning_rate = 0.01

initializer = tf.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, n_inputs])

hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
logits = tf.layers.dense(hidden, n_outputs)
outputs = tf.nn.sigmoid(logits)  # probability of action 0 (left)
p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])# 按列进行拼接
action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)

y = 1. - tf.to_float(action)
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)# 计算交叉熵
optimizer = tf.train.AdamOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(cross_entropy)# 并没有计算损失函数最小值，而是计算梯度
gradients = [grad for grad, variable in grads_and_vars]
gradient_placeholders = []
grads_and_vars_feed = []
for grad, variable in grads_and_vars:
    gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))
training_op = optimizer.apply_gradients(grads_and_vars_feed)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]
env = gym.make("CartPole-v0")

n_games_per_update = 10
n_max_steps = 1000
n_iterations = 250
save_iterations = 10
discount_rate = 0.95

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        print("\rIteration: {}".format(iteration), end="")
        all_rewards = []
        all_gradients = []
        for game in range(n_games_per_update):
            current_rewards = []
            current_gradients = []
            obs = env.reset()
            for step in range(n_max_steps):
                action_val, gradients_val = sess.run([action, gradients], feed_dict={X: obs.reshape(1, n_inputs)})
                obs, reward, done, info = env.step(action_val[0][0])
                current_rewards.append(reward)
                current_gradients.append(gradients_val)
                if done:
                    break
            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)

        all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate=discount_rate)
        feed_dict = {}
        for var_index, gradient_placeholder in enumerate(gradient_placeholders):
            mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index]
                                      for game_index, rewards in enumerate(all_rewards)
                                          for step, reward in enumerate(rewards)], axis=0)
            feed_dict[gradient_placeholder] = mean_gradients
        sess.run(training_op, feed_dict=feed_dict)
        if iteration % save_iterations == 0:
            saver.save(sess, "./my_policy_net_pg.ckpt")