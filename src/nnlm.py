import argparse
import datetime
import math
import time
import numpy as np
import tensorflow as tf
import os

from preprocessing import TextLoader


def main():
    os.chdir("/home/wusimpl/pycharm/project1/src")
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/',
                        help='data directory containing input.txt')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='mini batch size')  # feed 120 words per time from the whole dataset.
    parser.add_argument('--win_size', type=int, default=5,
                        help='context sequence length')
    parser.add_argument('--hidden_num', type=int, default=100,
                        help='number of hidden layers')  # 隐藏层神经元数量
    parser.add_argument('--word_dim', type=int, default=30,
                        help='number of word embedding')  # how many numbers in a vector to represent a word
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of epochs')  # how many times does the training apply over the whole dataset
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')

    args = parser.parse_args()
    args_msg = '\n'.join([arg + ': ' + str(getattr(args, arg)) for arg in vars(args)])

    data_loader = TextLoader(args.data_dir, args.batch_size, args.win_size)
    args.vocab_size = data_loader.vocab_size

    graph = tf.Graph()
    with graph.as_default():
        input_data = tf.placeholder(tf.int64, [args.batch_size, args.win_size])  # shape为(batch,windows)大小的矩阵
        targets = tf.placeholder(tf.int64, [args.batch_size, 1])

        '''
        注意，tf.variable() 和tf.get_variable()有不同的创建变量的方式：tf.Variable() 每次都会新建变量。
        如果希望重用（共享）一些变量，就需要用到了get_variable()，它会去搜索变量名，有就直接用，没有再新建。
        '''
        with tf.variable_scope('nnlm' + 'embedding'):
            # 使用随机正态分布生成一个[vocab_size, word_dim]大小的矩阵，对应论文的矩阵C
            embeddings = tf.Variable(tf.random_uniform([args.vocab_size, args.word_dim], -1.0, 1.0))
            # dim=0：按列进行L2范化 dim=1：按行 （为什么要进行范化？）
            embeddings = tf.nn.l2_normalize(embeddings, 1)

        with tf.variable_scope('nnlm' + 'weight'):
            weight_h = tf.Variable(tf.truncated_normal([args.win_size * args.word_dim, args.hidden_num],  # [1500,100]
                                                       stddev=1.0 / math.sqrt(args.hidden_num)))
            softmax_w = tf.Variable(tf.truncated_normal([args.win_size * args.word_dim, args.vocab_size],  # [1500,27022]
                                                        stddev=1.0 / math.sqrt(args.win_size * args.word_dim)))
            softmax_u = tf.Variable(tf.truncated_normal([args.hidden_num, args.vocab_size],  # [100,27022]
                                                        stddev=1.0 / math.sqrt(args.hidden_num)))  # stddev:标准差

            b_1 = tf.Variable(tf.random_normal([args.hidden_num]))
            b_2 = tf.Variable(tf.random_normal([args.vocab_size]))

        # 构建NNLM
        def infer_output(input_data):
            """
            hidden = tanh(x * H + b_1)
            output = softmax(x * W + hidden * U + b_2)
            """
            input_data_emb = tf.nn.embedding_lookup(embeddings, input_data)  # 取出对应词的嵌入表示向量
            input_data_emb = tf.reshape(input_data_emb, [-1, args.win_size * args.word_dim])
            hidden = tf.tanh(tf.matmul(input_data_emb, weight_h)) + b_1

            # U*tanh(?)+b + wx
            hidden_output = tf.matmul(hidden, softmax_u) + tf.matmul(input_data_emb, softmax_w) + b_2
            output = tf.nn.softmax(hidden_output)
            return output

        outputs = infer_output(input_data)

        # squeeze:删除维度是1的轴 depth:向量长度 on_value off_value
        one_hot_targets = tf.one_hot(tf.squeeze(targets), args.vocab_size, 1.0, 0.0)
        loss = -tf.reduce_mean(tf.reduce_sum(tf.log(outputs) * one_hot_targets, 1))  # reduce_sum(tensor,1) 按行求和
        # Clip grad.
        optimizer = tf.train.AdagradOptimizer(0.1)
        gvs = optimizer.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_value(grad, -args.grad_clip, args.grad_clip), var) for grad, var in gvs]
        optimizer = optimizer.apply_gradients(capped_gvs)

        embeddings_norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / embeddings_norm
    processing_message_lst = list()

    run_log_file = open("{}.txt".format('logs/run_log'), 'w', encoding='utf-8')
    with tf.Session(graph=graph) as sess:
        log_writer = tf.summary.FileWriter("./logs/", sess.graph)
        tf.global_variables_initializer().run()
        for e in range(args.num_epochs):  # 总共跑几个epoch就是几次循环
            data_loader.reset_batch_pointer()
            for b in range(data_loader.num_batches):  # 多个batch组成一个epoch
                start = time.time()
                x, y = data_loader.next_batch()
                feed = {input_data: x, targets: y}
                train_loss, _ = sess.run([loss, optimizer], feed)
                end = time.time()

                processing_message = "{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}".format(
                    b, data_loader.num_batches,
                    e, train_loss, end - start)

                run_log_file.write(processing_message+'\n')
                run_log_file.flush()
                print(processing_message)
                processing_message_lst.append(processing_message)
                # print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}".format(
                #     b, data_loader.num_batches,
            #     e, train_loss, end - start))

            np.save('data/nnlm_word_embeddings', normalized_embeddings.eval())
        log_writer.close()
    # record training processing

    local_time = str(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
    run_log_file.close()


if __name__ == '__main__':
    main()
