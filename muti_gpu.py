
# ... statistic -> prune -> train(multi gpu) -> evaluate(multi gpu) -> statistic ...

import numpy as np
import tensorflow as tf
from XavierLib.mobilenet_v2 import mobilenet_v2_224
import XavierLib.core.crelindr_lib as cl
import XavierLib.core.xavier_lib as ex
import os
from XavierLib.inception_preprocessing import preprocess_image, preprocess_for_train, preprocess_for_eval
from XavierLib.imagenet_label import *
import threading
from from_slim import sum_clones_gradients
import time

SOURCE_VARIABLES = {}

START = '/home/dl/DATA/ImageNet'# '/media/xavierliu/Seagate Backup Plus Drive'

# Train Config
GPU = 4
EPOCHS = 20
BATCH_SIZE = 32 * GPU
DATA_PATH = '/home/dl/DATA/ImageNet/ILSVRC/Data/CLS-LOC'

train_size = 1281167
validation_size = 50000
CONVERTER = wnid_to_2015(START+'/imagenet_lsvrc_2015_synsets.txt')

with tf.name_scope('read_data'):

    x = tf.placeholder(tf.float32, [None, None, 3], name='Images')
    y = tf.placeholder(tf.float32, [1001], name='Label')
    train_flag = tf.placeholder(tf.bool, name='train_flag')

    x_train_preporcess = preprocess_for_train(x, 224, 224, None, True, random_crop=True)

    x_val_preporcess = preprocess_for_eval(x, 224, 224, central_crop=True)

    multi = tf.cast(train_flag, tf.float32)
    x_preporcess = multi*x_train_preporcess+(1-multi)*x_val_preporcess

    x_preporcess = tf.reshape(x_preporcess, [1, 224, 224, 3])
    y_ = tf.reshape(y, [1, 1001])

    with tf.device('/cpu:0'):
        q = tf.FIFOQueue(BATCH_SIZE * 3, [tf.float32, tf.float32], shapes=[[224, 224, 3], [1001]])
        enqueue_op = q.enqueue_many([x_preporcess, y_])
        input_list = []
        for i in range(GPU):
            x_b, y_b = q.dequeue_many(BATCH_SIZE)
            input_list.append([x_b, y_b])

loss_list = []
acc_list = []
for i in range(GPU):
    [x_b, y_b] = input_list[i]
    if i == 0:
        with tf.device('/gpu:'+str(i)):
            with tf.variable_scope('MobilenetV2'):

                global_step = tf.get_variable('step', [], tf.float32, tf.zeros_initializer(), trainable=False)

                statistic_builder = cl.statistic_flow_builder()
                score_builder = cl.score_update_flow_builder()
                gradient_builder = cl.average_gradients_flow_builder()
                ema_builder = ex.exponential_moving_average_builder()

                out, restore_var = mobilenet_v2_224(x_b, train_flag, statistic_builder, score_builder, gradient_builder,
                                                    EMA=ema_builder)
                SOURCE_VARIABLES['model_var'] = restore_var.copy()
                SOURCE_VARIABLES['mask_var'] = score_builder.mask_list.copy()
                SOURCE_VARIABLES['var_in_statistic'] = statistic_builder.get_source_variables().copy()
                SOURCE_VARIABLES['EMA'] = ema_builder.moving_avg_list

                # Only consider the first gpu// Really?
                BN_UPDATE = tf.get_collection('bn_update_op_list').copy()
                EMA_UPDATE = ema_builder.update_avg_ops
                EMA_TAKE = ema_builder.set_avg_to_var_ops

                STATISTIC_UPDATE = []
                STATISTIC_UPDATE += statistic_builder.get_update_op().copy()
                STATISTIC_UPDATE += gradient_builder.get_update_op_list().copy()

            with tf.name_scope('cross_entropy_'+str(i)):
                cross_entropy = tf.reduce_mean(
                    tf.losses.softmax_cross_entropy(y_b, out, label_smoothing=0.1, weights=0.4))

            with tf.name_scope('total_loss_'+str(i)):
                total_loss = cross_entropy + tf.add_n(tf.losses.get_regularization_losses())
                loss_list.append(tf.div(total_loss, 1.0 * GPU))

            with tf.name_scope('accuracy_'+str(i)):
                correct = tf.equal(tf.argmax(out, 1), tf.argmax(y_b, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
                acc_list.append(accuracy)

    else:
        with tf.device('/gpu:'+str(i)):
            with tf.variable_scope('MobilenetV2', reuse=True):
                out, restore_var = mobilenet_v2_224(x_b, train_flag, statistic_builder, score_builder, gradient_builder,
                                                    EMA=None)

            with tf.name_scope('cross_entropy_'+str(i)):
                cross_entropy = tf.reduce_mean(
                    tf.losses.softmax_cross_entropy(y_b, out, label_smoothing=0.1, weights=0.4))

            with tf.name_scope('total_loss_'+str(i)):
                total_loss = cross_entropy + tf.add_n(tf.losses.get_regularization_losses())
                loss_list.append(tf.div(total_loss, 1.0 * GPU))

            with tf.name_scope('accuracy_'+str(i)):
                correct = tf.equal(tf.argmax(out, 1), tf.argmax(y_b, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
                acc_list.append(accuracy)


optimizer = tf.train.RMSPropOptimizer(0.03, 0.9, 0.9)

grad_var_list = []
for i, a_loss in enumerate(loss_list):
    with tf.device('/gpu:'+str(i)):
        grad_var_list.append(optimizer.compute_gradients(a_loss))

grads_and_vars = sum_clones_gradients(grad_var_list)

GRAD_UPDATE = optimizer.apply_gradients(grads_and_vars, global_step)

coord = tf.train.Coordinator()

to_restore = tf.train.Saver(SOURCE_VARIABLES['model_var'])
all_from_ori = tf.train.Saver(SOURCE_VARIABLES['EMA'] + SOURCE_VARIABLES['model_var'])
all_source = SOURCE_VARIABLES['model_var']+SOURCE_VARIABLES['mask_var']

all_var = tf.train.Saver(all_source)

for var in tf.global_variables():
    if var not in all_source and var not in SOURCE_VARIABLES['var_in_statistic']:
        if 'ExponentialMovingAverage' not in var.name and 'RMSProp' not in var.name:
            print(var)


with tf.Session() as sess:

    def initialize():
        sess.run(tf.global_variables_initializer())
        all_from_ori.restore(sess, './mobilenetv2_ckp/mobilenet_v2_1.4_224.ckpt')
        sess.run(EMA_TAKE)
        all_var.save(sess, './ckpt_731/snas.ckpt')
        with open('./ckpt_731/log.txt', 'w') as f:
            f.write('Initialized.\n')

    def statistic_and_update_mask():
        def enqueue_train_batches():
            while not coord.should_stop():
                im, l = read_one(os.path.join(DATA_PATH, 'train'), CONVERTER)
                sess.run(enqueue_op, feed_dict={x: im, y: l, train_flag:False})

        num_threads = 40
        tasks = []
        for i in range(num_threads):
            t = threading.Thread(target=enqueue_train_batches)
            t.setDaemon(True)
            t.start()
            tasks.append(t)

        sess.run(tf.global_variables_initializer())

        all_var.restore(sess, './ckpt_731/snas.ckpt')

        # show the number of channels before pruning
        pre_num = score_builder.remaining_activation(sess)
        print('The number of channels before pruning: '+str(pre_num))

        # take ema
        # sess.run(EMA_TAKE)

        for i in range(int(train_size/BATCH_SIZE*GPU)):
            sess.run(STATISTIC_UPDATE, feed_dict={train_flag: False})
            print('statistic: ', i, 'of', int(train_size/BATCH_SIZE*GPU))

        coord.request_stop()

        score_builder.compute_score_and_remove_last(sess)
        npy_mask = np.concatenate(sess.run(score_builder.mask_list))
        log_time = time.strftime('%b-%d-%Y-%H-%M-%S')
        np.save('./ckpt_731/'+log_time+'.npy', npy_mask)
        cur_num = score_builder.remaining_activation(sess)
        print('pruned number: ', pre_num - cur_num)

        with open('./ckpt_731/log.txt', 'a') as file:
            file.write(log_time+':')

        all_var.save(sess, './ckpt_731/snas.ckpt')
        print('Finish saving.')

        for t in tasks:
            t.join()

        print('Finish statistic.')


    def train_step():
        def enqueue_train_batches():
            while not coord.should_stop():
                im, l = read_one(os.path.join(DATA_PATH, 'train'), CONVERTER)
                sess.run(enqueue_op, feed_dict={x: im, y: l, train_flag: True})

        num_threads = 40
        tasks = []
        for i in range(num_threads):
            t = threading.Thread(target=enqueue_train_batches)
            t.setDaemon(True)
            t.start()
            tasks.append(t)

        all_var.restore(sess, './ckpt_731/snas.ckpt')
        iteration = 1000
        for i in range(iteration):
            sess.run([GRAD_UPDATE, BN_UPDATE], feed_dict={train_flag: True})
            sess.run(EMA_UPDATE)

        sess.run(EMA_TAKE)
        all_var.save(sess, './ckpt_731/snas.ckpt')


    def evaluate_step():
        reader = validation_set_reader(os.path.join(DATA_PATH, 'val'),
                                       '/home/dl/DATA/ImageNet/imagenet_2012_validation_synset_labels.txt',
                                       CONVERTER)

        def enqueue_batches():
            while not coord.should_stop():
                val_im, val_cls = reader.read_validation_one()
                sess.run(enqueue_op, feed_dict={x: val_im, y: val_cls, train_flag: False})

        num_threads = 40
        for i in range(num_threads):
            t = threading.Thread(target=enqueue_batches)
            t.setDaemon(True)
            t.start()

        all_var.restore(sess, './ckpt_731/snas.ckpt')

        summ = 0
        count = 0
        for i in range(int(validation_size/BATCH_SIZE)):
            acc = sess.run(acc_list, feed_dict={train_flag: False})
            summ += sum(acc)
            count += GPU
        all_acc = summ/count
        print(all_acc)

        with open('./ckpt_731/log.txt', 'a') as file:
            file.write(str(all_acc)+'\n')

    statistic_and_update_mask()
