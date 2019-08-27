import numpy as np
import tensorflow as tf
from XavierLib.mobilenet_v2 import mobilenet_v2_224
import XavierLib.core.crelindr_lib as cl
import os
from XavierLib.inception_preprocessing import preprocess_image
from XavierLib.imagenet_label import *
import threading


START = 'D:'# '/media/xavierliu/Seagate Backup Plus Drive'

# Config
EPOCHS = 20
BATCH_SIZE = 32
DATA_PATH = START + '/train_data/ImageNet'
STATE = 'prune'
# Done config

if STATE == 'train':
    TRAIN = True
    PRUNE = False
elif STATE == 'prune':
    TRAIN = False
    PRUNE = True
else:
    TRAIN = False
    PRUNE = False

train_size = 1281167
validation_size = 50000

converter = wnid_to_2015(START+'/train_data/ImageNet/data/imagenet_lsvrc_2015_synsets.txt')

with tf.name_scope('read_data'):

    x = tf.placeholder(tf.float32, [None, None, 3])
    y = tf.placeholder(tf.float32, [1001])

    x_preporcess = tf.reshape(preprocess_image(x, 224, 224, TRAIN), [1, 224,224, 3])
    y_ = tf.reshape(y, [1, 1001])

    train_flag = tf.placeholder(tf.bool)

    with tf.device('/cpu:0'):
        q = tf.FIFOQueue(BATCH_SIZE * 3, [tf.float32, tf.float32], shapes=[[224, 224, 3], [1001]])
        enqueue_op = q.enqueue_many([x_preporcess, y_])
        x_b, y_b = q.dequeue_many(BATCH_SIZE)

statistic_builder = cl.statistic_flow_builder()
score_builder = cl.score_update_flow_builder()
gradient_builder = cl.average_gradients_flow_builder()

out, restore_var = mobilenet_v2_224(x_b, train_flag, statistic_builder, score_builder, gradient_builder)

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(
        tf.losses.softmax_cross_entropy(y_b, out, label_smoothing=0.1))

with tf.name_scope('prune_statistic'):
    gradient_builder.set_y(cross_entropy)
    gradient_builder.build_flow_on_conv()
    score_builder.set_gard_var_wise_list(gradient_builder.post_process_and_get_tensor_list())
    statistic_update_op_list = []
    statistic_update_op_list += statistic_builder.get_update_op()
    statistic_update_op_list += gradient_builder.get_update_op_list()

    statistic_reset_op_list = []
    statistic_reset_op_list += statistic_builder.get_reset_op()
    statistic_reset_op_list += gradient_builder.get_reset_op()

with tf.name_scope('total_loss'):
    total_loss = cross_entropy + tf.add_n(tf.losses.get_regularization_losses())

with tf.name_scope('optimizer'):
    train = tf.train.RMSPropOptimizer(0.03, 0.9, 0.9)

with tf.name_scope('accuracy'):
    correct = tf.equal(tf.argmax(out, 1), tf.argmax(y_b, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

to_restore = tf.train.Saver(restore_var)

coord = tf.train.Coordinator()

graph = tf.get_default_graph()
summaryWriter = tf.summary.FileWriter('log/', graph)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    to_restore.restore(sess, './mobilenetv2_ckp/mobilenet_v2_1.4_224.ckpt')

    if (TRAIN) or (PRUNE):
        def enqueue_batches():
            while not coord.should_stop():
                im, l = read_one(os.path.join(DATA_PATH, 'train'), converter)
                sess.run(enqueue_op, feed_dict={x: im, y: l})

        num_threads = 1
        for i in range(num_threads):
            t = threading.Thread(target=enqueue_batches)
            t.setDaemon(True)
            t.start()
    else:
        sum = 0
        count = 0
        reader = validation_set_reader(os.path.join(DATA_PATH, 'val'),
                                       os.path.join(DATA_PATH,'data/imagenet_2012_validation_synset_labels.txt'),
                                       converter)

        def enqueue_batches():
            while not coord.should_stop():
                val_im, val_cls = reader.read_validation_one()
                sess.run(enqueue_op, feed_dict={x: val_im, y: val_cls})


        num_threads = 20
        for i in range(num_threads):
            t = threading.Thread(target=enqueue_batches)
            t.setDaemon(True)
            t.start()

    sess.run(tf.get_collection('set_ema_to_para_op_list'))

    if TRAIN:
        for i in range(int(train_size*EPOCHS/BATCH_SIZE)):
            pass
    elif PRUNE:
        print(score_builder.remaining_activation(sess))
        for i in range(10):
            sess.run(statistic_update_op_list, feed_dict={train_flag: False})
            print('statistic:', i)
        coord.request_stop()

        def check_3(ll):
            sum = 0
            for l in ll:
                sum += np.sum(np.sign(l) - 1)
            return sum


        #import scipy.io as sio
        #recc = sess.run(tf.get_collection('show'))
        #reccc = {}
        #for i in range(len(recc)):
        #    reccc['rec' + str(i)] = recc[i]
        #    print(recc[i])
        #sio.savemat('showww', reccc)

        score_builder.compute_score_and_remove_last(sess)
        print(score_builder.remaining_activation(sess))
    else:
        for i in range(int(validation_size/BATCH_SIZE)):
            acc = sess.run(accuracy, feed_dict={train_flag: False})
            sum += acc
            count += 1
            print(sum/count)
