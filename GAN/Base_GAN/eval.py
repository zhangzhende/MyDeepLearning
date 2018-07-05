# -*- coding: utf-8 -*-
import tensorflow as tf
import os

from MyDeepLearning.GAN.Base_GAN.read_data import *
from MyDeepLearning.GAN.Base_GAN.utils import *
from MyDeepLearning.GAN.Base_GAN.ops import *
from MyDeepLearning.GAN.Base_GAN.model import *
from MyDeepLearning.GAN.Base_GAN.model import BATCH_SIZE


def eval():
    # 用于存放测试图片
    test_dir = './eval/'
    # 从此处加载模型
    checkpoint_dir = './logs/'

    y = tf.placeholder(tf.float32, [BATCH_SIZE, 10], name='y')
    z = tf.placeholder(tf.float32, [None, 100], name='z')

    G = generator(z, y)
    data_x, data_y = read_data()
    sample_z = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
    sample_labels = data_y[120: 184]

    # 读取 ckpt 需要 sess，saver
    print("Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

    # saver
    saver = tf.train.Saver(tf.all_variables())

    # sess
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    sess = tf.InteractiveSession(config=config)

    # 从保存的模型中恢复变量
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))

    # 用恢复的变量进行生成器的测试
    test_sess = sess.run(G, feed_dict={z: sample_z, y: sample_labels})

    # 保存测试的生成器图片到特定文件夹
    save_images(test_sess, [8, 8], test_dir + 'test_%d.png' % 500)

    sess.close()


if __name__ == '__main__':
    eval()