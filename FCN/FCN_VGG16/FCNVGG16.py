import os
from matplotlib import pyplot as plt
import numpy as np
from skimage import io
from numpy import ogrid, repeat, newaxis
import tensorflow as tf
from myProject.tensorflowDemo.slimAdvance import configure
import myProject.tensorflowDemo.slimAdvance.datasets.imagenet_classes  as imagenet_classes
from tensorflow.contrib.slim.nets import vgg
import tensorflow.contrib.slim as slim
from myProject.tensorflowDemo.slimAdvance.preprocessing .vgg_preprocessing import (_mean_image_subtraction,_R_MEAN,_G_MEAN,_B_MEAN)

"""
全卷积神经网络--使用VGG16模型进行图像识别
"""
# 确定卷积核大小
def getKernelSize(factor):
    return 2 * factor - factor % 2


def upsampleFilt(size):
    factor = (size + 1) // 2  # 整除
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)


# 进行upsampling 卷积
def bilinear_upsample_weight(factor, number_of_class):
    filter_size = getKernelSize(factor=factor)
    weights = np.zeros((
        filter_size,
        filter_size,
        number_of_class,
        number_of_class
    ), dtype=np.float32)
    upsample_kernel = upsampleFilt(filter_size)
    for i in range(number_of_class):
        weights[:, :, i, i] = upsample_kernel
        # print(weights[:,:,i,i])
    return weights


def upsample_tf(factor,input_img):
    number_of_classes=input_img.shape[2]
    new_height=input_img.shape[0]*factor
    new_width=input_img.shape[1]*factor
    #拓展维度
    expanded_img=np.expand_dims(input_img,axis=0)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            upsample_filter_np=bilinear_upsample_weight(factor,number_of_classes)
            """
        除去name参数用以指定该操作的name，与方法有关的一共六个参数：
        第一个参数value：指需要做反卷积的输入图像，它要求是一个Tensor
        第二个参数filter：卷积核，它要求是一个Tensor，具有[filter_height, filter_width, out_channels, in_channels]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，卷积核个数，图像通道数]
        第三个参数output_shape：反卷积操作输出的shape，细心的同学会发现卷积操作是没有这个参数的，那这个参数在这里有什么用呢？下面会解释这个问题
        第四个参数strides：反卷积时在图像每一维的步长，这是一个一维的向量，长度4
        第五个参数padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式
        第六个参数data_format：string类型的量，'NHWC'和'NCHW'其中之一，这是tensorflow新版本中新加的参数，它说明了value参数的数据格式。'NHWC'指tensorflow标准的数据格式[batch, height, width, in_channels]，'NCHW'指Theano的数据格式,[batch, in_channels，height, width]，当然默认值是'NHWC'
        开始之前务必了解卷积的过程，参考我的另一篇文章：http://blog.csdn.net/mao_xiao_feng/article/details/53444333
        又一个很重要的部分！tf.nn.conv2d中的filter参数，是[filter_height, filter_width, in_channels, out_channels]的形式，
        而tf.nn.conv2d_transpose中的filter参数，是[filter_height, filter_width, out_channels，in_channels]的形式，
        注意in_channels和out_channels反过来了！因为两者互为反向，所以输入输出要调换位置
            """
            res=tf.nn.conv2d_transpose(expanded_img,upsample_filter_np,
                                       output_shape=[1,new_height,new_width,number_of_classes],
                                       strides=[1,factor,factor,1])
            res=sess.run(res)
    return np.squeeze(res)

def discrete_matshow(data,labels_names=[],title=""):
    fig_size=[7,6]
    plt.rcParams["figure.figsize"]=fig_size
    cmap=plt.get_cmap("Paired",np.max(data)-np.min(data)+1)
    mat=plt.matshow(data,cmap=cmap,vmin=np.min(data)-0.5,vmax=np.max(data)+0.5)
    cax=plt.colorbar(mat,ticks=np.arange(np.min(data),np.max(data)+1))
    if labels_names:
        cax.ax.set_yticklabels(labels_names)
    if title:
        plt.suptitle(title,fontsize=15,fontweight="bold")
        plt.show()

def imageBreak():
    imagepath="../../data/beauty/001/2.jpg"
    with tf.Graph().as_default():
        image=tf.image.decode_jpeg(tf.read_file(imagepath),channels=3)
        image=tf.image.resize_images(image,[224,224])
        #减去均值之前，将像素转为32位浮点数
        image_float=tf.to_float(image,name="ToFloat")
        #每个像素减去像素的均值
        processed_image=_mean_image_subtraction(image_float,[_R_MEAN,_G_MEAN,_B_MEAN])
        input_image=tf.expand_dims(processed_image,0)
        with slim.arg_scope(vgg.vgg_arg_scope()):
            logits,endpoints=vgg.vgg_16(inputs=input_image,
                                        num_classes=1000,
                                        is_training=False,
                                        spatial_squeeze=False)
            pred=tf.argmax(logits,dimension=3)#对输出层进行逐个比较，取得不同层同一位置中最大的概率所对应的值
            init_fn=slim.assign_from_checkpoint_fn(
                os.path.join(configure.Vgg16ModelPath,configure.Vgg16ModelName),
                slim.get_model_variables("vgg_16")
            )
            with tf.Session() as sess:
                init_fn(sess)
                fcn8s,fcn16s,fcn32s=sess.run([endpoints["vgg_16/pool3"],endpoints["vgg_16/pool4"],endpoints["vgg_16/pool5"]])
                upsampled_logits=upsample_tf(factor=16,input_img=fcn8s.squeeze())
                upsampled_predictions32=upsampled_logits.squeeze().argmax(2)
                """
                对于一维数组或者列表，unique函数去除其中重复的元素，并按元素由大到小返回一个新的无元素重复的元组或者列表
                a, s= np.unique(A, return_index=True)
                return_index=True表示返回新列表元素在旧列表中的位置，并以列表形式储存在s中。
                """
                unique_classes,relabeled_image=np.unique(upsampled_predictions32,return_inverse=True)
                relabeled_image=relabeled_image.reshape(upsampled_predictions32.shape)

                labels_names=[]
                names=imagenet_classes.class_names
                for index,current_class_number in enumerate(unique_classes):
                    labels_names.append(str(index)+" "+names[current_class_number+1])
                discrete_matshow(data=relabeled_image,labels_names=labels_names,title="Segmentation")

imageBreak()