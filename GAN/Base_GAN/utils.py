import scipy.misc
import numpy as np

# 保存图片函数
def save_images(images, size, path):
    """
    Save the samples images
    The best size number is
            int(max(sqrt(image.shape[0]),sqrt(image.shape[1]))) + 1
    example:
        The batch_size is 64, then the size is recommended [8, 8]
        The batch_size is 32, then the size is recommended [6, 6]
    """

    # 图片归一化，主要用于生成器输出是 tanh 形式的归一化
    img = (images + 1.0) / 2.0
    h, w = img.shape[1], img.shape[2]

    # 产生一个大画布，用来保存生成的 batch_size 个图像
    merge_img = np.zeros((h * size[0], w * size[1], 3))

    # 循环使得画布特定地方值为某一幅图像的值
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        merge_img[j * h:j * h + h, i * w:i * w + w, :] = image

    # 保存画布
    return scipy.misc.imsave(path, merge_img)