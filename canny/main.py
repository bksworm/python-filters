#cython: language_level=3
import numpy as np
import skimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import scipy.misc as sm
from canny.filter  import CannyEdgeDetector 
from skimage.feature import canny

def rgb2gray(rgb:np.array):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def load_data(dir_name:str = 'faces_imgs'):
    '''
    Load images from the "faces_imgs" directory
    Images are in JPG and we convert it to gray scale images
    '''
    imgs:list = []
    for filename in os.listdir(dir_name):
        if os.path.isfile(dir_name + '/' + filename):
            img = mpimg.imread(dir_name + '/' + filename)
            img = rgb2gray(img)
            imgs.append(img)
    return imgs


def visualize(imgs:list, format=None, gray:bool=False):
    plt.figure(figsize=(20, 40))
    for i, img in enumerate(imgs):
        if img.shape[0] == 3:
            img = img.transpose(1,2,0)
        plt_idx = i+1
        plt.subplot(2, 2, plt_idx)
        plt.imshow(img, format)
    plt.show()

#if __name__  == "__main__":
imgs = load_data("faces_imgs/archive")
res = []
ed = CannyEdgeDetector(imgs)
res = ed.detect()
#for im in imgs:
#    res.append(canny(im, sigma=5.0))

visualize(res)
