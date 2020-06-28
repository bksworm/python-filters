#
# based on https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
#cython: language_level=3
import numpy as np
from scipy.ndimage import gaussian_filter
from .conv2d import naive_convolve_int

cimport numpy as np

cdef class CannyEdgeDetector:
    cdef int weak_pixel
    cdef int strong_pixel
    cdef float sigma
    cdef int kernel_size
    cdef float lowThreshold
    cdef float highThreshold
    cdef list imgs 
    cdef list imgs_final
    cdef object img_smoothed
    cdef object gradientMat
    cdef object thetaMat
    cdef object nonMaxImg 
    cdef object thresholdImg
    cdef object gauss_kernel
    cdef object Kx, Ky

    def __init__(self, imgs:list, sigma:float=1.0, kernel_size:int =5, 
            weak_pixel:int =75, strong_pixel:int=255, 
            lowthreshold:float=0.05, highthreshold:float=0.15):
        self.imgs = imgs
        self.img_smoothed = None
        self.gradientMat = None
        self.thetaMat = None
        self.nonMaxImg = None
        self.thresholdImg = None
        self.weak_pixel  = weak_pixel
        self.strong_pixel = strong_pixel
        self.sigma = sigma
        self.kernel_size  = kernel_size
        self.lowThreshold  = lowthreshold
        self.highThreshold = highthreshold

        self.gauss_kernel = self.gaussian_kernel(self.kernel_size, self.sigma)
        #sobel kernels
        self.Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        self.Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        return

    cdef object gaussian_kernel(self, size: int , sigma: float =1.0) :
        size = size // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        return g

    def sobel_filters(self, object img:np.array ) -> (np.array, np.array):
        Ix = naive_convolve_int(img, self.Kx)
        Iy = naive_convolve_int(img, self.Ky)
        G = np.hypot(Ix, Iy)
        G = G / G.max() * 255
        theta = np.arctan2(Iy, Ix)
        return (G, theta)


    cdef  object non_max_suppression(self, img:np.array, D:np.array):
        cdef int M, N, i, j  
        cdef np.ndarray[np.int_t, ndim=2] cimg = img
        M, N = img.shape
        cdef np.ndarray[np.int_t, ndim=2] Z = np.zeros((M,N), dtype=np.int)
        angle:np.array = D * (180.0 / np.pi)
        angle[angle < 0.0] += 180.0
 
        cdef int q, r, p   
        cdef float a 

        for i  in range(1,M-1):
            for j in range(1,N-1):
                try:
                    q  = 255
                    r  = 255
                    cdef float a = angle[i,j]
                   #angle 0
                    if (0 <= a < 22.5) or (157.5 <= a <= 180.0):
                        q = cimg[i, j+1]
                        r = cimg[i, j-1]
                    #angle 45
                    elif (22.5 <= a < 67.5):
                        q = cimg[i+1, j-1]
                        r = cimg[i-1, j+1]
                    #angle 90
                    elif (67.5 <= a < 112.5):
                        q = cimg[i+1, j]
                        r = cimg[i-1, j]
                    #angle 135
                    elif (112.5 <= a < 157.5):
                        q = cimg[i-1, j-1]
                        r = cimg[i+1, j+1]

                    p = cimg[i,j]
                    if ( p >= q) and (p >= r):
                        Z[i,j] = p
                    else:
                        pass #it's zeros filled already
                        #Z[i,j] = 0


                except IndexError as e:
                    pass

        return Z

    def threshold(self, img:np.array) ->np.array:

        highThreshold:float = img.max() * self.highThreshold;
        lowThreshold:float = highThreshold * self.lowThreshold;

        M, N = img.shape
        res:np.array = np.zeros((M,N), dtype=np.int32)

        weak  = np.int32(self.weak_pixel)
        strong = np.int32(self.strong_pixel)

        strong_i, strong_j = np.where(img >= highThreshold)
        zeros_i, zeros_j = np.where(img < lowThreshold)

        weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        return res

    def hysteresis(self, img:np.array) ->np.array:

        M, N = img.shape
        weak:int  = self.weak_pixel
        strong:int  = self.strong_pixel

        for i in range(1, M-1):
            for j in range(1, N-1):
                if (img[i,j] == weak):
                    try:
                        if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                            or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                            or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                            img[i, j] = strong
                        else:
                            img[i, j] = 0
                    except IndexError as e:
                        pass

        return img

    def detect(self) ->list:
        cdef list imgs_final
        for i, img in enumerate(self.imgs):
            img_smoothed = gaussian_filter(img, sigma=1, output=np.int) # naive_convolve(img, self.gauss_kernel)
            self.gradientMat, self.thetaMat = self.sobel_filters(img_smoothed)
            self.nonMaxImg = self.non_max_suppression(self.gradientMat, self.thetaMat)
            self.thresholdImg = self.threshold(self.nonMaxImg)
            img_final = self.hysteresis(self.thresholdImg)
            self.imgs_final.append(img_final)

        return self.imgs_final
