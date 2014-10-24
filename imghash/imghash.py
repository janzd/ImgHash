import scipy as sp
from scipy.misc import imresize
import scipy.fftpack
import scipy.ndimage.filters
import numpy as np
from . import utils

def dhash(self, img, img_size=16):
    img_res = imresize(img, [img_size, img_size+1], interp='bilinear')
    differences = img_res[:, :-1] > img_res[:, 1:]
    differences = np.hstack(differences)
    hash_string = utils.binary_to_hex(differences)
    return hash_string

def dhash_xydiff(self, img, img_size=16):
    img_res1 = imresize(img, [img_size, img_size+1], interp='bilinear')
    img_res2 = imresize(img, [img_size+1, img_size], interp='bilinear')
    differences1 = img_res1[:, :-1] > img_res1[:, 1:]
    differences2 = img_res2[:-1, :] > img_res2[1:, :]
    differences1 = np.hstack(differences1)
    differences2 = np.hstack(differences2)
    differences = np.hstack([differences1, differences2])
    hash_string = utils.binary_to_hex(differences)
    return hash_string

def dhash_binary(self, img, img_size=16):
    img_res = imresize(img, [img_size, img_size+1], interp='bilinear')
    differences = img_res[:, :-1] > img_res[:, 1:]
    differences = np.hstack(differences)
    hash_string = "".join(differences.astype(int).astype(str))
    return hash_string
   
def ahash(self, img, img_size=16):
    img_res = imresize(img, [img_size, img_size], interp='bilinear')
    mean = np.mean(img_res)
    differences = img_res > mean
    differences = np.hstack(differences)
    hash_string = utils.binary_to_hex(differences)
    return hash_string

def phash_dct(self, img, img_size=32):
    img = sp.ndimage.filters.gaussian_filter(img, sigma=1)
    img_res = imresize(img, [img_size, img_size], interp='bilinear')
    dct = sp.fftpack.dct(img_res.astype(float))
    dct = dct[:12, :12]
    dct = np.hstack(dct)
    dct_median = np.median(dct[1:])
    differences = dct > dct_median
    hash_string = utils.binary_to_hex(np.hstack(differences))
    return hash_string

def phash_blockmean(self, img, img_size=64):
    img_res = imresize(img, [img_size, img_size], interp='bilinear')
    blocks_in_line = 16
    block_size = img_size / blocks_in_line
    block_sums = np.zeros(blocks_in_line ** 2, dtype=np.int)
    total_sum = 0
    for i in range(img_res.shape[0]):
        for j in range(img_res.shape[1]):
            block_sums[(np.floor(i / block_size) * blocks_in_line) + np.floor(j / block_size)] += img_res[i, j]
    block_means = block_sums / (block_size ** 2)
    median_of_block_means = np.median(block_means)
    differences = block_means > median_of_block_means
    hash_string = utils.binary_to_hex(np.hstack(differences))
    return hash_string

def hog_hash(self, img, img_size=64):
    img = sp.ndimage.filters.gaussian_filter(img, sigma=1)
    img_res = imresize(img, [img_size, img_size], interp='bicubic').astype(np.float64)
    sobel_x = sp.ndimage.sobel(img_res, axis=0, mode='constant')
    sobel_y = sp.ndimage.sobel(img_res, axis=1, mode='constant')
    sobel_mag = np.hypot(sobel_x, sobel_y)
    sobel_dir = np.arctan2(sobel_y, sobel_x)
    blocks_in_line = 4
    block_size = img_size / blocks_in_line
    grad_dir_number = 16
    hogs = np.zeros([grad_dir_number, blocks_in_line ** 2], dtype=np.float64)
    for i in range(img_res.shape[0]):
        for j in range(img_res.shape[1]):
            hogs[np.floor((sobel_dir[i, j] + np.pi) / (2*np.pi / grad_dir_number)-0.000001), (np.floor(i / block_size) * blocks_in_line) + np.floor(j / block_size)] += sobel_mag[i, j]
    hog_medians = np.zeros(blocks_in_line ** 2, dtype=np.float64)
    for i in range(len(hog_medians)):
        hog_medians[i] = np.median(hogs[:, i])
    differences = np.zeros([grad_dir_number, blocks_in_line ** 2], dtype=np.int)
    for i in range(len(hog_medians)):
        differences[:, i] = hogs[:, i] > hog_medians[i]
    hash_string = utils.binary_to_hex(np.hstack(differences))
    return hash_string




        
