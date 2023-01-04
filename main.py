import glob
import argparse
import os
import cv2
import numpy as np
import random
from fitHomo import *
from warping import *
from cylindrical_projection import *
from matching import *
from Haris import *

class Stitcher:
    def __init__(self):
        self.cache_kps = []
        self.cache_feature = []
        self.curr_shift = [[0,0]]
        self.warper = Warp()

    def stitch(self, stitched_img, img_l, img_r):
        print("feature detection")
        if len(self.cache_feature)==0 or len(self.cache_kps)==0:
            img_l = img_l.astype('uint8')
            img_l_kps, img_l_features = self.featureDetection(img_l)
        else:
            img_l_kps, img_l_features = self.cache_kps, self.cache_feature

        img_r = img_r.astype('uint8')
        img_r_kps, img_r_features = self.featureDetection(img_r)

        print("feature matching")
        '''
        =================================Here!!!!!!!!!!!!!!!!!!!!!=========================
        '''
        matches_pos = customizeMatch(img_l_kps, img_r_kps, img_l_features, img_r_features, ratio = 0.75)
        #drawMatches([img_l, img_r], matches_pos)

        #fit the homography model with RANSAC algorithm
        print("RANSAC")
        shift = RANSAC(matches_pos)
        self.curr_shift.append(shift[0])
        #HomoMat = fitHomoMat(matches_pos)
        #print(HomoMat)
        warp_img = self.warper.warp([stitched_img, img_r], self.curr_shift)
        #warp_img = warp([stitched_img, img_r], HomoMat, blending_mode)
        

        self.cache_kps = img_r_kps
        self.cache_feature = img_r_features
        
        return warp_img

    def featureDetection(self, img):
        #sift = cv2.SIFT_create()
        #kps, features = sift.detectAndCompute(img,None)
        Haris = HarisDetector()
        kps, features = Haris.DetectandDescribe(img)

        return kps, features

    def align(self, img):
        sum_x, sum_y = np.sum(self.curr_shift, axis=0)
        

        y_shift = np.abs(sum_y)

        # same sign
        if sum_x*sum_y > 0:
            col_shift = np.linspace(y_shift, 0, num=img.shape[1], dtype=np.uint16)
        else:
            col_shift = np.linspace(0, y_shift, num=img.shape[1], dtype=np.uint16)

        aligned = img.copy()
        for x in range(img.shape[1]):
            aligned[:,x] = np.roll(img[:,x], col_shift[x], axis=0)

        return aligned

    def crop(self, img, focal, input_H, input_W):
        max_focal = max(focal)
        warp_distort = int(abs(input_H//2 - max_focal*((input_H//2)/math.sqrt((input_W//2)**2+max_focal**2))))
        _, upper_shift_y = np.min(np.asarray(self.curr_shift),axis=0)
        _, lower_shift_y = np.max(np.asarray(self.curr_shift),axis=0)
        upper = self.warper.upper-warp_distort+upper_shift_y
        lower = self.warper.lower+warp_distort+lower_shift_y
        return img[-upper:img.shape[0]-lower,:]
        


    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_dir', type=str, default='../data/scene_16_new')

    args, unknown = parser.parse_known_args()

    stitcher = Stitcher()

    #Read images
    if 'parrington' in args.img_dir:
        images = sorted(glob.glob(os.path.join(args.img_dir, 'prtn*.jpg')))
        #Read focal length
        focal = []
        f = open(os.path.join(args.img_dir, 'pano.txt'))
        for line in f.readlines():
            line = line.split(' ')
            if len(line) == 1 and line[0]!='\n'and 'jpg' not in line[0]:
                focal.append(float(line[0].split('\n')[0]))
    '''
    if 'scene_1' in args.img_dir:
        images = sorted(glob.glob(os.path.join(args.img_dir, 'DSC*.jpg')))
        #Read focal length
        focal = []
        f = open(os.path.join(args.img_dir, 'pano.txt'))
        for line in f.readlines():
            line = line.split(' ')
            if len(line) == 1 and line[0]!='\n'and 'jpg' not in line[0]:
                focal.append(float(line[0].split('\n')[0]))
    '''
    if 'scene_16' in args.img_dir:
        images = sorted(glob.glob(os.path.join(args.img_dir, 'DSC*.jpg')))
        #Read focal length
        focal = []
        f = open(os.path.join(args.img_dir, 'pano.txt'))
        for line in f.readlines():
            line = line.split(' ')
            if len(line) == 1 and line[0]!='\n'and 'jpg' not in line[0]:
                focal.append(float(line[0].split('\n')[0]))

    print(focal)


    input_H, input_W = 0, 0
    cylindrical_images = []
    for i , filename in enumerate(images):
        img = cv2.imread(filename)
        H, W, _ = img.shape
        img = cv2.resize(img, (W//8, H//8), interpolation=cv2.INTER_AREA)
        cylindrical_images.append(cylindrical_projection(img, focal[i]))
        input_H, input_W, _ = cylindrical_images[-1].shape
        #cv2.imwrite('warp_{}.png'.format(str(i)), cylindrical_images[-1])

    stitched_img = cylindrical_images[0].copy()

    for i ,(img1, img2) in enumerate(zip(cylindrical_images[:-1], cylindrical_images[1:])):

        #cv2.imwrite('stitch_input.png', img2)
        stitched_img = stitcher.stitch(stitched_img, img1, img2)
        
        #cv2.imwrite('Stich_result.png', stitched_img.astype('uint8'))
        
    result_img = stitcher.align(stitched_img)
    #cv2.imwrite('Final_result.png', result_img.astype('uint8'))
    cropped_img = stitcher.crop(result_img, focal, input_H, input_W)
    cv2.imwrite('result.png', cropped_img.astype('uint8'))
    
           
