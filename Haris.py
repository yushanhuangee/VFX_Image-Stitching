import cv2
import numpy as np

class HarisDetector():
    def __init__(self) -> None:
        pass
    def DetectandDescribe(self, img):
        '''
        return kps, features
        '''
        Response, first_Grad = self.computeResponseandGrad(img)
        corners = self.findLocalMax(Response)
        #Descriptor
        kps, features = self.Descriptor(corners, first_Grad)

        return kps, features


    def computeResponseandGrad(self, img, ksize=9, sigma=3, k=0.04):
        im_gray = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY)
        gray_blur = cv2.GaussianBlur(im_gray, (ksize, ksize), sigma)
        Iy, Ix = np.gradient(gray_blur)
        Ixx = Ix ** 2
        Ixy = Iy * Ix
        Iyy = Iy ** 2

        Mxx = cv2.GaussianBlur(Ixx, (ksize, ksize), sigma)
        Mxy = cv2.GaussianBlur(Ixy, (ksize, ksize), sigma)
        Myy = cv2.GaussianBlur(Iyy, (ksize, ksize), sigma)

        detM = (Mxx * Myy) - (Mxy ** 2)
        traceM = Mxx + Myy

        R = detM - k * (traceM ** 2)

        return R, (Ix, Iy)

    def findLocalMax(self, R, window_size=3, threshold=100):
        corners = []

        H, W = R.shape

        for i in range(H):
            for j in range(W):
                if R[i][j]> threshold:
                    max_value = R[i][j]
                    #check if all its neighbors are smaller
                    check_max = True
                    for w_i in range(-(window_size//2), (window_size//2)):
                        for w_j in range(-(window_size//2), (window_size//2)):
                            if (i+w_i) >= 0 and (i+w_i) < H and (j+w_j) >= 0 and (j+w_j) < W:
                                if R[i+w_i][j+w_j] > max_value:
                                    check_max=False
                                    break
                    if check_max:
                        corners.append([i, j])
        return corners
    
    def Descriptor(self, corners, first_grad, num_bins=8, window_size=3):
        Ix, Iy = first_grad

        #=================Computing Orientation===================
        kps = []
        orientations = []

        H, W = Ix.shape

        Ixx = Ix ** 2
        Iyy = Iy ** 2
        m = (Ixx + Iyy) ** 0.5
        weighted_Grad_mag = cv2.GaussianBlur(m, (3, 3), 1.5)

        theta = np.arctan(Iy / (Ix + 1e-8)) * (180 / np.pi)
        theta[Ix < 0] += 180
        theta = (theta + 360) % 360

        #only dealing with 45 degree(other small shifts are considered by Taylor's expansion)
        bin_size = 360 // num_bins
        theta_bins = theta // bin_size

        for kps_i, kps_j in corners:
            ori = np.zeros(num_bins)
            for w_i in range(-(window_size//2), (window_size//2)):
                for w_j in range(-(window_size//2), (window_size//2)):
                    if (kps_i+w_i) >= 0 and (kps_i+w_i) < H and (kps_j+w_j) >= 0 and (kps_j+w_j) < W:
                        ori[int(theta_bins[kps_i+w_i][kps_j+w_j])] += weighted_Grad_mag[kps_i+w_i][kps_j+w_j]
            peak = np.max(ori)
            main_peaks = np.where(ori/peak > 0.8)[0]

            #choose top 2
            for i, peak_idx in enumerate(main_peaks):
                if i<2:
                    kps.append([kps_i, kps_j])
                    orientations.append(peak_idx)

        #====================Computing features=====================
        features = np.zeros((len(kps), 4, 4, 8))

        for i, (kps_i, kps_j) in enumerate(kps):
            main_direction = np.zeros(num_bins)
            main_direction[orientations[i]] = 1
            feature = np.zeros((4, 4, num_bins))

            ## 16x16 window around the keypoint, which is divided into 16 sub-blocks of 4x4 size.
            subi = [kps_i - 8, kps_i - 4, kps_i, kps_i + 4]
            subj = [kps_j - 8, kps_j - 4, kps_j, kps_j + 4]
            for idx_i, b_i in enumerate(subi):
                for idx_j, b_j in enumerate(subj):
                    ori = np.zeros(num_bins)
                    for win_i in range(b_i, b_i+5):
                        for win_j in range(b_j, b_j+5):
                            if win_i>=0 and win_i<H and win_j>=0 and win_j<W:
                                ori[int(theta_bins[win_i][win_j])] += weighted_Grad_mag[win_i][win_j]
                    ## for rotation dependence
                    ori = ori - main_direction
                    ## for illumination dependence
                    ori_norm = ori/(np.sum(ori) + 1e-8)
                    ori_clip = [o if o < 0.2 else 0.2 for o in ori_norm]
                    orient_norm2 = ori_clip / (np.sum(ori_clip) + 1e-8)
                    feature[idx_i, idx_j, :] = orient_norm2
            features[i] = feature
        features = features.reshape((len(kps),-1))
        return kps, features
            




        
