import numpy as np
import cv2

class Warp():
    def __init__(self):
        self.upper = 0
        self.lower = 0

    def warp(self, imgs, shift_list, blending=True):

        stitched_img, img2 = imgs
        shift = shift_list[-1]
        sum_x, sum_y = np.sum(np.asarray(shift_list), axis=0)
        
        print("shift: ", shift)
        '''
        _, thresh = cv2.threshold(cv2.cvtColor(img2.astype('uint8'), cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
        mask = np.where(thresh<1)
        print(mask)
        '''
        #cv2.imwrite('Mask.png', thresh)
        #input("mask")
        if sum_y < 0 and sum_y<self.upper:
            new_pad = int(self.upper - sum_y)
            padding = [
                (new_pad, 0),
                (-shift[0], 0) if shift[0] < 0 else (0, shift[0]),
                (0, 0)]
            stitched_img = np.lib.pad(stitched_img, padding, 'constant', constant_values=0)
            self.upper = sum_y
        elif sum_y>0 and sum_y > self.lower:
            new_pad = sum_y - self.lower
            padding = [
                (0, new_pad),
                (-shift[0], 0) if shift[0] < 0 else (0, shift[0]),
                (0, 0)]
            stitched_img = np.lib.pad(stitched_img, padding, 'constant', constant_values=0)
            self.lower = sum_y
        else:
            padding = [
                (0, 0),
                (-shift[0], 0) if shift[0] < 0 else (0, shift[0]),
                (0, 0)]
            stitched_img = np.lib.pad(stitched_img, padding, 'constant', constant_values=0)


        mask_ind = (img2>0)
        
        
        if not blending:
            h1, w1, _ = stitched_img.shape
            h2, w2, _ = img2.shape
            if shift[0] < 0:
                direction = 'left'
                if shift[1] < 0:
                    stitched_img[:h2, :w2][mask_ind] = img2[mask_ind]
                else:
                    stitched_img[shift[1]:shift[1]+h2, :w2][mask_ind] = img2[mask_ind]
            else:
                direction = 'right'
                if shift[1] < 0:
                    stitched_img[:h2, shift[0]:shift[0]+w2][mask_ind] = img2[mask_ind]
                else:
                    stitched_img[shift[1]:shift[1]+h2, shift[0]:shift[0]+w2][mask_ind] = img2[mask_ind]
        else:
            #Do alpha blending

            #cut out previous region
            region = img2.shape[1]+abs(shift[0])
            previous = stitched_img[:, region: ] if shift[0]< 0 else stitched_img[:, :-region]
            stitched_img = stitched_img[:, : region] if shift[0]< 0 else stitched_img[:, -region:]
            window_size = (img2.shape[1] - abs(shift[0]))//2

            h1, w1, _ = stitched_img.shape
            h2, w2, _ = img2.shape
            if sum_y < 0:
                upper_pad = int(sum_y - self.upper)
                lower_pad = int(h1-h2-upper_pad)
            else:
                lower_pad = int(self.lower - sum_y)
                upper_pad = int(h1-h2-lower_pad)

            inv_padding = [
            (upper_pad, lower_pad),
            (0, w1-w2) if shift[0] < 0 else (w1-w2, 0),
            (0, 0)]
            stitched_img2 = np.lib.pad(img2, inv_padding, 'constant', constant_values=0)

            stitched_img = self.alpha_blending(stitched_img, stitched_img2, shift, window_size)
            stitched_img = np.concatenate((stitched_img, previous) if shift[0] < 0 else (previous, stitched_img), axis=1)        

        return stitched_img

    def alpha_blending(self, stitched_img, img2, shift, window_size):
        border = stitched_img.shape[1]//2
        
        result = stitched_img.copy()
        if shift[0]<0:
            left_img = img2
            right_img = stitched_img
        else:
            left_img = stitched_img
            right_img = img2
        

        result[:, : border-window_size] = left_img[:, : border-window_size]
        result[:, border+window_size: ] = right_img[:, border+window_size: ]
        weight_right = np.arange(2*window_size)
        weight_right = (weight_right/(2*window_size)).reshape((1, 2*window_size, 1))
        weight_left = 1.0-weight_right
        result[:, border-window_size: border+window_size] = weight_left*left_img[:, border-window_size: border+window_size] \
                                                        + weight_right*right_img[:, border-window_size: border+window_size]
                    
        return result

       

