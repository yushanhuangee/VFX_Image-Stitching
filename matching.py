import numpy as np
import cv2

import scipy.spatial

def match(des1, des2):
    matches = []
    # use KD-tree to find neighbor and count Euclidean distance
    des2_kdtree = scipy.spatial.KDTree(des2)
    distances, trainIdx = des2_kdtree.query(des1, k=2)
    for queryIdx in range(des1.shape[0]):
        matches.append([cv2.DMatch(queryIdx, trainIdx[queryIdx][0], distances[queryIdx][0]), \
                            cv2.DMatch(queryIdx, trainIdx[queryIdx][1], distances[queryIdx][1])])
    return matches

def customizeMatch(kp1, kp2, des1, des2, ratio=0.75):
    matches = match(des1, des2)
    good = []
    for m, n in matches:
        if m.distance < ratio*n.distance:
            good.append(m)
    src_pts = np.float32([ [kp1[m.queryIdx][1],kp1[m.queryIdx][0]] for m in good ]).reshape(-1,2).astype('int')
    dst_pts = np.float32([ [kp2[m.trainIdx][1],kp2[m.trainIdx][0]] for m in good ]).reshape(-1,2).astype('int')

    combine_pts = []
    for i in range(len(src_pts)):
        combine_pts.append([src_pts[i], dst_pts[i]])
    return combine_pts
'''
def matchKeyPoint(kps_l, kps_r, features_l, features_r, ratio):
        
           # Match the Keypoints beteewn two image
        
        Match_idxAndDist = [] # min corresponding index, min distance, seccond min corresponding index, second min distance
        for i in range(len(features_l)):
            min_IdxDis = [-1, np.inf]  # record the min corresponding index, min distance
            secMin_IdxDis = [-1 ,np.inf]  # record the second corresponding min index, min distance
            for j in range(len(features_r)):
                dist = np.linalg.norm(features_l[i] - features_r[j])
                if (min_IdxDis[1] > dist):
                    secMin_IdxDis = np.copy(min_IdxDis)
                    min_IdxDis = [j , dist]
                elif (secMin_IdxDis[1] > dist and secMin_IdxDis[1] != min_IdxDis[1]):
                    secMin_IdxDis = [j, dist]
            
            Match_idxAndDist.append([min_IdxDis[0], min_IdxDis[1], secMin_IdxDis[0], secMin_IdxDis[1]])

        # ratio test as per Lowe's paper
        # reject the point if ||f1 - f2 || / || f1 - f2' || >= ratio, that represent it's ambiguous point
        goodMatches = []
        for i in range(len(Match_idxAndDist)):
            if (Match_idxAndDist[i][1] <= Match_idxAndDist[i][3] * ratio):
                goodMatches.append((i, Match_idxAndDist[i][0]))
            
        goodMatches_pos = []
        for (idx, correspondingIdx) in goodMatches:
            psA = (int(kps_l[idx][1]), int(kps_l[idx][0]))
            psB = (int(kps_r[correspondingIdx][1]), int(kps_r[correspondingIdx][0]))
            goodMatches_pos.append([psA, psB])
            
        return goodMatches_pos
'''
def drawMatches( imgs, matches_pos):
    '''
        Draw the match points img with keypoints and connection line
    '''
    
    # initialize the output visualization image
    img_left, img_right = imgs
    (hl, wl) = img_left.shape[:2]
    (hr, wr) = img_right.shape[:2]
    vis = np.zeros((max(hl, hr), wl + wr, 3), dtype="uint8")
    vis[0:hl, 0:wl] = img_left
    vis[0:hr, wl:] = img_right
    
    # Draw the match
    for (img_left_pos, img_right_pos) in matches_pos:
        
        pos_l = (img_left_pos[0],img_left_pos[1])
        pos_r = img_right_pos[0] + wl, img_right_pos[1]
        cv2.circle(vis, pos_l, 3, (0, 0, 255), 1)
        cv2.circle(vis, pos_r, 3, (0, 255, 0), 1)
        cv2.line(vis, pos_l, pos_r, (255, 0, 0), 1)
            
    # return the visualization
    #plt.figure(1)
    #plt.title("img with matching points")
    #plt.imshow(vis[:,:,::-1])
    cv2.imwrite("matching.jpg", vis)
    
    return vis