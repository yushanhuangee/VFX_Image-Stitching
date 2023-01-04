import numpy as np
import random


def RANSAC(matched_points, threshold=10):
    #random sample points
    Num_subsample = 1
    Num_iter = 8000
    Best_H = None
    N_pair = len(matched_points)
    matched_points = np.asarray(matched_points)
    max_Num_inlier = 0

    for it in range(Num_iter):

        sample_idx = np.random.randint(0, N_pair, size=Num_subsample)
        
        sample_pairs = np.asarray(matched_points[sample_idx])
        tar_points = sample_pairs[:, 0, :]
        src_points = sample_pairs[:, 1, :]

        #H = solve_homography(src_points, tar_points)
        shift = tar_points - src_points
        #Calculating number of inlier

        
        cand_src_pt = []
        cand_tar_pt = []
        for i, (src_pt, tar_pt) in enumerate(zip(matched_points[:, 1, :], matched_points[:, 0, :])):
            if i not in sample_idx:
                cand_src_pt.append(src_pt)
                cand_tar_pt.append(tar_pt)

        cand_src_pt = np.asarray(cand_src_pt)
        cand_tar_pt = np.asarray(cand_tar_pt)

        #homo_src_pt = np.concatenate((cand_src_pt, np.ones((cand_src_pt.shape[0], 1))), axis=1)
        #warped_src_pt = np.matmul(H, homo_src_pt.T)
        #warped_src_pt = warped_src_pt[:2, :]/np.expand_dims((warped_src_pt[2, :]+1e-8), axis=0)
        #warped_src_pt = warped_src_pt.T
        warped_src_pt = cand_src_pt + shift
    
        diff = np.linalg.norm(warped_src_pt-cand_tar_pt, axis=-1, keepdims=True)
        inlier_idx = np.where(diff<threshold)
        num_inlier = len(inlier_idx[0])
        if num_inlier>0:
            avg_shift = np.mean((cand_tar_pt-cand_src_pt)[inlier_idx[0]], axis=0).astype('int')
    
        
        if num_inlier > max_Num_inlier:
            Best_H = [avg_shift]
            max_Num_inlier = num_inlier

    #print("The Number of Maximum Inlier:", max_Num_inlier)

    return Best_H
        

        
        



'''
def solve_homography(src_points, tar_points):
    A = []
    for src_pt, tar_pt in zip(src_points, tar_points):
        A.append([-src_pt[0], -src_pt[1], -1, 0, 0, 0, src_pt[0]*tar_pt[0], src_pt[1]*tar_pt[0], tar_pt[0]])
        A.append([0, 0, 0, -src_pt[0], -src_pt[1], -1, src_pt[0]*tar_pt[1], src_pt[1]*tar_pt[1], tar_pt[1]])

    u, s, vt = np.linalg.svd(A) # Solve s ystem of linear equations Ah = 0 using SVD
    # pick H from last line of vt
    H = np.reshape(vt[-1], (3,3))
    # normalization, let H[2,2] equals to 1
    H = H/H[2,2]

    return H



class Homography:
    def solve_homography(self, P, m):
        """
        Solve homography matrix 
        Args:
            P:  Coordinates of the points in the original plane,
            m:  Coordinates of the points in the target plane
        Returns:
            H: Homography matrix 
        """
        A = []  
        for r in range(len(P)): 
            #print(m[r, 0])
            A.append([-P[r,0], -P[r,1], -1, 0, 0, 0, P[r,0]*m[r,0], P[r,1]*m[r,0], m[r,0]])
            A.append([0, 0, 0, -P[r,0], -P[r,1], -1, P[r,0]*m[r,1], P[r,1]*m[r,1], m[r,1]])

        u, s, vt = np.linalg.svd(A) # Solve s ystem of linear equations Ah = 0 using SVD
        # pick H from last line of vt  
        H = np.reshape(vt[8], (3,3))
        # normalization, let H[2,2] equals to 1
        H = (1/H.item(8)) * H

        return H

def fitHomoMat(matches_pos):
        
            #Fit the best homography model with RANSAC algorithm - noBlending、linearBlending、linearBlendingWithConstant
        
        dstPoints = [] # i.e. left image(destination image)
        srcPoints = [] # i.e. right image(source image) 
        for dstPoint, srcPoint in matches_pos:
            dstPoints.append(list(dstPoint)) 
            srcPoints.append(list(srcPoint))
        dstPoints = np.array(dstPoints)
        srcPoints = np.array(srcPoints)
        
        homography = Homography()
        
        # RANSAC algorithm, selecting the best fit homography
        NumSample = len(matches_pos)
        threshold = 5.0  
        NumIter = 8000
        NumRamdomSubSample = 4
        MaxInlier = 0
        Best_H = None
        
        for run in range(NumIter):
            SubSampleIdx = random.sample(range(NumSample), NumRamdomSubSample) # get the Index of ramdom sampling
            H = homography.solve_homography(srcPoints[SubSampleIdx], dstPoints[SubSampleIdx])
            
            # find the best Homography have the the maximum number of inlier
            NumInlier = 0 
            for i in range(NumSample):
                if i not in SubSampleIdx:
                    concateCoor = np.hstack((srcPoints[i], [1])) # add z-axis as 1
                    dstCoor = H @ concateCoor.T # calculate the coordination after transform to destination img 
                    if dstCoor[2] <= 1e-8: # avoid divide zero number, or too small number cause overflow
                        continue
                    dstCoor = dstCoor / dstCoor[2]
                    if (np.linalg.norm(dstCoor[:2] - dstPoints[i]) < threshold):
                        NumInlier = NumInlier + 1
            if (MaxInlier < NumInlier):
                MaxInlier = NumInlier
                Best_H = H
                
        print("The Number of Maximum Inlier:", MaxInlier)
        
        return Best_H

'''