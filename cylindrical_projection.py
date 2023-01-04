import numpy as np
import math
import cv2

def cylindrical_projection(img, focal_length):
    H, W, C = img.shape
    cyl_proj = np.zeros(img.shape)

    s = focal_length
    map_x = np.zeros((H, W))
    map_y = np.zeros((H, W))

    for y in range(H):
        for x in range(W):
            cyl_x = W//2 + s*math.atan((x-W//2)/focal_length)
            cyl_y = H//2 + s*((y-H//2)/math.sqrt((x-W//2)**2+focal_length**2))
            #print(x,y)
            #print(cyl_x,)
            if cyl_x >=0 and cyl_x < W and cyl_y >=0 and cyl_y < H:
                #map_x[y][x] = cyl_x
                #map_y[y][x] = cyl_y
                cyl_proj[int(cyl_y)][int(cyl_x)] = img [y][x]

    #map_x = map_x.reshape(H, W).astype(np.float32)
    #map_y = map_y.reshape(H, W).astype(np.float32)

    #Remove black border
    _, thresh = cv2.threshold(cv2.cvtColor(cyl_proj.astype('uint8'), cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(thresh)

    #cyl_proj = cv2.remap(img, map_x, map_y, interpolation = cv2.INTER_LINEAR)
    #cv2.imwrite('Cylindrical.png', cyl_proj[y:y+h, x:x+w])
    return cyl_proj[y:y+h, x:x+w]

