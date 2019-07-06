
# 先读一下图
import cv2
import random
import numpy as np                         # 这个 as 操作看上去有点像重命名
from matplotlib import pyplot as plt


##  截取操作
def crop(im, l, w ):
    height, width, channels = im.shape
    img_crop = im[1:l, 1:w]
    return img_crop

##  颜色变换
def colorshift(im, rc, bc, gc):
    (G, R, B) = cv2.split(im)
    if bc == 0:
        pass
    elif bc > 0:
        lim = 255 - bc
        B[B > lim] = 255
        B[B <= lim] = (bc + B[B <= lim]).astype(img.dtype)
    elif bc < 0:
        lim = 0 - bc
        B[B < lim] = 0
        B[B >= lim] = (bc + B[B >= lim]).astype(img.dtype)
    # 然后是绿色
    if gc== 0:
        pass
    elif gc > 0:
        lim = 255 - gc
        G[G > lim] = 255
        G[G <= lim] = (gc + G[G <= lim]).astype(img.dtype)
    elif gc < 0:
        lim = 0 - gc
        G[G < lim] = 0
        G[G >= lim] = (gc + G[G >= lim]).astype(img.dtype)
    # 最后是红色
    if rc == 0:
        pass
    elif rc > 0:
        lim = 255 - rc
        R[R > lim] = 255
        R[R <= lim] = (rc + R[R <= lim]).astype(img.dtype)
    elif rc < 0:
        lim = 0 - rc
        R[R < lim] = 0
        R[R >= lim] = (rc + R[R >= lim]).astype(img.dtype)
    img_merge = cv2.merge((B, G, R))
    return img_merge

## 旋转
def rotate(im, angle):
    M = cv2.getRotationMatrix2D((im.shape[1] / 2, im.shape[0] / 2), angle, 1)
    img_rotate = cv2.warpAffine(im, M, (im.shape[1], im.shape[0]))
    return img_rotate

## 投射perspective
#  只变换图像的一组对角，向内压缩。
def perspective(im, hc, wc):
    height, width, channels = im.shape
    [x1, y1] = [0, 0]
    [x2, y2] = [0, height]
    [x3, y3] = [width, 0]
    [x4, y4] = [width, height]
    [dx1, dy1] = [0, 0]
    [dx2, dy2] = [wc, height - hc]
    [dx3, dy3] = [width - wc , hc]
    [dx4, dy4] = [width, height]
    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    M_warp = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(im, M_warp, (width, height))
    return img_warp


#  开始调用
img = cv2.imread('D:/Python/pictures/vango1.jpg')     # 这里改图片路径！
print (img.shape)
img_crop = crop(img, 400, 400 )
img_colorshift = colorshift(img_crop, -50, -75, 100)
img_rotate = rotate(img_colorshift, 45)
img_pers = perspective(img_rotate, 100, 100)

cv2.imshow('img',img)
cv2.imshow('img_crop',img_crop)
cv2.imshow("img_colorshift", img_colorshift)
cv2.imshow('img_rotate',img_rotate)
cv2.imshow('img_pers',img_pers)

# 这三行不写好像就没有运行结果
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()

