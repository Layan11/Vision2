# import cv
import math

import cv as cv
import cv2
import matplotlib.pyplot as plt
import numpy
import numpy as np

def drawlines(img1,img2,lines,pts1,pts2, secondPicColors = None):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape[:-1]
    Colors = []
    i = 0
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    # img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    # img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        if secondPicColors:
            color = secondPicColors[i]
            i+=1
        else:
            color = tuple(np.random.randint(0,255,3).tolist())
            Colors.append(color)
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),20,color,-1)

    return img1,img2, Colors

if __name__ == '__main__':
    left = cv2.imread('location_2_frame_001.jpg')
    right = cv2.imread('location_2_frame_002.jpg')
    PointsNum = 10
    plt.subplot(1, 2, 1), plt.imshow(left)
    plt.title('left')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(right)
    plt.title('right')
    plt.xticks([]), plt.yticks([])
    plt.show()

    # setting the first set of matches (s1) manually
    # left_points = [(55, 176), (90, 176), (90, 259), (55, 259), (428, 160), (446, 159), (444, 239), (425, 232), (245, 403), (241, 447)]
    # right_points = [(65, 177), (99, 177), (100, 254), (67, 254), (414, 160), (429, 160), (427, 233), (412, 226), (245, 390), (242, 428)]

    left_points = [(214, 150), (263, 154), (345, 170), (381, 164), (326, 344), (582, 392), (82, 259), (563, 248), (155, 45), (518, 13)]
    right_points = [(219, 128), (256, 133), (325, 149), (362, 143), (328, 296), (567, 312), (112, 226), (531, 207), (165, 31), (489, 18)]

    # left_points = [(101, 150), (333, 150), (333, 379), (101, 379), (100,149), (332, 149), (332, 378), (100, 378), (215, 238)]
    # right_points = [(898, 285), (1127, 285), (1127, 517), (898, 517), (897, 284), (1126, 284), (1126, 516), (897, 516), (1024, 403)]

    S1 = []
    for i in range(PointsNum):
        S1.append([left_points[i], right_points[i]])
    # show the chosen matches on the images
    for i in range(PointsNum):
        newimage1 = cv2.circle(left, (S1[i][0][0], S1[i][0][1]), radius=6, color=(0, 0, 255), thickness=-1)
        newimage2 = cv2.circle(right, (S1[i][1][0], S1[i][1][1]), radius=6, color=(255, 0, 0), thickness=-1)

    plt.subplot(1, 2, 1), plt.imshow(newimage1)
    plt.title('newimage1')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(newimage2)
    plt.title('newimage2')
    plt.xticks([]), plt.yticks([])
    plt.show()

    # find the fundamental matrix from the first set of matches (s1)
    F = cv2.findFundamentalMat(np. array(left_points), np. array(right_points), cv2.FM_8POINT)
    F = np.array(F[0])########### !

    # print('F = ')
    # print(F)
    xleft = np.array([S1[0][0][0], S1[0][0][1], 1])
    xright = np.array([S1[0][1][0], S1[0][1][1], 1])
    lines1 = F.dot(xleft)
    lines2 = F.dot(xright)
    SED2 = []
    Ri = numpy.transpose(xleft).dot(F).dot(xright)
    ans = ((1/(lines1[0]**2 + lines1[1]**2)) + (1 / (lines2[0]**2 + lines2[1]**2)))
    finalans = ans * (Ri**2)
    SED2.append(finalans)
    # SED2.append(math.dist(xleft, lines1)**2 + math.dist(xright, lines2)**2)
    # print('x1 = ')
    # print(xleft)
    # l1 = F.dot(xleft)
    # print('l1 = ')
    # print(l1)

    for i in range(PointsNum-1):
       xleft = np.array([S1[i+1][0][0], S1[i+1][0][1], 1])
       xright = np.array([S1[i+1][1][0], S1[i+1][1][1], 1])

       lines1 = np.append(lines1, F.dot(xleft))
       lines1 = lines1.reshape(-1, 3)
       lines2 = np.append(lines2, F.dot(xright))
       lines2 = lines2.reshape(-1, 3)
       # SED2.append(math.dist(xleft, lines1[i])**2 + math.dist(xright, lines2[i])**2)

       Ri = numpy.transpose(xleft).dot(F).dot(xright)
       ans = ((1 / (lines1[i][0] ** 2 + lines1[i][1] ** 2)) + (1 / (lines2[i][0] ** 2 + lines2[i][1] ** 2)))
       finalans = ans * (Ri ** 2)
       SED2.append(finalans)

       # Find epilines corresponding to points in left image (first image) and
       # drawing its lines on right image
       # lines2 = F.dot(xright)
       # lines2 = lines1.reshape(-1, 3)
       # cv2.computeCorrespondEpilines(left_points.reshape(-1, 1, 2), 1, F)

    img5, img6, colors = drawlines(left, right, lines1, left_points, right_points)
    img3, img4, k = drawlines(right, left, lines2, right_points, left_points, colors)
    plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img3)
    plt.show()
    error = sum(SED2)
    print("ERROR IS  = ")
    print(error)
    # for i in range(PointsNum):
    #    xleft = np.array([S1[i][0][0], S1[i][0][1], 1])
    #    xright = np.array([S1[i][1][0], S1[i][1][1], 1])
    #    lines1 = F.dot(xleft)
    #    lines1 = lines1.reshape(-1, 3)
    #    lines2 = F.dot(xright)
    #    lines2 = lines2.reshape(-1, 3)
    #
    #    # Find epilines corresponding to points in left image (first image) and
    #    # drawing its lines on right image
    #    # lines2 = F.dot(xright)
    #    # lines2 = lines1.reshape(-1, 3)
    #    # cv2.computeCorrespondEpilines(left_points.reshape(-1, 1, 2), 1, F)
    #
    #    img5, img6, colors = drawlines(left, right, lines1, left_points, right_points)
    #    img3, img4, k = drawlines(right, left, lines2, right_points, left_points, colors)
    #    plt.subplot(121), plt.imshow(img5)
    #    plt.subplot(122), plt.imshow(img3)
    #    plt.show()
