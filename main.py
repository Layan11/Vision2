import cv
import cv2
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    left = cv2.imread('location_1_frame_001.jpg')
    right = cv2.imread('location_1_frame_002.jpg')

    # plt.subplot(1, 2, 1), plt.imshow(left)
    # plt.title('left')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(1, 2, 2), plt.imshow(right)
    # plt.title('right')
    # plt.xticks([]), plt.yticks([])
    # plt.show()

    # setting the first set of matches (s1) manually
    left_points = [(55, 176), (90, 176), (90, 259), (55, 259), (428, 160), (446, 159), (444, 239), (425, 232), (245, 403), (241, 447)]
    right_points = [(65, 177), (99, 177), (100, 254), (67, 254), (414, 160), (429, 160), (427, 233), (412, 226), (245, 390), (242, 428)]
    S1 = []
    for i in range(10):
        S1.append([left_points[i], right_points[i]])
    # show the chosen matches on the images
    for i in range(10):
        newimage1 = cv2.circle(left, (S1[i][0][0], S1[i][0][1]), radius=3, color=(0, 0, 255), thickness=-1)
        newimage2 = cv2.circle(right, (S1[i][1][0], S1[i][1][1]), radius=3, color=(255, 0, 0), thickness=-1)

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
    print('F = ')
    print(F)
    x1 = np.array([S1[0][0][0], S1[0][0][1], 1])

    print('x1 = ')
    print(x1)
    l1 = F.dot(x1)
    print('l1 = ')
    print(l1)

    pt1 = [S1[0][0][0], S1[0][0][1]]
    r, c = left.shape[0], left.shape[1]
    # img1 = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    for r, pts1 in zip(l1.reshape(-1, 3), pt1):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(left, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pts1), 5, color, -1)



