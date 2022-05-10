import math

import cv2
import numpy as np
from matplotlib import pyplot as plt


def DLT(P1, P2, point1, point2):
    A = [point1[1] * P1[2, :] - P1[1, :],
         P1[0, :] - point1[0] * P1[2, :],
         point2[1] * P2[2, :] - P2[1, :],
         P2[0, :] - point2[0] * P2[2, :]
         ]
    A = np.array(A).reshape((4, 4))
    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices=False)

    return Vh[3, 0:3] / Vh[3, 3]

if __name__ == '__main__':
    pic1 = cv2.imread('house_1.png')
    pic2 = cv2.imread('house_2.png')
    matchedPoints1 = []
    matchedPoints2 = []
    # reading the input and organizing the given points in variables accordingly
    a_file = open("matchedPoints1.txt")
    file_contents = a_file.read()
    contents_split = file_contents.splitlines()
    for i in range(len(contents_split)):
        matchedPoints1.append((np.round(float(contents_split[i].split(",")[0])).astype("int"),
                               np.round(float(contents_split[i].split(",")[1])).astype("int")))
    a_file = open("matchedPoints2.txt")
    file_contents = a_file.read()
    contents_split = file_contents.splitlines()
    for i in range(len(contents_split)):
        matchedPoints2.append((np.round(float(contents_split[i].split(",")[0])).astype("int"),
                               np.round(float(contents_split[i].split(",")[1])).astype("int")))

    # drawing the matched points and lines in both images accordingly and with corresponding colors
    colors = []
    for i in range(len(contents_split)):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        colors.append(color)

        newimage1 = cv2.circle(pic1, (matchedPoints1[i][0], matchedPoints1[i][1]), radius=7, color=color, thickness=-1)
        newimage2 = cv2.circle(pic2, (matchedPoints2[i][0], matchedPoints2[i][1]), radius=7, color=color, thickness=-1)


    for i in range(len(contents_split) - 1):
        newimage1 = cv2.line(pic1, (matchedPoints1[i][0], matchedPoints1[i][1]), (matchedPoints1[i+1][0], matchedPoints1[i+1][1]),
                            colors[i], 3)
        newimage2 = cv2.line(pic2, (matchedPoints2[i][0], matchedPoints2[i][1]),
                        (matchedPoints2[i + 1][0], matchedPoints2[i + 1][1]),
                        colors[i], 3)

    plt.subplot(1, 2, 1), plt.imshow(newimage1)
    # plt.title('newimage1')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(newimage2)
    # plt.title('newimage2')
    plt.xticks([]), plt.yticks([])
    # plt.show()
    plt.close()

    # reading the camera matrices
    P1 = np.loadtxt('cameraMatrix1.txt', dtype=None, delimiter=',')
    P2 = np.loadtxt('cameraMatrix2.txt', dtype=None, delimiter=',')

    # finding the 3d coordinates of al 22 points using the given DLT function
    p3ds = []
    for uv1, uv2 in zip(matchedPoints1, matchedPoints2):
        _p3d = DLT(P1, P2, uv1, uv2)
        p3ds.append(_p3d)
    p3ds = np.array(p3ds)

    # centering the point cloud
    avgx = 0
    avgy = 0
    avgz = 0
    for i in range(len(p3ds)):
        avgx += p3ds[i][0]
        avgy += p3ds[i][1]
        avgz += p3ds[i][2]
    avgx /= 22
    avgy /= 22
    avgz /= 22

    for i in range(len(p3ds)):
        p3ds[i][0] -= avgx
        p3ds[i][1] -= avgy
        p3ds[i][2] -= avgz

    # plotting the 3d points projected on to 2d coordinate system
    colors2 = []
    for i in range(len(contents_split)):
        color2 = tuple(np.random.uniform(0, 1, 3).tolist())
        colors2.append(color2)
    for i in range(len(p3ds)):
        plt.plot(p3ds[i][0], p3ds[i][1], 'o', c=colors2[i])
    for i in range(len(p3ds) - 1):
        plt.plot([p3ds[i][0], p3ds[i+1][0]], [p3ds[i][1], p3ds[i+1][1]], c=colors2[i])
    # plt.show()
    plt.close()

    # random 3d rotation and plotting the result
    theta = np.random.randint(0, 2*math.pi)
    xrotationMat = np.array([[1, 0, 0], [0, math.cos(theta), -math.sin(theta)], [0, math.sin(theta), math.cos(theta)]])

    for i in range(len(p3ds)):
        p3ds[i] = xrotationMat.dot(np.array(p3ds[i]))

    colors2 = []
    for i in range(len(contents_split)):
        color2 = tuple(np.random.uniform(0, 1, 3).tolist())
        colors2.append(color2)
    for i in range(len(p3ds)):
        plt.plot(p3ds[i][0], p3ds[i][1], 'o', c=colors2[i])
    for i in range(len(p3ds) - 1):
        plt.plot([p3ds[i][0], p3ds[i + 1][0]], [p3ds[i][1], p3ds[i + 1][1]], c=colors2[i])
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("first position.png")
    # plt.show()
    plt.close()

    for i in range(36):
        theta = (2*math.pi / 35)*i
        xrotationMat = np.array([[1, 0, 0], [0, math.cos(theta), -math.sin(theta)], [0, math.sin(theta), math.cos(theta)]])

        for j in range(len(p3ds)):
            p3ds[j] = xrotationMat.dot(np.array(p3ds[j]))

        colors2 = []
        for j in range(len(contents_split)):
            color2 = tuple(np.random.uniform(0, 1, 3).tolist())
            colors2.append(color2)
        for j in range(len(p3ds)):
            plt.plot(p3ds[j][0], p3ds[j][1], 'o', c=colors2[j])
        for j in range(len(p3ds) - 1):
            plt.plot([p3ds[j][0], p3ds[j + 1][0]], [p3ds[j][1], p3ds[j + 1][1]], c=colors2[j])
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig("xrotate" + str(i) + ".png")
        # plt.show()
        plt.close()

    for i in range(36):
        theta = (2*math.pi / 35)*i
        yrotationMat = np.array([[math.cos(theta), 0, math.sin(theta)], [0, 1, 0], [-math.sin(theta), 0, math.cos(theta)]])

        for j in range(len(p3ds)):
            p3ds[j] = yrotationMat.dot(np.array(p3ds[j]))

        colors2 = []
        for j in range(len(contents_split)):
            color2 = tuple(np.random.uniform(0, 1, 3).tolist())
            colors2.append(color2)
        for j in range(len(p3ds)):
            plt.plot(p3ds[j][0], p3ds[j][1], 'o', c=colors2[j])
        for j in range(len(p3ds) - 1):
            plt.plot([p3ds[j][0], p3ds[j + 1][0]], [p3ds[j][1], p3ds[j + 1][1]], c=colors2[j])
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig("yrotate" + str(i) + ".png")
        # plt.show()
        plt.close()
