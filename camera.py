import numpy as np 
import scipy as sp
import cv2
import glob
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


# Global variable 
I = np.identity(3)

# Test values   
alpha = -2.6337
beta = -0.47158
gamma = -1.2795
w1 = 1
w2 = 1
w3 = 0
X = 2
Y = 2
Z = 2

def eulerToR(alpha, beta, gamma):
    # computer R from Euler angles
    #                                           yaw     pitch   roll
    # Equation is -> R(gamma, beta, alpha) = Rz(gamma)Ry(beta)Rx(alpha)
    Rzgamma = np.matrix([[np.cos(gamma), -(np.sin(gamma)), 0], 
                         [np.sin(gamma), np.cos(gamma), 0], 
                         [0, 0, 1]])

    Rybeta = np.matrix([[np.cos(beta), 0, np.sin(beta)], 
                        [0, 1, 0], 
                        [-(np.sin(beta)), 0, np.cos(beta)]])

    Rxalpha = np.matrix([[1, 0, 0], 
                         [0, np.cos(alpha), -(np.sin(alpha))], 
                         [0, np.sin(alpha), np.cos(alpha)]])

    return np.dot(Rzgamma, np.dot(Rybeta, Rxalpha))

def expToR(w1, w2, w3):
    # computer R from exponential coordinates
    # notes page 51
    wangle = np.angle([w1, w2, w3])
    normw = np.linalg.norm([w1, w2, w3], np.inf)

    wcap = np.matrix([[0, -w3, w2], 
                      [w3, 0, -w1], 
                      [-w2, w1, 0]])
    return (np.identity(3) + (wcap/normw)*np.sin(normw) + (wcap**2/(normw)**2)*(1 - np.cos(normw)))


def eulerToT(X, Y, Z, alpha, beta, gamma):
    # computer homogeneous transformation T from 6D pose parameters
    # Euclidean transformation is made up of a translation followed by a rotation
    # (X,Y,Z) are the translation component
    R = eulerToR(alpha, beta, gamma)
    t = np.matrix([[X], [Y], [Z]])
    

    R = np.vstack((R,np.matrix([[0, 0, 0]])))
    t = np.vstack((t, np.matrix([[1]])))
    return np.hstack((R, t))   
 

def expToT(X, Y, Z, w1, w2, w3):
    # computer homogeneous transformation T from 6D pose parameters
    R = expToR(w1, w2, w3)
    t = np.matrix([[X], [Y], [Z]])

    R = np.vstack((R, np.matrix([[0, 0, 0]])))
    t = np.vstack((t, np.matrix([[1]])))
    return np.hstack((R, t))

def intrinsicToK(X, Y, Z, f, k, l, xo, yo):
    # intrinsic_params => fc(2 X 1 vector), c(x, y), alpha_c, 
    # f, [X,Y,Z], Retinal coordinate system x,y,z parallel to [X,Y,Z]
    # P = [X, Y, Z], point = [x, y, z] -> camera coordinate system point = [x, y, z, f]
    f_alpha = k*f; f_beta = l*f
    x = f_alpha*(np.divide(X, Z)) + xo; y = f_beta*(np.divide(Y, Z)) + yo

    return np.matrix([[f_alpha, 0, x], [0, f_beta, y], [0, 0, 1]])

    # A = ([[ax, 0(gamma), u0], [0, ay, v0], [0, 0, 1]])
    # ax = f.mx, ay = f.my, mx and my are scale factors and f is focal length
    # gamma represents the skew coefficient -> often 0
    # u0 and v0 represent the principal point

def simulateCamera(l1, l2, l3):
    # origin is on the ground plane : therefore P = (X, Y, Z, -2)
    # Pcam = RtP
    # Pwrl = [[X], [2 meters above ground plane], [Z]]

    # Plane = (ax, by, cz, -d)
    # P = (X, Y, X, -2)
    # Coordinate transformation => P' = (homogeneous 6 digit params)P

    # Parallel lines => (1, 2, 0) (2, 4, 0) (4, 8, 0)
    # ax + by + cz -d = 0
    
    # X = np.matrix([[0], [1], [0]])
    # Y = np.matrix([[0], [0], [0]])
    # Z = np.matrix([[0], [-1], [0]])
    # rotation along the x-axis by 90 degrees
    # -> alpha = np.pi/2

    # ax + by + cz - d = 0
    # b - d = 0
    # - d = 0
    # -b - d = 0

    p1 = np.matrix([[0], [l1], [0]])
    p2 = np.matrix([[0], [l2], [0]])
    p3 = np.matrix([[0], [l3], [0]])

    alpha_ = 0
    beta_ = 0
    gamma_ = -(np.pi/2)

    # R = eulerToR(alpha_, beta_, gamma_)

    # euler angles...............check
    # 

    t = eulerToT(l1, l2, l3, alpha_, beta_, gamma_)

    # need f and image points
    # f = 0.400
    # k = l = 0.023

    # origin as zero
    # xo = yo = 0

    # k = intrinsicToK(p1.item(0), p1.item(1), p1.item(2), f, k, l, xo, yo)

    return t

################################################################################################
################################################################################################

# QUESTION 2 #

data = np.loadtxt('data.txt')

def calibrateCamera3D(data):
    # (X, Y, Z) => data[:,0] data[:,1] data[:,2]
    # (x, y) => data[:,3] data[:,4]

    X = data[:,0]
    Y = data[:,1]
    Z = data[:,2]
    x = data[:,3]
    y = data[:,4]

    # need to get Pm = 0
    A = np.matrix([[X[0], Y[0], Z[0], 1, 0, 0, 0, 0, -x[0]*(X[0]), -x[0]*(Y[0]), -x[0]*(Z[0]), -x[0]], 
                    [0, 0, 0, 0, X[0], Y[0], Z[0], 1, -y[0]*(X[0]), -y[0]*(Y[0]), -y[0]*(Z[0]), -y[0]]])

    # finishes creating the P matrix
    for k in range(1, len(data)):
        A_r = np.matrix([[X[k], Y[k], Z[k], 1, 0, 0, 0, 0, -x[k]*(X[k]), -x[k]*(Y[k]), -x[k]*(Z[k]), -x[k]], 
                    [0, 0, 0, 0, X[k], Y[k], Z[k], 1, -y[k]*(X[k]), -y[k]*(Y[k]), -y[k]*(Z[k]), -y[k]]])
        A = np.r_[A, A_r]

    A = A.T * A
    eigenValues,eigenVectors = np.linalg.eig(A)

    # idx = eigenValues.argsort()[::-1]   
    # eigenValues = eigenValues[idx]
    # eigenVectors = eigenVectors[:,idx]
    idx = eigenValues.argsort()
    # print idx
    # smallest eigenvalue is at the end of the array
    # print idx[len(idx)-1]
    smallest = eigenVectors[len(eigenVectors)-1]
    
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    smallest = eigenVectors[:,idx]

    M = np.matrix([[smallest.item(0), smallest.item(1), smallest.item(2), smallest.item(3)],
                    [smallest.item(4), smallest.item(5), smallest.item(6), smallest.item(7)],
                    [smallest.item(8), smallest.item(9), smallest.item(10), smallest.item(11)]])

    B = np.matrix([[M.item(0), M.item(1), M.item(2)],
                    [M.item(4), M.item(5), M.item(6)],
                    [M.item(8), M.item(9), M.item(10)]])

    b = np.matrix([[M.item(3)],
                    [M.item(7)],
                    [M.item(11)]])

    K = B * B.T
    uo = K.item(2)
    vo = K.item(5)
    beta_ = math.sqrt(K.item(4) - vo**2)
    gamma_ = (K.item(3) - uo * vo)/ beta_
    alpha_ = math.sqrt(K.item(0) - uo**2 - gamma_**2)

    R = np.linalg.inv(K) * B
    t = np.linalg.inv(K) * b

    print "\nR:\n", R
    print "\nt:\n", t

    P = np.hstack((R, t))
    P = np.vstack((P, np.matrix([0, 0, 0, 1])))
    return P


def visualiseCameraCalibration3D(data, P):
    # # first calculate the 3 x 3 matrix P by B
    # # then the last column b
    # B = np.matrix([[P.item(0), P.item(1), P.item(2)], 
    #               [P.item(4), P.item(5), P.item(6)], 
    #               [P.item(8), P.item(9), P.item(10)]])
    # b = np.matrix([[P.item(3)], [P.item(7)], [P.item(11)]])

    # # B = AR
    # # b = At

    # # K = BB(transpose)
    # K = B * B.T
    # print "\nK:\n", K

    # uo_ = K.item(2)
    # vo_ = K.item(5)
    # beta_ = math.sqrt(K.item(4)-(vo_**2))
    # gamma_ = (K.item(1) - (uo_ * vo_))/beta_
    # alpha_ = math.sqrt(K.item(0) - (uo_**2) - (gamma_**2))

    # print "\nuo\n", uo_
    # print "\nvo\n", vo_
    # print "\nalpha\n", alpha_
    # print "\nbeta\n", beta_
    # print "\ngamma\n", gamma_

    # print "\nB:\n", B
    # print "\nb:\n", b

    # R = eulerToR(alpha_, beta_, gamma_)
    # A = R(inverse)B
    # A = np.linalg.inv(R)*B
    # print 

    # x = np.matrix([[data[0][0]], [data[0][1]], [data[0][2]], [1]])
    # m = np.dot(P, x)
    # print "\nM", m
    # new_data = np.array([[m.item(0), m.item(1), m.item(2)]])
    # # new_data = []
    # for i in data[1:]:
    #   x = np.matrix([[i.item(0)], [i.item(1)], [i.item(2)], [1]])
    #   m = P.dot(x)
    #   new_x = np.array([[m.item(0), m.item(1), m.item(2)]])
    #   new_data = np.r_[new_data, new_x]   

    # print new_data[:,2]
    # m_points = np.toarray(list_points)
    # ax.plot(new_data[:,0], new_data[:,1], new_data[:,2],'k.')
    # fig = plt.figure()
    # ax = fig.gca(projection="3d")
    # x = ax.plot(P[:,0], P[:,1], P[:,2],'k.')

    x = np.matrix([[data[0][0]], [data[0][1]], [data[0][2]], [1]])
    new_x = np.dot(P, x)
    print new_x
    new_data = np.array([[new_x.item(0), new_x.item(1), new_x.item(2)]])
    for i in data[1:]:
        x = np.matrix([[i.item(0)], [i.item(1)], [i.item(2)], [1]])
        m = P.dot(x)
        new_x = np.array([[m.item(0), m.item(1), m.item(2)]])
        new_data = np.r_[new_data, new_x]

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot(new_data[:,0], new_data[:,1], new_data[:,2], '.k')

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(data[:,3], data[:,4],'r.')

    plt.show()


def evaluateCameraCalibration3D(data, P):
    orig_X = data[:,0]
    orig_Y = data[:,1]
    orig_Z = data[:,2]

    new_X = P[:,0]
    new_Y = P[:,1]
    new_Z = P[:,2]

    orig_points = np.array([[orig_X], [orig_Y], [orig_Z]])
    orig_points = orig_points.T
    new_points = np.array([[new_X], [new_Y], [new_Z]])
    new_points = new_points.T
    # print "\nmean of P: \n", np.mean(P)
    # print "\nsd of P: \n", np.std(P)

#########################################################################################
#########################################################################################

#termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#prepare object points, like (0,0,0), (1,0,0) ...., (6, 4, 0)
pattern_size = (7, 6)
pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 )
pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)

objp = np.zeros((6*7, 3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
# print len(pattern_points), len(objp)

#Arrays to store objects points and real image points from all the images.
objpoints = [] #3D point in real world space
imgpoints = [] #2D points in image plane

images = glob.glob('*.jpg')

def computeCameraMatrix():
    counter = int(x=1)

    h, w = 0, 0
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        #Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size,None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            
            imgpoints.append(corners)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, pattern_size, corners,ret)
            cv2.imshow('img',img)

            rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h))

            newcameramtx, roi=cv2.getOptimalNewCameraMatrix(camera_matrix,dist_coefs,(w,h),1,(w,h))
            
            dst = cv2.undistort(gray, camera_matrix, dist_coefs, None, newcameramtx)
            cv2.imshow('undistort image', dst)
            cv2.waitKey(100)
            counter = counter + 1
        else:
            print("No corners found on Picture " + str(counter))

    cv2.destroyAllWindows()


def viewCamera():
    objpoints = []
    imgpoints = []
    cap = cv2.VideoCapture(0)

    print "\nPress 'q' to quit\nPress 'h' to compute Homography\nPress 'p' to project Image onto chessboard - error!!!"
    count = 0

    computeH = False
    projectImage = False
    while True:
        ret, img = cap.read()
        if count == 0:
            cameraList = computeCameraValues(img)
            global camera_matrix
            camera_matrix = cameraList[0]
            global newcameramtx
            newcameramtx = cameraList[1]
            global dist_coefs
            dist_coefs = cameraList[2]
            count = count + 1

        original = img
        img = undistortImage(img, camera_matrix, newcameramtx, dist_coefs)

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6))

        ret_landscape, corners_landscape = cv2.findChessboardCorners(gray, (6, 9))
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

            imgpoints.append(corners)

            pattern_size = (9, 6)
            if projectImage == True:
                img = imageProjection(original, img, pattern_size)
            else:
                if computeH == True:
                    img = computeHomographyChess(original, img, pattern_size)

                else:
                    # Draw and display the corners
                    cv2.drawChessboardCorners(img, (9, 6), corners,ret)
        elif ret_landscape == True:
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners_landscape,(11,11),(-1,-1),criteria)

            imgpoints.append(corners) 

            pattern_size = (6, 9)
            if projectImage == True:
                img = imageProjection(original, img, pattern_size)
            else:
                if computeH == True:
                    img = computeHomographyChess(original, img, pattern_size)
                else:
                    cv2.drawChessboardCorners(img, (6,9), corners_landscape,ret_landscape)                
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.waitKey(1) & 0xFF == ord('h'):
            computeH = True

        if cv2.waitKey(1) & 0xFF == ord('p'):
            projectImage = True

        cv2.imshow('undistorted',img)

    cv2.destroyAllWindows()

def undistortImage(img, camera_matrix, newcameramtx, dist_coefs):
    undistortImage = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

    return undistortImage

def computeCameraValues(img):
    h, w = img.shape[:2]
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h))

    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(camera_matrix,dist_coefs,(w,h),1,(w,h))
    return [camera_matrix, newcameramtx, dist_coefs]

def printCameraValues():
    global camera_matrix
    global dist_coefs
    print "\nIntrinsic: \n", camera_matrix
    print "\nDistortion: \n", dist_coefs  

def computeHomographyChess(src_img, undst_img, pattern_size):
    ret, corners = cv2.findChessboardCorners(src_img, pattern_size,None)
    srcp = np.array([corners[0,0], corners[8,0], corners[45,0], corners[corners.shape[0]-1,0]])

    ret, corners_ = cv2.findChessboardCorners(undst_img, pattern_size,None)
    if ret:
        dstp = np.array([corners_[0,0], corners_[8,0], corners_[45,0], corners_[corners.shape[0]-1,0]])

    global H
    H, mask = cv2.findHomography(srcp, dstp)

    return drawCorrespondents(undst_img, srcp, dstp)

def computeHomographyImage(src_img, dst_img, pattern_size):
    ret, corners = cv2.findChessboardCorners(src_img, pattern_size,None)
    srcp = np.array([corners[0,0], corners[8,0], corners[45,0], corners[corners.shape[0]-1,0]])

    dstp = np.array([ [0.0, 0.0], [0.0, (dst_img.shape[1]-1) + 0.0], 
        [(dst_img.shape[0]-1) + 0.0, (dst_img.shape[1]-1) + 0.0], [(dst_img.shape[0]-1) + 0.0, 0.0]])  
    
    print srcp, "\n", dstp
    H, mask = cv2.getPerspectiveTransform(srcp, dstp)
    return H  

def drawCorrespondents(img, src_points, dst_points):
    # Corresponding points are shown in green
    cv2.line(img, (src_points[0][0], src_points[0][1]), (dst_points[0][0], dst_points[0][1]), (0, 255, 0), 3)
    cv2.line(img, (src_points[1][0], src_points[1][1]), (dst_points[1][0], dst_points[1][1]), (0, 255, 0), 3)
    cv2.line(img, (src_points[2][0], src_points[2][1]), (dst_points[2][0], dst_points[2][1]), (0, 255, 0), 3)
    cv2.line(img, (src_points[3][0], src_points[3][1]), (dst_points[3][0], dst_points[3][1]), (0, 255, 0), 3)

    return img

def printHomography():
    global H
    print "\nHomography: \n", H

def imageProjection(src_img, dst_img, pattern_size):
    logo = cv2.imread('logo.png')
    cv2.imshow('l', logo)
    cv2.waitKey(0)
    img = computeHomographyImage(dst_img, logo, pattern_size)

    global H
    img = cv2.perspectiveTransform(img, H, logo)
    return img


# Question 1
# A
R = eulerToR(alpha, beta, gamma)
print('Rotation matrix R is:')
print(R)

R = expToR(w1, w2, w3)
print "\nExponential to R:"
print(R)
# print "\n"

T = eulerToT(X, Y, Z, alpha, beta, gamma)
print "\nEuler to T:"
print T

T = expToT(X, Y, Z, w1, w2, w3)
print "\nExponential to T:"
print T

# B
K = intrinsicToK(X, Y, Z, 0.400, .02, .02,  2, 2)
print "\nIntrinsic parameters: "
print K

# C
S = simulateCamera(1, 0, -1)
print "\nSimulate Camera matrix result:"
print S

# Question 2 #
P =calibrateCamera3D(data)
print "\n3D P:"
print P

visualiseCameraCalibration3D(data, P)
evaluateCameraCalibration3D(data, P)


# OpenCV functions #

# Set camera matrix values and distortion coeffictients
# Needs to run
computeCameraMatrix()

viewCamera()

printCameraValues()

# Error if Homography has not been set
printHomography()

