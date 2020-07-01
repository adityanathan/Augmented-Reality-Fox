import argparse
import glob
import cv2
import numpy as np
import math
import os

MIN_MATCHES = 15
TAU_C = 32
TAU_I = 0.02
BINARIZATION_THRESHOLD = 127
CONTOUR_APPROX = 0.015


def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))


def render(img, obj, projection, model, color=False, shift_x=0, shift_y=0):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array(
            [[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        dst = [[[shift_x+p[0][0], shift_y+p[0][1]]]for p in dst]
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img


def draw(img, corners, imgpts):
    # print(corners)
    # print(imgpts)
    corner = (corners[0], corners[1])
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 0, 255), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 255, 255), 5)
    return img


def convert_to_ndarray(x):
    return np.array(x).reshape((-1, 1, 2)).astype(np.int32)


def convert2grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def create_list(root_folder, gray=True):
    lst = []
    for file in glob.glob(root_folder+"/model*.png"):
        # print(file)
        img = cv2.imread(file)
        if gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img is not None:
            lst.append(img)
        else:
            print('Error reading this file:'+file)
    return lst


def find_model_keypoints(model_list):
    # sift = cv2.xfeatures2d.SIFT_create()
    orb = cv2.ORB_create()
    ans = []
    for x in model_list:
        # kp_model, des_model = sift.detectAndCompute(x, None)
        kp_model = orb.detect(x, None)
        des_model = orb.compute(x, kp_model)
        ans.append((kp_model, des_model))
    return ans


def preprocess(img):
    img = binarize_image(img)
    img = resize_img(img)
    return(img)


def resize_img(img):
    # Image Resizing
    dim_r, dim_c = img.shape[0], img.shape[1]
    tau_prime = TAU_C + max(dim_r, dim_c)*TAU_I
    resize_factor = TAU_C/tau_prime
    img = cv2.resize(img, None, fx=resize_factor,
                     fy=resize_factor, interpolation=cv2.INTER_AREA)
    return img


def binarize_image(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bilateralFilter(img, 11, 17, 17)
    ret, edged = cv2.threshold(
        img, BINARIZATION_THRESHOLD, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # edged = cv2.Canny(img, 30, 200)

    return edged


def find_contours(img, color_img=None):
    '''
    Image has to be resized(for efficiency), thresholded/canny edge detected
    and img - grayscale, color_img-BGR
    '''
    # find contours in the edged image, keep only the valid
    # ones, and initialize our screen contour
    # _, cnts, hierarchy = cv2.findContours(
    cnts, hierarchy = cv2.findContours(
        img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, CONTOUR_APPROX * peri, True)

        if len(approx) == 4 and peri > TAU_C*4:
            screenCnt.append(approx)
    if len(screenCnt) > 0:
        return screenCnt[:2]
    else:
        return None


def crop_without_perspective(img, pts):
    pts_list = []
    for i in range(len(pts)):
        pts_list.append((pts[i][0][0], pts[i][0][1]))
        rect = cv2.boundingRect(pts)
        x, y, w, h = rect
        cropped = img[y:y+h, x:x+w].copy()

        return cropped


def crop_img(img, pts):
    '''
    crop_img
    Given an image and 4 points it performs perspective transform
    and crops the rest of the image to return only the region within
    the 4 points.
    img - source image
    pts - 4x1x2 matrix
    Returns a 300*300 image
    '''

    if(pts.shape == (4, 1, 2)):
        pts = np.float32([x[0] for x in pts])

    pts2 = np.float32([[0, 0], [0, 300], [300, 300], [300, 0]])
    M = cv2.getPerspectiveTransform(pts, pts2)
    warped_image = cv2.warpPerspective(img, M, (300, 300))
    return warped_image


def downsample(img):
    return cv2.pyrDown(img)


def find_downsample_iterations(img):
    # Returns number of times the image has to be downsampled in the image pyramid
    r, c = img.shape[0], img.shape[1]
    min = abs((r*c)-(TAU_C*TAU_C))
    count = 0
    while(True):
        r = r/2
        c = c/2
        temp = abs((r*c)-(TAU_C*TAU_C))
        if(temp < min):
            min = temp
            count += 1
        else:
            break
    return count


def find_matches(scene, model_list, model_keypoints, cropped_img, camera_calib, dist):
    # cv2.imshow('scene', scene)
    # cv2.waitKey(0)
    count = 0
    for model_i in model_list:
        # cv2.imshow('model', model_i)
        # cv2.waitKey(0)
        # brisk = cv2.BRISK_create(70, 4)
        # sift = cv2.xfeatures2d.SIFT_create()
        orb = cv2.ORB_create()

        # Brute force matching
        bf = cv2.BFMatcher()
        # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        kp_model, des_model = model_keypoints[count]
        # print(model_keypoints[count])
        # kp_scene, des_scene = sift.detectAndCompute(scene, None)
        kp_scene = orb.detect(scene, None)
        des_scene = orb.compute(scene, kp_scene)

        matches = []
        if(len(kp_scene) > 0 and len(kp_model) > 0):
            # matches = bf.match(des_model, des_scene)
            # matches = bf.knnMatch(des_model, trainDescriptors=des_scene, k=2)
            matches=bf.knnMatch(des_model[1], des_scene[1], k=2)
            good_matches = []
            try:
                for m, n in matches:
                    if m.distance < .75 * n.distance:
                        good_matches.append(m)
                matches = good_matches
                frame = cv2.drawMatches(
                    model_i, kp_model, scene, kp_scene, matches[:10], 0, flags=2)
            except:
                pass
            # print(len(matches))
            # cv2.imshow('frame', frame)
            # cv2.waitKey(0)
        if(len(matches) > MIN_MATCHES):
            # cv2.imshow('frame', cropped_img)
            # cv2.waitKey(0)
            return True, model_i
        count += 1
    return False, None


def upsample_corner(cnt, down_dim, dim):
    new_corners = []
    for x in cnt:
        pt_x = x[0][0]
        pt_y = x[0][1]
        new_x = int(pt_x*dim[1]/down_dim[1])
        new_y = int(pt_y*dim[0]/down_dim[0])
        pos = [[new_x, new_y]]
        new_corners.append(pos)
    return new_corners


def find_marker(img, model_list, model_keypoints, camera_calib, dist):
    '''
    Image has to be resized.
    '''
    original_img = img.copy()
    original_dim = (img.shape[0], img.shape[1])
    img = resize_img(img)
    current_dim = (img.shape[0], img.shape[1])
    n = find_downsample_iterations(img)
    contours = []
    homographies = []
    mod = []
    cnt = find_contours(binarize_image(img.copy()))
    if cnt is not None:
        for x in cnt:
            img2 = crop_img(img.copy(), x)
            img3 = crop_without_perspective(original_img.copy(), np.float32(upsample_corner
                                                                            (x, current_dim, original_dim)))
            match = find_matches(
                img2, model_list, model_keypoints, img3, camera_calib, dist)
            if(match[0]):
                cnt_match = upsample_corner(
                    x, current_dim, original_dim)
                contours.append(cnt_match)
                model_matched = match[1]
                row, col = model_matched.shape[0], model_matched.shape[1]
                # print(type(cnt_match[0][0]))
                pts1 = np.float32([[0, 0], [0, row], [col, row], [col, 0], [
                                  0, row/2], [col/2, row], [col, row/2], [col/2, 0]])
                p1 = np.float32(cnt_match[0][0])
                p2 = np.float32(cnt_match[1][0])
                p3 = np.float32(cnt_match[2][0])
                p4 = np.float32(cnt_match[3][0])
                pts2 = np.float32(
                    [p1, p2, p3, p4, (p1+p2)/2, (p2+p3)/2, (p3+p4)/2, (p4+p1)/2])
                homography, mask = cv2.findHomography(
                    pts1, pts2, cv2.RANSAC, 5.0)
                homographies.append(homography)
                mod.append(model_matched)

        for i in range(0, n):
            img = downsample(img)
            current_dim = (img.shape[0], img.shape[1])
            cnt = find_contours(binarize_image(img.copy()))
            if cnt is not None:
                for x in cnt:
                    img2 = crop_img(img.copy(), np.float32(x))
                    img3 = crop_without_perspective(original_img.copy(), np.float32(upsample_corner(
                        x, current_dim, original_dim)))
                    match = find_matches(
                        img2, model_list, model_keypoints, img3, camera_calib, dist)
                    if(match[0]):
                        cnt_match = upsample_corner(
                            x, current_dim, original_dim)
                        if(len(contours) < 2 and not similar_rectangle_in_lst(cnt_match, contours)):
                            contours.append(cnt_match)
                            model_matched = match[1]
                            row, col = model_matched.shape[0], model_matched.shape[1]
                            # print(type(cnt_match[0][0]))
                            pts1 = np.float32([[0, 0], [0, row], [col, row], [col, 0], [
                                0, row/2], [col/2, row], [col, row/2], [col/2, 0]])
                            p1 = np.float32(cnt_match[0][0])
                            p2 = np.float32(cnt_match[1][0])
                            p3 = np.float32(cnt_match[2][0])
                            p4 = np.float32(cnt_match[3][0])
                            pts2 = np.float32(
                                [p1, p2, p3, p4, (p1+p2)/2, (p2+p3)/2, (p3+p4)/2, (p4+p1)/2])
                            homography, mask = cv2.findHomography(
                                pts1, pts2, cv2.RANSAC, 5.0)
                            homographies.append(homography)
                            mod.append(model_matched)

    print('Contours:'+str(len(contours)))
    if len(contours) > 0:
        return (contours, homographies, mod)
    else:
        return None, None, None


def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d /
                   np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d /
                   np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    print("Rotation")
    print(rot_1, rot_2, rot_3)
    print("______________________")
    print("Translation")
    print(translation)
    print("______________________")
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)


def camera_calibration(root_folder):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    cbrow = 9
    cbcol = 6

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((cbcol*cbrow, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cbrow, 0:cbcol].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob(root_folder+'/calib_*.png')

    for fname in images:
        img = cv2.imread(fname, 0)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(img, (cbrow, cbcol), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(
                img, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

    #         # Draw and display the corners
    #         img = cv2.drawChessboardCorners(img, (cbrow, cbcol), corners2, ret)
    #         cv2.imshow('img', img)
    #         cv2.waitKey(0)

    # cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img.shape[::-1], None, None)
    return mtx, dist


def xmax(a):
    return max(a, key=lambda x: x[0][0])[0][0]


def xmin(a):
    return min(a, key=lambda x: x[0][0])[0][0]


def ymax(a):
    return max(a, key=lambda x: x[0][1])[0][1]


def ymin(a):
    return min(a, key=lambda x: x[0][1])[0][1]


def overlap_area(a, b):  # returns None if rectangles don't intersect
    dx = min(xmax(a), xmax(b)) - max(xmin(a), xmin(b))
    dy = min(ymax(a), ymax(b)) - max(ymin(a), ymin(b))
    return abs(dx*dy)


def union_area(a, b):
    dx = max(xmax(a), xmax(b)) - min(xmin(a), xmin(b))
    dy = max(ymax(a), ymax(b)) - min(ymin(a), ymin(b))
    return abs(dx*dy)


def similar_rectangle_in_lst(a, lst):
    for l in lst:
        overlap = overlap_area(a, l)
        rect_area = union_area(a, l)
        if overlap is not None:
            if ((overlap/rect_area) > 0.85):
                # print(overlap/rect_area)
                return True
    return False
