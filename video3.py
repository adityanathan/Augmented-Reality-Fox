import argparse
import glob
import cv2
import numpy as np
import math
import os
import helper as he
from objloader_simple import *
# import imutils

MIN_MATCHES = 15
TAU_C = 32
TAU_I = 0.02
BINARIZATION_THRESHOLD = 200
CONTOUR_APPROX = 0.015
STEP_SIZE = 1

# Command line argument parsing
# NOT ALL OF THEM ARE SUPPORTED YET
parser = argparse.ArgumentParser(description='Augmented reality application')

parser.add_argument('-r', '--rectangle',
                    help='draw rectangle delimiting target surface on frame and axes', action='store_true')

parser.add_argument('-p', '--problem',
                    help='specify problem no.', action='store_true')

args = parser.parse_args()


def main():
    # cap_img = cv2.imread('Images/Test_Markers/7.jpeg')
    camera_parameters, dist = he.camera_calibration('Images/Calibration')
    dir_name = os.getcwd()
    obj = OBJ(os.path.join(dir_name, 'Models/fox/fox.obj'), swapyz=True)
    model_list = he.create_list('Images')
    model_keypoints = he.find_model_keypoints(model_list)

    # print('b')
    if not args.problem:
        cap = cv2.VideoCapture(0)
        while True:
            # read the current frame
            ret, frame = cap.read()
            if not ret:
                print("Unable to capture video")
                return
            cnt = None
            temp = he.find_marker(
                he.convert2grayscale(frame), model_list, model_keypoints, camera_parameters, dist)
            if temp is not None:
                cnt, homographies, models = temp
            # compute Homography if enough matches are found
            if cnt is not None:
                # if a valid homography matrix was found render cube on model plane
                for x, h, m in zip(cnt, homographies, models):
                    try:
                        projection = he.projection_matrix(
                            camera_parameters, h)

                        frame = he.render(frame, obj, projection, m, False)

                        # rect = cnt[0]
                        # pseudo_model = h.crop_img(frame.copy(), rect)
                        # r_p, c_p = pseudo_model.shape[0], pseudo_model.shape[1]
                        # pts = [x[0] for x in rect]
                        # pts2 = [[0, 0], [0, r_p], [c_p, r_p], [c_p, 0]]
                        # projection = cv2.getPerspectiveTransform(pts, pts2)
                        break
                    except:
                        pass
                # draw first 10 matches.
                if args.rectangle:
                    if cnt is not None:
                        for x in cnt:
                            x = he.convert_to_ndarray(x)
                            cv2.drawContours(frame, [x], -1, (0, 255, 0), 3)

                            points_dum = np.array(
                                [[[0, 0, 0]], [[100, 0, 0]], [[0, 100, 0]], [[0, 0, 100]]])

                            points_dum = np.float32([np.float32(p)
                                                     for p in points_dum])
                            # print(axis)
                            newpts = cv2.perspectiveTransform(
                                points_dum, projection)

                            cap_img = he.draw(frame, newpts[0][0], [
                                newpts[1][0], newpts[2][0], newpts[3][0]])
                            break
                # show result
            else:
                print('No marker found')

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

    elif args.problem:
        # print('c')
        no = input('Enter image no. to process: ')
        # img nos -
        cap_img = cv2.imread('Images/Test_Markers/Prob2/'+str(no)+'.jpeg')
        camera_parameters, dist = he.camera_calibration('Images/Calibration')
        model_list = he.create_list('Images')
        model_keypoints = he.find_model_keypoints(model_list)
        cnt = None
        temp = he.find_marker(
            he.convert2grayscale(cap_img), model_list, model_keypoints, camera_parameters, dist)
        if temp is not None:
            cnt, homographies, models = temp
        cnt = [cnt[1], cnt[0]]
        homographies = [homographies[1], homographies[0]]
        models = [models[1], models[0]]
        if cnt is not None:
            if(len(cnt) == 2):
                ro1, co1 = models[0].shape
                ro2, co2 = models[1].shape

                proj1 = he.projection_matrix(
                    camera_parameters, homographies[0])
                proj2 = he.projection_matrix(
                    camera_parameters, homographies[1])
                points1 = np.array(
                    [[[co1/2, ro1/2, 0]]])
                points1 = np.float32([np.float32(p) for p in points1])
                origin1 = np.array(
                    cv2.perspectiveTransform(points1, proj1)[0][0])

                points2 = np.array(
                    [[[0, 0, 0]], [[100, 0, 0]], [[0, 100, 0]], [[0, 0, 100]]])
                points2 = np.float32([np.float32(p) for p in points2])
                axes2 = cv2.perspectiveTransform(points2, proj2)
                origin2 = axes2[0][0]
                ptz = axes2[3][0]

                ######
                if args.rectangle:
                    projection = proj1
                    points_dum = np.array(
                        [[[0, 0, 0]], [[100, 0, 0]], [[0, 100, 0]], [[0, 0, 100]]])
                    points_dum = np.float32([np.float32(p)
                                             for p in points_dum])
                    newpts = cv2.perspectiveTransform(
                        points_dum, projection)

                    cap_img = he.draw(cap_img, newpts[0][0], [
                        newpts[1][0], newpts[2][0], newpts[3][0]])

                    projection = proj2
                    points_dum = np.array(
                        [[[0, 0, 0]], [[100, 0, 0]], [[0, 100, 0]], [[0, 0, 100]]])
                    points_dum = np.float32([np.float32(p)
                                             for p in points_dum])
                    newpts = cv2.perspectiveTransform(
                        points_dum, projection)

                    cap_img = he.draw(cap_img, newpts[0][0], [
                        newpts[1][0], newpts[2][0], newpts[3][0]])
                ######

                origin2 = np.array(origin2)

                origin1_x = origin1[0]
                origin1_y = origin1[1]
                origin2_x = origin2[0]
                ptz_x = ptz[0]
                origin2_y = origin2[1]
                ptz_y = ptz[1]
                moving_slope = (ptz_y-origin2_y)/(ptz_x-origin2_x)
                # moving_slope = (origin2_y-origin1_y)/(origin2_x-origin1_x)
                moving_positive = origin1 + STEP_SIZE * \
                    np.array([1, moving_slope])
                moving_negative = origin1 - STEP_SIZE * \
                    np.array([1, moving_slope])
                distance_pos = np.linalg.norm(origin2-moving_positive)
                distance_neg = np.linalg.norm(origin2-moving_negative)
                if(distance_pos < distance_neg):
                    counter = 0
                    old_distance = math.inf
                    while True:
                        print('Slope:'+str(moving_slope))
                        frame = cap_img.copy()
                        moving_positive = origin1 + counter * STEP_SIZE * \
                            np.array([1, moving_slope])
                        new_distance = np.linalg.norm(origin2-moving_positive)
                        print('new:'+str(new_distance))
                        print('old:'+str(old_distance))
                        if(new_distance > old_distance):
                            print('break')
                            break
                        old_distance = new_distance
                        counter += 1
                        shift = moving_positive - origin1
                        frame = he.render(
                            frame, obj, proj1, models[0], False, shift[0], shift[1])
                        print(shift)
                        cv2.imshow('Moving_obj', frame)
                        cv2.waitKey(1)
                else:
                    counter = 0
                    old_distance = math.inf
                    while True:
                        print('Slope:'+str(moving_slope))
                        frame = cap_img.copy()
                        moving_negative = origin1 - counter * STEP_SIZE * \
                            np.array([1, moving_slope])
                        new_distance = np.linalg.norm(origin2-moving_negative)
                        print('new:'+str(new_distance))
                        print('old:'+str(old_distance))
                        if(new_distance > old_distance):
                            print('break')
                            break
                        old_distance = new_distance
                        counter += 1
                        shift = moving_negative - origin1
                        frame = he.render(
                            frame, obj, proj1, models[0], False, shift[0], shift[1])
                        print(shift)
                        cv2.imshow('Moving_obj', frame)
                        cv2.waitKey(1)
            else:
                print('Less than 2 Markers Found')
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    main()
