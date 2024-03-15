import time

import cv2
import numpy as np

cam = cv2.VideoCapture('Lane Detection Test Video-01.mp4')

left_top_x = 99
left_bottom_x = 0
right_top_x = 290
right_bottom_x = 0


while True:

    # 1
    start = time.perf_counter()

    ret, frame = cam.read()
    if ret is False:
        break
    # fps = cam.get(cv2.CAP_PROP_FPS)
    # 2
    # print(fps)
    scale = 30
    OrigibalW = frame.shape[1]
    OriginalH = frame.shape[0]
    width = int(OrigibalW * scale / 100)
    height = int(OriginalH * scale / 100)
    dim = (width, height)
    frame = cv2.resize(frame, dim)
    cv2.imshow('Color', frame)
    # 3

    blank = np.zeros((height, width), dtype=np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # for i in range(0, height):
    #     for j in range(0, width):
    #         blank[i][j] = frame[i][j]

    # 4
    cv2.imshow('Gray', gray)
    # 0.78 0.43 0.53
    cv2.imshow('Color', frame)
    y = int(height * 0.78)
    upper_left = (int(width * 0.43), y)
    upper_right = (int(width * 0.53), y)
    lower_left = (0, int(height))
    lower_right = (int(width), int(height))
    trapezoid = np.array((upper_right, upper_left, lower_left, lower_right), dtype=np.int32)
    trapezoid_frame = cv2.fillConvexPoly(blank, trapezoid, 1)
    # for i in range(0, height):
    #    for j in range(0, width):
    #        a = i**j

    gray = gray * trapezoid_frame

    cv2.imshow('Trapez gray', gray)

    # 5

    trapezoid_bounds = np.float32([upper_right, upper_left, lower_left, lower_right])
    screen_bounds = np.float32([(width, 0), (0, 0), lower_left, lower_right])
    magic_matrix = cv2.getPerspectiveTransform(trapezoid_bounds, screen_bounds)
    stretched = cv2.warpPerspective(gray, magic_matrix, (width, height))
    cv2.imshow('Stretched', stretched)

    # 6

    blurred = cv2.blur(stretched, ksize=(13, 13))
    cv2.imshow('Blurred', blurred)

    # 7

    sobel_vertical = np.float32([[-1, -2, -1],
                                 [0, 0, 0],
                                 [1, 2, 1]])
    sobel_horizontal = np.transpose(sobel_vertical)
    blurred32 = np.float32(blurred)
    filteredV = cv2.filter2D(blurred32, -1, sobel_vertical)
    filteredH = cv2.filter2D(blurred32, -1, sobel_horizontal)
    filtered = np.sqrt(filteredV**2+filteredH**2)
    filtered8 = cv2.convertScaleAbs(filtered)
    cv2.imshow('Filtered8', filtered8)

    # 8

    threshold = int(255/12)  # 12
    ret2, binarizat = cv2.threshold(filtered8, threshold, 255, cv2.THRESH_BINARY)

    cv2.imshow('Binarizat', binarizat)

    # 9

    binarizat_copy = binarizat.copy()

    percent95W = int(width * 0.97)
    percent5W = int(width * 0.03)
    percent97H = int(height * 0.96)
    percent3H = int(height * 0.03)

    binarizat_copy[:, 0:percent5W] = 0
    binarizat_copy[:, percent95W:width] = 0
    binarizat_copy[percent97H:height, :] = 0

    cv2.imshow('Cropped', binarizat_copy)

    xL = binarizat_copy[:, 0:(int(width/2))]
    xR = binarizat_copy[:, (int(width/2)):width]
    array_left = np.argwhere(xL == 255)
    array_right = np.argwhere(xR == 255)
    array_right[:, 1] += int(width/2)

    # print(int(width))

    left_xs = array_left[:, 1]
    left_ys = array_left[:, 0]

    right_xs = array_right[:, 1]
    right_ys = array_right[:, 0]

    # 10

    right_line = np.polynomial.polynomial.polyfit(right_xs, right_ys, deg=1)
    left_line = np.polynomial.polynomial.polyfit(left_xs, left_ys, deg=1)

    left_top_y = 0

    prevTL = left_top_x
    left_top_x = ((left_top_y - left_line[0]) / left_line[1])
    if left_top_x < -10 or left_top_x > int(width/2):
        left_top_x = prevTL

    # print(left_top_x)

    left_bottom_y = height

    prevBL = left_bottom_x
    left_bottom_x = ((left_bottom_y - left_line[0]) / left_line[1])
    if left_bottom_x < -10 or left_bottom_x > int(width/2):
        left_bottom_x = prevBL

    right_top_y = 0

    prevTR = right_top_x
    right_top_x = ((right_top_y - right_line[0]) / right_line[1])
    if right_top_x < int(width/2)-20 or right_top_x > int(width)+20:
        right_top_x = prevTR

    right_bottom_y = height

    prevBR = right_bottom_x
    right_bottom_x = ((right_bottom_y - right_line[0]) / right_line[1])
    if right_bottom_x < int(width/2)-20 or right_bottom_x > int(width)+20:
        right_bottom_x = prevBR

    left_top = int(left_top_x), int(left_top_y)
    left_bottom = int(left_bottom_x), int(left_bottom_y)

    right_top = int(right_top_x), int(right_top_y)
    right_bottom = int(right_bottom_x), int(right_bottom_y)

    cv2.line(binarizat_copy, left_top, left_bottom, (200, 0, 0), 5)
    cv2.line(binarizat_copy, right_top, right_bottom, (100, 0, 0), 5)

    cv2.imshow('Lines B/W', binarizat_copy)

    # 11

    blank_final = np.zeros((height, width), dtype=np.uint8)
    cv2.line(blank_final, left_top, left_bottom, (255, 0, 0), 5)
    magic_matrix_final = cv2.getPerspectiveTransform(screen_bounds, trapezoid_bounds)
    normal_stretchedL = cv2.warpPerspective(blank_final, magic_matrix_final, (width, height))

    finalLeft = normal_stretchedL[:, 0:(int(width / 2))]

    array_left_final = np.argwhere(finalLeft == 255)

    blank_final2 = np.zeros((height, width), dtype=np.uint8)
    cv2.line(blank_final2, right_top, right_bottom, (255, 0, 0), 5)
    normal_stretchedR = cv2.warpPerspective(blank_final2, magic_matrix_final, (width, height))

    finalRight = normal_stretchedR[:, (int(width/2)):width]

    array_right_final = np.argwhere(finalRight == 255)
    array_right_final[:, 1] += int(width / 2)

    cv2.imshow('Left Line', normal_stretchedL)
    cv2.imshow('Right Line', normal_stretchedR)

    first_copy = frame.copy()

    for i in array_left_final:
        first_copy[i[0], i[1]] = (50, 50, 250)

    for i in array_right_final:
        first_copy[i[0], i[1]] = (50, 250, 50)

    cv2.imshow('Final', first_copy)

    stop = time.perf_counter()
    print(1//(stop-start))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
