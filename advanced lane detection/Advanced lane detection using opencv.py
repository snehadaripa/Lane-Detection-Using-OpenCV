import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time


dist_pickle = pickle.load(open("calibration_wide/wide_dist_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]


def undistort_image(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)



def abs_sobel_thresh(img,  orient = 'x', thresh_min=0, thresh_max=255):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if orient=='x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    abs_sobel = np.absolute(sobel)

    scaled_sobel = np.uint8(255*abs_sobel / np.max(abs_sobel))
    abs_binary = np.zeros_like(scaled_sobel)
    abs_binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 255
    return abs_binary

def mag_thresh(img,  sobel_kernel=3, thresh_min=0, thresh_max=255):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    abs_sobel = ((abs_sobelx)**2 + (abs_sobely)**2)**0.5

    scaled_sobel = np.uint8(255*abs_sobel / np.max(abs_sobel))
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 255
    return mag_binary


def dir_thresh(img,  sobel_kernel=3, thresh_min=0, thresh_max=np.pi/2):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    arctanSobel=np.arctan2(abs_sobely, abs_sobelx)


    dir_binary = np.zeros_like(arctanSobel)
    dir_binary[(arctanSobel >= thresh_min) & (arctanSobel <= thresh_max)] = 255
    return dir_binary

def combined_thresh(img):
    ksize = 3  # Choose a larger odd number to smooth gradient measurements
    gradx = abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=100)
    grady = abs_sobel_thresh(img, orient='y', thresh_min=20, thresh_max=100)
    mag_binary = mag_thresh(img, sobel_kernel=ksize, thresh_min=20, thresh_max=100)
    dir_binary = dir_thresh(img, sobel_kernel=ksize, thresh_min=0.7, thresh_max=1.3)
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 255) & (grady == 255)) | ((mag_binary == 255) & (dir_binary == 255))] = 255
    return combined


def hls_select(img, thresh=(0,255)):
    hls=cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel =hls[:, : , 1]
    l_Binary = np.zeros_like(l_channel)
    l_Binary[(l_channel >= thresh[0]) & (l_channel <= thresh[1])] = 255
    return l_Binary

def pipeline(img):

    combined = combined_thresh(img)
    l_channel = hls_select(img, (150,255))
    combined_binary = np.zeros_like(combined)
    combined_binary[(l_channel == 255) | (combined == 255)] = 255
    return combined_binary


def warp(img):
    h, w = img.shape[:2]

    src = np.float32([[w, h - 10],  # br
                      [0, h-10],    # bl
                      [546, 460],  # tl
                      [732, 460]])  # tr

    dst = np.float32([[w, h],  # br
                      [0, h],  # bl
                      [0, 0],  # tl
                      [w, 0]])  # tr

    M = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

    return warped

def unwarp(img):
    h, w = img.shape[:2]
    src = np.float32([[w, h - 10],  # br
                      [0, h - 10],  # bl
                      [546, 460],  # tl
                      [732, 460]])  # tr

    dst = np.float32([[w, h],  # br
                      [0, h],  # bl
                      [0, 0],  # tl
                      [w, 0]])  # tr


    Minv = cv2.getPerspectiveTransform(dst, src)

    unwarped = cv2.warpPerspective(img, Minv, (w, h), flags=cv2.INTER_LINEAR)

    return unwarped


def find_lane_pixels(img):
    binary = pipeline(img)
    warped = warp(binary)

    histogram = np.sum(warped[int(warped.shape[0] / 2):, :], axis=0)
    out_img = np.dstack((warped, warped, warped))

    midpoint= int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows= 9
    margin=100
    minpix=50

    window_height=int(warped.shape[0]//nwindows)
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range (nwindows):
        win_y_low = warped.shape[0] - (window + 1) * window_height
        win_y_high = warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_poly(img_shape, leftx, lefty, rightx, righty):
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])

    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    return left_fitx, right_fitx, ploty


def search_around_poly(binary_warped, leftx, lefty, rightx, righty, undist):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search

    margin = 100

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                     ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (255, 0, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 0, 255))

    # Plot the polynomial lines onto the image
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = unwarp(color_warp)
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    plt.imshow(result)


    return result


lane_video = cv2.VideoCapture('lane.mp4')

if not lane_video.isOpened():
    print("Error in opening video source file.")

prev_frame_time = 0

new_frame_time = 0

while lane_video.isOpened():
    ret, image_of_car=lane_video.read()

    if ret:


        img = np.copy(image_of_car)
        ysize=img.shape[0]
        xsize=img.shape[1]
        undist = undistort_image(img, mtx, dist)
        combined_warped=pipeline(img)
        binary_warped=warp(combined_warped)

        leftx, lefty, rightx, righty, out_img = find_lane_pixels(img)

        result=search_around_poly(binary_warped, leftx, lefty, rightx, righty, undist)
        font = cv2.FONT_HERSHEY_SIMPLEX

        new_frame_time = time.time()


        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        fps = int(fps)

        fps = str(fps)


        cv2.putText(result, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow('final', result)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
