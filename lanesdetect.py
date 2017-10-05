#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 14:21:37 2017

@author: guilherme
"""

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

from Line import Line

left=Line()
right=Line()
lane=Line()
lane.detected = False 

i=0


def show_2_images(img_1, title_1, img_2, title_2, cmap_value=None, convert_1=False, convert_2=False):
    
    if convert_1 == True:
        img_1=cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
    if convert_2 == True:
        img_2=cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
        
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img_1,cmap=cmap_value)
    ax1.set_title(title_1, fontsize=30)
    ax2.imshow(img_2,cmap=cmap_value)
    ax2.set_title(title_2, fontsize=30)
    
def show_3_images(img_1, title_1, img_2, title_2, img_3, title_3, cmap_value=None, convert=False):
    
    if convert == True:
        img_1=cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
        img_2=cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
        img_3=cv2.cvtColor(img_3, cv2.COLOR_BGR2RGB)
    
    
    f, (ax1, ax2, ax3) = plt.subplots(1, 3,  figsize=(12, 9))
    f.tight_layout()
    ax1.imshow(img_1,cmap=cmap_value)
    ax1.set_title(title_1, fontsize=50)
    ax2.imshow(img_2,cmap=cmap_value)
    ax2.set_title(title_2, fontsize=50)
    ax3.imshow(img_3,cmap=cmap_value)
    ax3.set_title(title_3, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    
    

def calibrate_camera(path_to_calibrate_images, show=False , save=False):
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob(path_to_calibrate_images)


    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

   
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            img_teste = cv2.imread('test_images/test1.jpg')
            img_size = (img_teste.shape[1], img_teste.shape[0])
            ret_, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
            
            if show == True:
                cv2.drawChessboardCorners(img, (9,6), corners, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)
                
            if save == True:
                dist_pickle = {}
                dist_pickle["mtx"] = mtx
                dist_pickle["dist"] = dist
                pickle.dump( dist_pickle, open( "calibration.p", "wb" ) )
            
    return mtx, dist

def undist(img,mtx,dist):
    dst = cv2.undistort(img, mtx, dist, None, mtx) 
    return dst

def bird_view (img):
    
    img_size=(img.shape[1],img.shape[0])
    src_pts = np.float32([[300,720],[590,450],[685,450],[1000,720]])
    dst_pts = np.float32([[300,720],[300,0],[1000,0],[1000,720]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    bird_view_image = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    
    return bird_view_image, Minv

def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

   
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    
    return binary_output

def RGB_Split(img):
    R=img[:,:,0]
    G=img[:,:,1]
    B=img[:,:,2]
    
    return R,G,B

def HLS_Split(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    H=hls[:,:,0]
    L=hls[:,:,1]
    S=hls[:,:,2]
    
    return H,L,S

def thresh_color_channel(ch, thresh_min=0, thresh_max=255):
    binary = np.zeros_like(ch)
    binary[(ch >= thresh_min) & (ch <= thresh_max)] = 1
    
    return binary








mtx, dist = calibrate_camera("camera_cal/*.jpg", show=False, save=False)
##############################################################################
def lanes_curvature (ploty, leftx, rightx):
    
    y_eval = np.max(ploty)
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left.radius_of_curvature = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right.radius_of_curvature = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    lane.pix_to_meters =  '{0:.3g}'.format(((lane.carpos-lane.midpos)*3.7)/(right.allx[719] - left.allx[719]))
    
    return 

##############################################################################
def find_lanes(binary_warped, show = False):
     
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and int visuintalize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    lane.midpos = midpoint #Center point of camera
    lane.carpos = (leftx_base + rightx_base)/2
    lane.width = rightx_base - leftx_base #Detected lane width 
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    left.current_fit = left_fit  ### polynomial coefficients for the most recent fit
    right.current_fit = right_fit #### polynomial coefficients for the most recent fit
    
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    lane.ally = ploty ###All Y points 
    left.allx = left_fitx ####All left X points
    right.allx = right_fitx ### All right X points 
    
    
    ##Sanity Check
    
    if right.allx[719] - left.allx[719] < 0.98*lane.width or right.allx[719] - left.allx[719] > 1.02*lane.width:
        lane.detected = False
               
    else:
        lane.detected = True
    
    
     
    
    #Show Output
    if show == True:
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        
       
    
    
    
    
    
    

##############################################################################
def find_next_lane(binary_warped, left_fit, right_fit, show = False):
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    left.current_fit = left_fit  ### polynomial coefficients for the most recent fit
    right.current_fit = right_fit #### polynomial coefficients for the most recent fit
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    lane.ally = ploty ###All Y points 
    left.allx = left_fitx ####All left X points
    right.allx = right_fitx ### All right X points 
    
    ##Sanity Check
    
    if right.allx[719] - left.allx[719] < 0.98*lane.width or right.allx[719] - left.allx[719] > 1.02*lane.width:
        lane.detected = False
               
    else:
        lane.detected = True
    
    
    if show == True:
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        
##############################################################################
        
def draw_lines(undist,warped, Minv, left_fitx, right_fitx, ploty, show = False):

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    #result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    cv2.putText(result,"Left Rad: " + str(int(left.radius_of_curvature)) + "m", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 5)
    cv2.putText(result,"Right Rad: " + str(int(right.radius_of_curvature)) + "m", (100,200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 5)
    cv2.putText(result,"Center Error: " + str(lane.pix_to_meters) + "m", (100,300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 5)
    cv2.putText(result,"Lane Detect: " + str(lane.detected) , (100,400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 5)

    if show == True:
        plt.imshow(result)
    
    return result

##############################################################################
    
def color_gradient (img):  
    
    img = undist(img, mtx, dist)
    _,_,s_channel = HLS_Split(img)
    r_channel,_,_ = RGB_Split(img)
    s_binary = thresh_color_channel(s_channel, thresh_min=100, thresh_max=255)
    r_binary = thresh_color_channel(s_channel, thresh_min=150, thresh_max=255)
      
    
    sobel_binary = abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=150)
    mag_binary = mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255))  
    
    combined_binary = (sobel_binary | s_binary) | r_binary
    
    return img,combined_binary

##############################################################################
    
def advanced_lane_lines(image):
    global i
    i=i+1
    
    
    undistorced,result = color_gradient(image)

    result, Minv  = bird_view(result)
    
    if lane.detected == False:
        find_lanes(result, show = False)
    
    else:
        find_next_lane(result, left.current_fit, right.current_fit, show = False)
        
    left.recent_xfitted.append(left.allx)
    right.recent_xfitted.append(right.allx)
    left.bestx = np.average(left.recent_xfitted[-40:],axis=0)
    right.bestx = np.average(right.recent_xfitted[-40:],axis=0)

    lanes_curvature (lane.ally, left.allx, right.allx)
    result = draw_lines(undistorced,result, Minv, left.bestx , right.bestx , lane.ally,show=True)
    
    if i > 100:
        left.recent_xfitted = []
        right.recent_xfitted = []
        i=0
          
    
    return result



    
    
    
          
