
import cv2

import numpy as np
import math
import cv2
import random
import time
import sys
from util.kalman2d import Kalman2D
import operator

from util.UAV_subfunctions import *


def generatePatches_MV(frameidx, gray, Xt, weightedError, centers, H_back, ftparmes, ftparmes_ori,lk_params_track, radius, Xt_1, Xt_color, gt_mask, gt_img):
    h, w = gray.shape
    DetectNo = 0
    HitNo = 0
    FANo = 0
    pall = []

    weightedError *= 255.0/weightedError.max()
    featuresforBackgroundSubtractedImage = cv2.cvtColor(np.uint8(weightedError),cv2.COLOR_GRAY2RGB)
    ft = cv2.goodFeaturesToTrack(np.uint8(weightedError),  **ftparmes)
    pall.append(ft)

    posIndex=[]
    Patches = []
    Patches_errImg = []
    MV = []
    gt_label=[]

    posdetect=0
    for p in pall:
        if p is None:
            print('noPoint1')
            continue
        if len(p.shape)!=3:
            print('noPoint2')
            continue
        frame0, frame1 = np.uint8(gray), np.uint8(Xt)
        # find corresponding points pCur  in current frame for pPre
        pCur, st, err = cv2.calcOpticalFlowPyrLK(frame0, frame1, p, None, **lk_params_track)
        # track back pCur in previous frame
        p0Pre, st, err = cv2.calcOpticalFlowPyrLK(frame1, frame0, pCur, None, **lk_params_track)
        # compute distance between the location of original feature points and tracked feature points
        d = abs(p-p0Pre).reshape(-1, 2).max(-1)
        #print(d.shape)
        # keep features have good matching ones
        good_frame = d < 1
        # compute corresponding location based on perpective transform
        converted = cv2.perspectiveTransform(p, np.linalg.inv(H_back))
        for (x, y),(xx,yy), (xhat,yhat),dist in zip( pCur.reshape(-1, 2),p.reshape(-1, 2),converted.reshape(-1,2), d):
            DetectNo +=1     
            mv_opx = x-xx
            mv_opy = y-yy
            mv_px = xhat-xx
            mv_py = yhat-yy
            dt_x = x-xhat
            dt_y = y-yhat
            mag_op = np.sqrt(mv_opx*mv_opx+mv_opy*mv_opy)
            mag_p = np.sqrt(mv_px*mv_px+mv_py*mv_py)
            theta_op = np.math.atan2(mv_opy,mv_opx)
            theta_p = np.math.atan2(mv_py,mv_px)
            mag = np.sqrt(dt_x*dt_x+dt_y*dt_y)
            theta = np.math.atan2(dt_y,dt_x)
            magd = abs(mag_op-mag_p)
            thetad = abs(theta_op-theta_p)
            if thetad>np.pi:
                thetad = 2*np.pi-thetad
            datapatch = [mv_opx,mv_opy,mv_px,mv_py,dt_x,dt_y,mag_op,mag_p,theta_op,theta_p, mag, theta,magd, thetad, dist]
            xxC,yyC = boundary(xx,yy,radius, w,h)
            
            MV.append(datapatch)
            Patches.append(Xt_1[yyC-2*radius:yyC+2*radius,xxC-2*radius:xxC+2*radius,:])
            Patches_errImg.append(weightedError[yyC-2*radius:yyC+2*radius,xxC-2*radius:xxC+2*radius])
            if gt_mask[np.int16(yy),np.int16(xx)]>0:
                #print('bingo')
                posdetect+=1
                gt_label.append(1)
                posIndex.append(np.trim_zeros(np.unique(gt_mask[yyC-2*radius:yyC+2*radius,xxC-2*radius:xxC+2*radius])))                                                        
            else:
                gt_label.append(0)
                FANo+=1             

    if len(posIndex)>0:       
        detects = np.unique(np.hstack(posIndex))
        HitNo += detects.shape[0]
    return np.array(MV), np.array(Patches),np.array(Patches_errImg), np.array(gt_label), HitNo, DetectNo, FANo

def generatePatches_MV_track(trackedpt, gray, Xt, H_back, lk_params_track_ori, radius,w, h, Xt_1, gt_ft_maske):
    p0 = np.float32([tr for tr in trackedpt]).reshape(-1, 1, 2)
    pPers = cv2.perspectiveTransform(p0, np.linalg.inv(H_back))
    p1, st1, err = cv2.calcOpticalFlowPyrLK(np.uint8(gray), np.uint8(Xt), p0, pPers.copy(), **lk_params_track_ori) 
    pdummy = cv2.perspectiveTransform(p1, H_back)
    p0r, st0, err = cv2.calcOpticalFlowPyrLK(np.uint8(Xt), np.uint8(gray), p1, pdummy, **lk_params_track_ori)
    d1 = abs(p0-p0r).reshape(-1, 2).max(-1)
    
    d = abs(pPers-p1).reshape(-1, 2).max(-1)               
    dt_x = abs(p0-p0r).reshape(-1, 2)[:,0].reshape(-1,1)
    dt_y = abs(p0-p0r).reshape(-1, 2)[:,1].reshape(-1,1)
                
    mag_dt = np.sqrt(dt_x*dt_x+dt_y*dt_y).reshape(-1,1)
    theta_dt = np.arctan2(dt_y,dt_x).reshape(-1,1)
                
    dp_x = (pPers.reshape(-1,2)-p0.reshape(-1,2))[:,0].reshape(-1,1)
    dp_y = (pPers.reshape(-1,2)-p0.reshape(-1,2))[:,1].reshape(-1,1)
    mag_p = np.sqrt(dp_x*dp_x+dp_y*dp_y).reshape(-1,1)
    theta_p = np.arctan2(dp_y,dp_x).reshape(-1,1)
                
    do_x = (p1.reshape(-1,2)-p0.reshape(-1,2))[:,0].reshape(-1,1)
    do_y = (p1.reshape(-1,2)-p0.reshape(-1,2))[:,1].reshape(-1,1)
    mag_o = np.sqrt(do_x*do_x+do_y*do_y).reshape(-1,1)
    theta_o = np.arctan2(do_y,do_x).reshape(-1,1)                
                
    mag_d = abs(mag_o-mag_p).reshape(-1,1)
    theta_d = abs(theta_o-theta_p)
    theta_d[theta_d>np.pi] = 2*np.pi-theta_d[theta_d>np.pi]
    theta_d = theta_d.reshape(-1,1)
                
    ft_mv = np.hstack([dt_x, dt_y,mag_dt, theta_dt, mag_d, theta_d, d.reshape(-1,1)])
                #print(ft_mv)
    #score_mv = mvmodel.predict(ft_mv, batch_size = 1000000)
    p0_int = np.int32([tr for tr in trackedpt]).reshape(-1,  2)
    p0_x = p0_int[:,0]
    p0_y = p0_int[:,1]
    p0_x[p0_x<0]=0
    p0_y[p0_y<0]=0
    p0_x[p0_x>=w] = w-1
    p0_y[p0_y>=h] = h-1
    gt_labels = gt_ft_maske[p0_y,p0_x]
    Patches = []    
    
    for (xx,yy) in p0.reshape(-1, 2):
        xxC,yyC = boundary(xx,yy,radius, w,h)
        Patches.append(Xt_1[yyC-2*radius:yyC+2*radius,xxC-2*radius:xxC+2*radius,:]/255.0)
    return d1, p1, p0, st1, np.array(ft_mv), np.array(Patches), np.array(gt_labels)

def wirtefile(file, appPa, mvPa, gt):
    for p_a,p_m,label in zip(appPa, mvPa, gt):
        file.write(str(p_a)+ ' ,'+ str(p_m)+' ,'+str(label))
        file.write('\n')
