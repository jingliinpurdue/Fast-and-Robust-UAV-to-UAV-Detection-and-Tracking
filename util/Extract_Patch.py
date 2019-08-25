
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




def Extract_Patch(frameidx, gray, Xt, weightedError, centers, H_back, ftparmes, ftparmes_ori,lk_params_track, radius, Xt_1, Xt_color, gt_mask, gt_img):
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
    locList = []
    locNext = []

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
            locList.append([xxC,yyC])
            locNext.append([x,y])
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
    return np.array(MV), np.array(Patches),np.array(Patches_errImg), np.array(gt_label), np.array(locList), np.array(locNext), HitNo, DetectNo, FANo


def DetectOnX(vis, gray, Xt, lk_param, H_back, detectedLocs, pred_y, detectedPatches,feature_params_Detect, r, mask):
    frame0, frame1 = np.uint8(gray), np.uint8(Xt)
    h, w =frame1.shape
    dtpt=[]
    for (m,n), predicted, oriPatch in zip(detectedLocs, pred_y, detectedPatches):
        if predicted!=1:
            continue
        oriPatch_gray = np.float32(cv2.cvtColor(oriPatch, cv2.COLOR_RGB2GRAY))

        pImg = cv2.goodFeaturesToTrack(np.uint8(oriPatch_gray), **feature_params_Detect)
        if pImg is None:
            print('No Points Detected')
            continue
        else:
            #pt_cornerness = corner[np.int(pImg[:,:,1]),np.int(pImg[:,:,0])]
            #print('lll')
            pImg[:,:,0]+=m-2*r
            pImg[:,:,1]+=n-2*r
        
            pPre_H = cv2.perspectiveTransform(pImg, np.linalg.inv(H_back))
            pPre_of, st2, err = cv2.calcOpticalFlowPyrLK(frame0, frame1, pImg, pPre_H, **lk_param)        
       
            for (x,y), (xpre_h,ypre_h), (xpre_of, ypre_of),s1 in zip(pImg.reshape(-1,2), pPre_H.reshape(-1,2), pPre_of.reshape(-1,2), st2):
                if s1==0:
                    cv2.rectangle(vis,(np.int16(x-1),np.int16(y-1)),(np.int16(x+1),np.int16(y+1)), (255,0,255),1)  
                    continue
                dx = xpre_of-xpre_h
                dy = ypre_of-ypre_h
                mag = np.sqrt(dx*dx+dy*dy)
                dtpt.append([xpre_of,ypre_of])
                
                #if mag<1:
                    #cv2.rectangle(vis,(np.int16(x-1),np.int16(y-1)),(np.int16(x+1),np.int16(y+1)), (0,255,0),1)
                    #continue
                #if mag>10:
                    #cv2.rectangle(vis,(np.int16(x-1),np.int16(y-1)),(np.int16(x+1),np.int16(y+1)), (255,0,0),1)
                    #continue
                cv2.rectangle(vis,(np.int16(x-1),np.int16(y-1)),(np.int16(x+1),np.int16(y+1)), (0,0,255),1)    
                
                

    return vis,np.array(dtpt)

def visPosPatch(dt_lable, gt_lables, detectedLocs, oriImage, radius,):
    R = 2*radius
    h,w,c = oriImage.shape
    for dt,gt,(x,y) in zip(dt_lable,gt_lables, detectedLocs):
        xxC,yyC = boundary(x,y,radius, w,h)
        if dt!=1:
            #cv2.rectangle(oriImage,(np.int16(xxC-R),np.int16(yyC-R)),(np.int16(xxC+R),np.int16(yyC+R)), (255,255,0),1)
            continue
        
        if gt==1:
            cv2.rectangle(oriImage,(np.int16(xxC-R),np.int16(yyC-R)),(np.int16(xxC+R),np.int16(yyC+R)), (0,255,0),1)
        else:
            cv2.rectangle(oriImage,(np.int16(xxC-R),np.int16(yyC-R)),(np.int16(xxC+R),np.int16(yyC+R)), (0,0,255),1)
        
    return oriImage

def writePatches(videoName, patch_savePath, oriImage, r, frameid, dt_lable,detectedLocs):
    h, w, c =oriImage.shape
    ftcount = 0
    for dt, (m,n) in zip(dt_lable, detectedLocs):
        if dt!=1:
            continue
        m,n = boundary(m,n,2*r, w,h)
        cv2.imwrite(patch_savePath+videoName+'_'+str(frameid)+'_'+str(ftcount)+'.png', oriImage[n-2*r:n+2*r,m-2*r:m+2*r,:])
        ftcount+=1
def updatetr(p,d, d1, st1, st0):
    updatept=[]
    good=d>0.4
    good_bi = d1<1.5
    for (x,y), good_flag, good_flag_bi,s1, s0 in zip(p.reshape(-1,2),good,good_bi,st1, st0):
        #print(s1,s0,s1==0)
        if s1==0 :
            #print('lol')
            continue
        if not good_flag_bi:
            continue
        updatept.append([x,y])
    return updatept

def updatetr_deep(p,scores):
    updatept=[]
    for (x,y), sc in zip(p.reshape(-1,2),scores):
        #print(s1,s0,s1==0)
        if sc<0.16 :
            #print('lol')
            continue
        updatept.append([x,y])
    return updatept

def updatetr_stat(p,scores):
    updatept=[]
    for (x,y), sc in zip(p.reshape(-1,2),scores):
        #print(s1,s0,s1==0)
        if sc==0:
            continue
        updatept.append([x,y])
    return updatept

def updatetr_nothing(p):
    updatept=[]
    for (x,y) in p.reshape(-1,2):
        #print(s1,s0,s1==0)
        updatept.append([x,y])
    return updatept


def updatetr_prune(p,d, d1, st1, st0):
    updatept=[]
    good=d>0.5
    good_bi = d1<1.5
    for (x,y), good_flag, good_flag_bi,s1, s0 in zip(p.reshape(-1,2),good,good_bi,st1, st0):
        #print(s1,s0,s1==0)
        if s1==0 :
            #print('lol')
            continue
        if not good_flag_bi:
            continue
        #print(good_flag)
        if not good_flag:
            continue
        updatept.append([x,y])
    return updatept

def updatetr_combine(p, score, st, d):
    updatept=[]
    for (x,y), sc, s1,dist  in zip(p.reshape(-1,2), score, st,d):
        if dist>2:
            continue
        if s1==0:
            continue
        if sc!=1:
            continue
        updatept.append([x,y])
    return updatept
def visPtV1(oriImage, p0, st1, d1):
    for (x,y), st, d in zip(p0.reshape(-1, 2), st1, d1):
        if d>2:
            continue
        if st==0:
            continue
        #draw_str(oriImage, np.int16(x-1), np.int16(y-1), 'ID: %d' % (pt.classID))
        cv2.circle(oriImage, (x, y), 4, (255, 0, 0), 1)
    return oriImage
def visPt(oriImage, p0,gt_mask):    
    dt = 0
    fa=0
    hitindex=[]
    h,w = gt_mask.shape
    for x,y in p0.reshape(-1, 2):
        cv2.circle(oriImage, (x, y), 2, (0, 255, 0), -1) 
        dt+=1
        x,y =boundary(x,y,1,w,h)
        if gt_mask[np.int(y), np.int(x)]>0:
            hitindex.append(gt_mask[np.int(y), np.int(x)])
        else:
            fa+=1
    hit = np.unique(hitindex).shape[0]
    return oriImage, dt, fa, hit