
import cv2

import numpy as np
import math
import cv2
import random
import time
import sys
from util.kalman2d import Kalman2D
import operator
from numpy import zeros, newaxis
from util.UAV_subfunctions import *


def dottrack_detect(oriDot, p1, p0, score, st1, d1, d2, Patches):
    N=4
    newDot=[]
    #print(len(oriDot), p1.shape)
    for Dot, (x, y), (xx,yy), dt, st, d, dist in zip(oriDot, p1.reshape(-1,2), p0.reshape(-1,2), score, st1,d1, d2):
        #print(dt, dt==1)
        if dt!=1:
            continue
        if st==0:
            continue
        if d>1:
            continue
        if dist>15:
            continue
        #Patch1=[pt for pt in Patches if pt.classID==Dot.classID]
        dx = xx-x
        dy = yy-y
        #Patch = Patch1[0]
        #if dx>= Patch.meanMotionDifference[0]-N*Patch.stdMotionDifference[0] and dx<=Patch.meanMotionDifference[0]+N*Patch.stdMotionDifference[0] and dy<=Patch.meanMotionDifference[1]+N*Patch.stdMotionDifference[1] and dy>=Patch.meanMotionDifference[1]-N*Patch.stdMotionDifference[1] or (abs(dx)>0.5 or abs(dy)>0.5):
        Dot.location = [x,y]
        Dot.motionDifference=[dx, dy]
            #print('classID:', Dot.classID)
        newDot.append(Dot)
    return newDot



def prunddt(Dotft, PatchPt):
    newDot=[]
    status=[]
    N=50.0
    for Dot in Dotft:
        
        Patches = [tr for tr in PatchPt if tr.classID==Dot.classID]
        #print(len(Patches))
        if len(Patches)==0:
            status.append(False)
            continue
        patch = Patches[0]

        #dx = Dot.motionDifference[0]
        #dy = Dot.motionDifference[1]
        #pdx = patch.meanMotionDifference[0]
        #pdy = patch.meanMotionDifference[1]
        #[ex, ey]= patch.Kalman1.getEstimate()
        ddx= Dot.motionDifference[0]-patch.meanMotionDifference[0]#x
        ddy=Dot.motionDifference[1]-patch.meanMotionDifference[1]#ey
        stdx=patch.stdMotionDifference[0]
        stdy = patch.stdMotionDifference[1]
        #print(ddx,ddy, stdx, stdy,Dot.motionDifference[0],patch.meanMotionDifference[0])
        if abs(ddx)> N*stdx or abs(ddy) > N*stdy:
            status.append(True)
            continue
        status.append(True)
        newDot.append(Dot)
        
    return newDot, status
        #if Dot.motionDifference[0]
def dotupdate(oriDot, PatchPt):
    r=10
    N=3
    newDot=[]
    status = []
    for Dot in oriDot:
        [mcur, ncur] = Dot.location
        [dx, dy] = Dot.motionDifference
        #print(len(PatchPt))
        #for tr in PatchPt:
            #print(tr.location)
        #Patches = [tr for tr in PatchPt if tr.location[0]>=mcur-N*r and tr.location[0]<=mcur+N*r and tr.location[1]>=ncur-N*r and tr.location[1]<=ncur+N*r] 
        
        Patches = [tr for tr in PatchPt if mcur>=tr.location[0]-N*r and mcur<=tr.location[0]+N*r and ncur>=tr.location[1]-N*r and ncur<= tr.location[1]+N*r]
        dist = [abs(dx -tr.meanMotionDifference[0])+abs(dy-tr.meanMotionDifference[1]) for tr in Patches]
                    #print('dist:', dist)
        if len(Patches)>0:
            #print('numberPatches:', len(Patches))
            dotID = Patches[np.argmin(dist)].classID
            Dot.classID = dotID
            newDot.append(Dot)
            status.append(True)
        else:
            status.append(False)
            #oriDot.remove(Dot)#print('bug')
        
    return newDot, status





def patch_KalmanTracking(Dotft, Patchft, H, w,h ):
    updatePatch = []
    for Pattr in Patchft:
        dots = [tr for tr in Dotft if tr.classID==Pattr.classID]
        locations = np.array([tr.location for tr in Dotft if tr.classID==Pattr.classID]).reshape(-1,2)
        meandf = np.array([tr.motionDifference for tr in Dotft if tr.classID==Pattr.classID]).reshape(-1,2)
        Pattr.history+=1
        [ex, ey]= Pattr.Kalman1.getEstimate()
        
        loc= Pattr.location
        #print(loc)
        #print(ex, ey)
        #print(np.float32(cv2.perspectiveTransform(np.float32(loc).reshape(-1, 1, 2), np.linalg.inv(H))).reshape(-1,2)[0])
        [perx, pery] = np.float32(cv2.perspectiveTransform(np.float32(loc).reshape(-1, 1, 2), np.linalg.inv(H))).reshape(-1,2)[0]
               
        #print('ptNo:', locations.shape)
        if len(locations)==0:
            Pattr.totalinvisibleCount+=1
            Pattr.invisibleCount+=1
            
            if np.float(Pattr.totalinvisibleCount)/np.float(Pattr.history)>0.8:
                continue
            #if Pattr.history<5:
                #Patchft.remove(Pattr)
                #continue
            if Pattr.invisibleCount>10:
                #Patchft.remove(Pattr)
                continue
            if loc[0]<20 or loc[0]>w-20 or loc[1]>h-20 or loc[1]<20:
                continue
            Pattr.trackLocation = [perx-ex, pery-ey]
            Pattr.location = [perx-ex, pery-ey]
            Pattr.Kalman1.update(ex, ey)
            updatePatch.append(Pattr)
        else:
            #print(perx-ex, pery-ey, locations.mean(0))
            #print(Pattr.classID, locations)
            Pattr.invisibleCount=0
            Pattr.location = locations.mean(0)
            Pattr.meanMotionDifference= meandf.mean(0)
            Pattr.stdMotionDifference= meandf.std(0)
            [est1,est2] = meandf.mean(0)
            Pattr.Kalman1.update((3*est1+1*ex)/4.0, (3*est2+1*ey)/4.0)#(est1+ex)/2.0, (est2+ey)/2.0)#est1,est2)#(est1+ex)/2.0, (est2+ey)/2.0)
            Pattr.trackLocation=[perx-ex, pery-ey]
            updatePatch.append(Pattr)
    return updatePatch
    



def patchtrack(Dotft, Patchft):
    updatePatch = []
    for Pattr in Patchft:
        locations = np.array([tr.location for tr in Dotft if tr.classID==Pattr.classID]).reshape(-1,2)
        meandf = np.array([tr.motionDifference for tr in Dotft if tr.classID==Pattr.classID]).reshape(-1,2)
        Pattr.history+=1
        #print('ptNo:', locations.shape)
        if len(locations)==0:
            Pattr.invisibleCount+=1
            continue
        else:
            Pattr.history+=1
            Pattr.invisibleCount=0
            Pattr.box.append(Pattr.location)
            Pattr.location = locations.mean(0)
            Pattr.meanMotionDifference= meandf.mean(0)
            Pattr.stdMotionDifference= meandf.std(0)
            updatePatch.append(Pattr)
            Pattr.Kalman1.update(meandf.mean(0))
    return updatePatch


def DetectOnX_V2(N, maxID, vis, gray, Xt, lk_param, H_back, detectedLocs, CurLocs, pred_y, detectedPatches,feature_params_Detect, r, Dotpt, PatchPt):
    frame0, frame1 = np.uint8(gray), np.uint8(Xt)
    h, w =frame1.shape
    M = 5
    for (m,n), (mcur, ncur), predicted, oriPatch in zip(detectedLocs,CurLocs, pred_y, detectedPatches):
        if predicted!=1:
            continue            
        oriPatch_gray = np.float32(cv2.cvtColor(oriPatch, cv2.COLOR_RGB2GRAY))
        pImg = cv2.goodFeaturesToTrack(np.uint8(oriPatch_gray), **feature_params_Detect)       
        
        if pImg is None:
            print('No Points Detected')
            continue
        else:
            pImg[:,:,0]+=m-2*r
            pImg[:,:,1]+=n-2*r
        
            pPre_H = cv2.perspectiveTransform(pImg, np.linalg.inv(H_back))
            pPre_of, st2, err = cv2.calcOpticalFlowPyrLK(frame0, frame1, pImg, pPre_H.copy(), **lk_param)      
     
            #print(err)
            
            #IDs = np.int32([tr.classID for tr in Dotpt if tr.location[0]>=mcur-3*r and tr.location[0]<=mcur+3*r and tr.location[1]>=ncur-3*r and tr.location[1]<=ncur+3*r])
            #IDs = np.unique(IDs)
            #print('allpointsIDs', IDs)
            #print(N)
            Patches = [tr for tr in PatchPt if mcur>=tr.location[0]-M*r and mcur<=tr.location[0]+M*r and ncur>=tr.location[1]-M*r and ncur<= tr.location[1]+M*r]           
            #print('numberof Patches', len(Patches))
            pPre_H = pPre_H[st2>0]
            pPre_of = pPre_of[st2>0]
            pImg = pImg[st2>0]
            
            if pImg.shape[0]==0:
                continue
            if len(Patches)==0:
                maxID+=1
                [locx, locy] = pPre_of.reshape(-1,2).mean(0)
                [dtx,dty] = (pPre_H-pPre_of).reshape(-1,2).mean(0)
                [stdx,stdy] = (pPre_H-pPre_of).reshape(-1,2).std(0)              

                
                #print('center', locx, locy)
                #print('ptloc', reallocx, reallocy)
                newPatch = FeaturePatchV1(maxID, locx, locy,  dtx, dty, stdx, stdy,1)
                newPatch.Kalman1 = Kalman2D()
                newPatch.Kalman1.update(dtx,dty)
                PatchPt.append(newPatch)
                for (x,y), (xpre_h,ypre_h), (xpre_of, ypre_of) in zip(pImg.reshape(-1,2), pPre_H.reshape(-1,2), pPre_of.reshape(-1,2)):
                    if max(abs(xpre_h-xpre_of), abs(ypre_h-ypre_of))>10:
                        continue
                    newpt = FeatrueDotsV1(xpre_of, ypre_of, xpre_h-xpre_of, ypre_h-ypre_of, maxID)
                    Dotpt.append(newpt)
                    cv2.rectangle(vis,(np.int16(x-1),np.int16(y-1)),(np.int16(x+1),np.int16(y+1)), (0,255,255),1)
            else:
                for (x,y), (xpre_h,ypre_h), (xpre_of, ypre_of) in zip(pImg.reshape(-1,2), pPre_H.reshape(-1,2), pPre_of.reshape(-1,2)):
                    [dx, dy] = [xpre_h-xpre_of, ypre_h-ypre_of] 
                    dist = [abs(dx -tr.meanMotionDifference[0])+abs(dy-tr.meanMotionDifference[1]) for tr in Patches]
                    #print('dist:', dist)
                    dotID = Patches[np.argmin(dist)].classID
                    #print('dotIDs:', dotID)
                    newpt = FeatrueDotsV1(xpre_of, ypre_of, xpre_h-xpre_of, ypre_h-ypre_of, dotID)
                    Dotpt.append(newpt)
                    cv2.rectangle(vis,(np.int16(x-1),np.int16(y-1)),(np.int16(x+1),np.int16(y+1)), (255,0,255),1)   

    return vis, Dotpt, PatchPt, maxID





def DetectOnX_Vis(N, maxID, vis, gray, Xt, lk_param, H_back, detectedLocs, CurLocs, pred_y, detectedPatches,feature_params_Detect, r, Dotpt, PatchPt):
    frame0, frame1 = np.uint8(gray), np.uint8(Xt)
    h, w =frame1.shape
    M = 5
    for (m,n), (mcur, ncur), predicted, oriPatch in zip(detectedLocs,CurLocs, pred_y, detectedPatches):
        if predicted!=1:
            continue            
        oriPatch_gray = np.float32(cv2.cvtColor(oriPatch, cv2.COLOR_RGB2GRAY))
        pImg = cv2.goodFeaturesToTrack(np.uint8(oriPatch_gray), **feature_params_Detect)       
        
        if pImg is None:
            print('No Points Detected')
            continue
        else:
            pImg[:,:,0]+=m-2*r
            pImg[:,:,1]+=n-2*r
        
            pPre_H = cv2.perspectiveTransform(pImg, np.linalg.inv(H_back))
            pPre_of, st2, err = cv2.calcOpticalFlowPyrLK(frame0, frame1, pImg, pPre_H.copy(), **lk_param)      
     
            #print(err)
            
            #IDs = np.int32([tr.classID for tr in Dotpt if tr.location[0]>=mcur-3*r and tr.location[0]<=mcur+3*r and tr.location[1]>=ncur-3*r and tr.location[1]<=ncur+3*r])
            #IDs = np.unique(IDs)
            #print('allpointsIDs', IDs)
            #print(N)
            Patches = [tr for tr in PatchPt if mcur>=tr.location[0]-M*r and mcur<=tr.location[0]+M*r and ncur>=tr.location[1]-M*r and ncur<= tr.location[1]+M*r]           
            #print('numberof Patches', len(Patches))
            pPre_H = pPre_H[st2>0]
            pPre_of = pPre_of[st2>0]
            pImg = pImg[st2>0]
            
            if pImg.shape[0]==0:
                continue
            if len(Patches)==0:
                maxID+=1
                [locx, locy] = pPre_of.reshape(-1,2).mean(0)
                [dtx,dty] = (pPre_H-pPre_of).reshape(-1,2).mean(0)
                [stdx,stdy] = (pPre_H-pPre_of).reshape(-1,2).std(0)              

                
                #print('center', locx, locy)
                #print('ptloc', reallocx, reallocy)
                newPatch = FeaturePatchV1(maxID, locx, locy,  dtx, dty, stdx, stdy,1)
                newPatch.Kalman1 = Kalman2D()
                newPatch.Kalman1.update(dtx,dty)
                PatchPt.append(newPatch)
                for (x,y), (xpre_h,ypre_h), (xpre_of, ypre_of) in zip(pImg.reshape(-1,2), pPre_H.reshape(-1,2), pPre_of.reshape(-1,2)):
                    if max(abs(xpre_h-xpre_of), abs(ypre_h-ypre_of))>10:
                        continue
                    newpt = FeatrueDotsV1(xpre_of, ypre_of, xpre_h-xpre_of, ypre_h-ypre_of, maxID)
                    Dotpt.append(newpt)
                    cv2.rectangle(vis,(np.int16(xpre_of-1),np.int16(ypre_of-1)),(np.int16(xpre_of+1),np.int16(ypre_of+1)), (0,255,255),1)
            else:
                for (x,y), (xpre_h,ypre_h), (xpre_of, ypre_of) in zip(pImg.reshape(-1,2), pPre_H.reshape(-1,2), pPre_of.reshape(-1,2)):
                    [dx, dy] = [xpre_h-xpre_of, ypre_h-ypre_of] 
                    dist = [abs(dx -tr.meanMotionDifference[0])+abs(dy-tr.meanMotionDifference[1]) for tr in Patches]
                    #print('dist:', dist)
                    dotID = Patches[np.argmin(dist)].classID
                    #print('dotIDs:', dotID)
                    newpt = FeatrueDotsV1(xpre_of, ypre_of, xpre_h-xpre_of, ypre_h-ypre_of, dotID)
                    Dotpt.append(newpt)
                    cv2.rectangle(vis,(np.int16(xpre_of-1),np.int16(ypre_of-1)),(np.int16(xpre_of+1),np.int16(ypre_of+1)), (255,0,255),1)   

    return vis, Dotpt, PatchPt, maxID






def DetectOnX_V1(maxID, vis, gray, Xt, lk_param, H_back, detectedLocs, CurLocs, pred_y, detectedPatches,feature_params_Detect, r, Dotpt, PatchPt):
    frame0, frame1 = np.uint8(gray), np.uint8(Xt)
    h, w =frame1.shape
    N = 4
    for (m,n), (mcur, ncur), predicted, oriPatch in zip(detectedLocs,CurLocs, pred_y, detectedPatches):
        if predicted!=1:
            continue            
        oriPatch_gray = np.float32(cv2.cvtColor(oriPatch, cv2.COLOR_RGB2GRAY))
        pImg = cv2.goodFeaturesToTrack(np.uint8(oriPatch_gray), **feature_params_Detect)       
        
        if pImg is None:
            print('No Points Detected')
            continue
        else:
            pImg[:,:,0]+=m-2*r
            pImg[:,:,1]+=n-2*r
        
            pPre_H = cv2.perspectiveTransform(pImg, np.linalg.inv(H_back))
            pPre_of, st2, err = cv2.calcOpticalFlowPyrLK(frame0, frame1, pImg, pPre_H.copy(), **lk_param)      
     
            #print(err)
            
            #IDs = np.int32([tr.classID for tr in Dotpt if tr.location[0]>=mcur-3*r and tr.location[0]<=mcur+3*r and tr.location[1]>=ncur-3*r and tr.location[1]<=ncur+3*r])
            #IDs = np.unique(IDs)
            #print('allpointsIDs', IDs)
            #Patches = [tr for tr in PatchPt if tr.location[0]>=mcur-N*r and tr.location[0]<=mcur+N*r and tr.location[1]>=ncur-N*r and tr.location[1]<=ncur+N*r] 
            Patches = [tr for tr in PatchPt if mcur>=tr.location[0]-2*r-N and mcur<=tr.location[0]+2*r+N and ncur>=tr.location[1]-2*r-N and ncur<= tr.location[1]+2*r+N]
                      #tr.location[0]>=mcur-N*r and tr.location[0]<=mcur+N*r and tr.location[1]>=ncur-N*r and tr.location[1]<=ncur+N*r] 
            #print('numberof Patches', len(Patches))
            pPre_H = pPre_H[st2>0]
            pPre_of = pPre_of[st2>0]
            pImg = pImg[st2>0]
            
            if pImg.shape[0]==0:
                continue
            if len(Patches)==0:
                maxID+=1
                [locx, locy] = pPre_of.reshape(-1,2).mean(0)
                [dtx,dty] = (pPre_H-pPre_of).reshape(-1,2).mean(0)
                [stdx,stdy] = (pPre_H-pPre_of).reshape(-1,2).std(0)              

                
                #print('center', locx, locy)
                #print('ptloc', reallocx, reallocy)
                newPatch = FeaturePatchV1(maxID, locx, locy,  dtx, dty, stdx, stdy,1)
                newPatch.Kalman1 = Kalman2D()
                newPatch.Kalman1.update(dtx,dty)
                PatchPt.append(newPatch)
                for (x,y), (xpre_h,ypre_h), (xpre_of, ypre_of) in zip(pImg.reshape(-1,2), pPre_H.reshape(-1,2), pPre_of.reshape(-1,2)):
                    newpt = FeatrueDotsV1(xpre_of, ypre_of, xpre_h-xpre_of, ypre_h-ypre_of, maxID)
                    Dotpt.append(newpt)
                    cv2.rectangle(vis,(np.int16(x-1),np.int16(y-1)),(np.int16(x+1),np.int16(y+1)), (0,255,255),1)
            else:
                for (x,y), (xpre_h,ypre_h), (xpre_of, ypre_of) in zip(pImg.reshape(-1,2), pPre_H.reshape(-1,2), pPre_of.reshape(-1,2)):
                    [dx, dy] = [xpre_h-xpre_of, ypre_h-ypre_of] 
                    dist = [abs(dx -tr.meanMotionDifference[0])+abs(dy-tr.meanMotionDifference[1]) for tr in Patches]
                    #print('dist:', dist)
                    dotID = Patches[np.argmin(dist)].classID
                    #print('dotIDs:', dotID)
                    newpt = FeatrueDotsV1(xpre_of, ypre_of, xpre_h-xpre_of, ypre_h-ypre_of, dotID)
                    Dotpt.append(newpt)
                    cv2.rectangle(vis,(np.int16(x-1),np.int16(y-1)),(np.int16(x+1),np.int16(y+1)), (255,0,255),1)   

    return vis, Dotpt, PatchPt, maxID

def writeDetect(outputFeature, radius, PatchPt,w, h):
    R = 2*radius
    for pt in PatchPt:
        [x,y] = pt.location
        xxC,yyC = boundary(x,y,radius, w,h)
        outputFeature = outputFeature+"("+str(yyC-R)+', '+str(xxC-R)+', '+str(yyC+R)+', '+str(xxC+R)+")"+', '
    return outputFeature
def visDetect(PatchPt, oriImage, radius, w, h):
    R = 2*radius
    for pt in PatchPt:
        [x,y] = pt.location
        #print('locs:', x,y)
        xxC,yyC = boundary(x,y,radius, w,h)
        cv2.rectangle(oriImage,(np.int16(xxC-R),np.int16(yyC-R)),(np.int16(xxC+R),np.int16(yyC+R)), (255,0,255),1)
    return oriImage

def visDotft(oriImage, Dotft, w, h):
    for pt in Dotft:
        [x,y] = pt.location
        #print('locs:', x,y)
        xxC,yyC = boundary(x,y,1, w,h)
        #draw_str(oriImage, np.int16(xxC-2), np.int16(yyC-2), 'ID: %d' % (pt.classID))
        cv2.rectangle(oriImage,(np.int16(xxC-1),np.int16(yyC-1)),(np.int16(xxC+1),np.int16(yyC+1)), (0,0,255),1)
        

    return oriImage

def visDetect_Kalman(PatchPt, oriImage, radius, w, h):
    R = 2*radius
    for pt in PatchPt:
        [x,y] = pt.location
        #print('locs:', x,y)
        xxC,yyC = boundary(x,y,radius, w,h)
        draw_str(oriImage, np.int16(xxC-R), np.int16(yyC-R), 'ID: %d, %d, %d' % (pt.classID, pt.invisibleCount, pt.history))
        cv2.rectangle(oriImage,(np.int16(xxC-R),np.int16(yyC-R)),(np.int16(xxC+R),np.int16(yyC+R)), (0,0,255),1)
        
        [xh, yh] = pt.trackLocation
        xhC, yhC = boundary(xh,yh,radius+1, w,h)
        cv2.rectangle(oriImage,(np.int16(xhC-R-2),np.int16(yhC-R-2)),(np.int16(xhC+R+2),np.int16(yhC+R+2)), (0,255,0),1)
    return oriImage

def visPosPatch_Kalman(dt_lable, gt_lables, detectedLocs, oriImage, radius,):
    R = 2*radius
    h,w,c = oriImage.shape
    for dt,gt,(x,y) in zip(dt_lable,gt_lables, detectedLocs):
        xxC,yyC = boundary(x,y,radius, w,h)
        if dt!=1:
            #cv2.rectangle(oriImage,(np.int16(xxC-R),np.int16(yyC-R)),(np.int16(xxC+R),np.int16(yyC+R)), (255,255,0),1)
            continue
        
        cv2.rectangle(oriImage,(np.int16(xxC-R),np.int16(yyC-R)),(np.int16(xxC+R),np.int16(yyC+R)), (0,255,255),1)
        
    return oriImage





def generatePatches_MV_trackV1(trackedpt, gray, Xt, H_back, lk_params_track_ori, radius,w, h, Xt_1, gt_ft_maske):
    p0 = np.float32([tr.location for tr in trackedpt]).reshape(-1, 1, 2)
    pPers = cv2.perspectiveTransform(p0, np.linalg.inv(H_back))
    p1, st1, err = cv2.calcOpticalFlowPyrLK(np.uint8(gray), np.uint8(Xt), p0, pPers.copy(), **lk_params_track_ori) 
    pdummy = cv2.perspectiveTransform(p1, H_back)
    p0r, st0, err = cv2.calcOpticalFlowPyrLK(np.uint8(Xt), np.uint8(gray), p1, pdummy, **lk_params_track_ori)
    d1 = abs(p0-p0r).reshape(-1, 2).max(-1)
    
    d = abs(pPers-p1).reshape(-1, 2).max(-1)               
    dt_x = (p1-pPers).reshape(-1, 2)[:,0].reshape(-1,1)
    dt_y = (p1-pPers).reshape(-1, 2)[:,1].reshape(-1,1)
                
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
                
    ft_mv = np.hstack([dt_x, dt_y,mag_dt, theta_dt, mag_d, theta_d, d1.reshape(-1,1)])
                #print(ft_mv)
    #score_mv = mvmodel.predict(ft_mv, batch_size = 1000000)
    p0_int = np.int32([tr.location for tr in trackedpt]).reshape(-1,  2)
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
    return d1, d, p1, pPers, p0, st1, np.array(ft_mv), np.array(Patches), np.array(gt_labels)