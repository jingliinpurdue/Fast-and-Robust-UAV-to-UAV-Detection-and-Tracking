
import cv2

import numpy as np
import math
import cv2
import random
import time
import sys
from util.kalman2d import Kalman2D
import operator


def errWarn():
    '''More inputs are needed.
    "python Feature_BackgroundSubtracted_Detection_V6_Debug_Groundtruth_With_Detection.py  videoName 4coordinates_for_pitut_tube filename DebugMode feature_thr resizeFactor feature_thr_ori thr_dt"
    To use this code you need the following parameters:
    4coordinates_for_pitut_tube:
    location for the pitut tube
    filename:
    name of file to record the results
    DebugMode:
    0:not show the detect results, 1: show the detect result
    feature_thr:
    quality of feature points in background subtracted image, by default for original size, it is 1.5
    resizeFacter:
    To achieve real time processing, video might be in lower resolution, parameter tuning is reccommended for different resolution
    feature_thr_ori:
    quality level of feature point detection in the first step(0.001 for original size)
    thr_dt:
    threshold for feature pruning (motion difference) 1.8(recommended for original size)
    thr_density:
    threshold for cluster pruning (feature point density in each cluster) 0.02(recommended for original size)
    
        '''
    print (errWarn.__doc__)

# We'll create a Track class which will track features
class FeatrueDotsV1:
    """Simple Track class
        status is true if the feature is being tracked
        location is x y coordinate of feature
        classID is which cluster it is belong to
        motionDifferenceA is the angle of motion difference
        motionDifferenceM is the magnitude of motion difference"""
    # class initialization function
    def __init__(self,x,y, dx,dy,index):
        self.location=[x,y]
        self.classID=index
        self.motionDifference=[dx,dy]

# We'll create a Cluster class which will record clusters    
    
# We'll create a Cluster class which will record clusters
class FeaturePatchV1:
    """Simple Track class
        status is true if the feature is being tracked
        location is x y coordinate of feature
        classID is which cluster it is belong to
        classType is 1 if it is foreground 0 is background
        box is 4 coordinate record for the center and size of the bbox
        stdMotionDifference is angle variance for motion differences for features inside cluster
        meanMotionDifference is angle main 
        density is the number of features divided by area of the cluster
        historyStatus records for/background in previous frames
        historybox is bbox in previous frames
        history is the number of frames the cluster is being detected"""
    # class initialization function
    def __init__(self,classId, x, y,meandx, meandy, stdx, stdy, hist):
        self.classID = classId
        self.box =[]
        self.location = [x, y]
        self.totalinvisibleCount = 0
        self.meanMotionDifference=[meandx, meandy]
        self.stdMotionDifference=[stdx, stdy]
        self.trackLocation=[x, y]
        self.history = hist            
        self.trackedStatus = False      
        self.invisibleCount = 0      
        self.Kalman1 = None
        self.Kalman2 = None




        
def boundary(x,y,radius, w,h):
    if y+2*radius>=h:
        y=h-2*radius-1
    if y-2*radius<0:
        y = 2*radius
    if x+2*radius>=w:
        x= w-2*radius-1
    if x-2*radius<0:
        x = 2*radius
    return np.int(x), np.int(y)





                













         
def draw_str(dst, x, y, s):
    # show text in the image
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)


def checkedTrace(img0, img1, p0,lk_params, back_threshold = 1.0):
    """ one-directional feature point tracking:
        track feature points between two images:
        p0: detected feature point in img0
        p1: tracked point in img1 for p0
        lk_params: parameters for LK track
        """
    # Track p0(feature points in img0) in img1 and the corresponding points is p1
    p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
    st = st.reshape(-1,1).max(-1)
    status = st>0
    return p1, status


def maskOut(blockmask, oripoint):
    """ Only keep feature points fall on the region of interest(ex. image without pitot tube region)
        blockmask has 0 on the pitot tube region and 1 else
    # that is the same as keep feature points which satisfy blockmask[py,px]>0
    """
    intpoint = oripoint.copy()
    intpoint = intpoint.reshape((-1, 2))
    h,w = blockmask.shape
    i = np.array([1,0])
    intpoint = intpoint[:,i]
    intpoint = np.int16(intpoint)
    # keep feature points which satisfy blockmask[py,px]>0
    # keepstatus represents whether the point lies inside the region of interest
    keepstatus = blockmask[intpoint[:,0]%h,intpoint[:,1]%w] > 0
    keepstatus = keepstatus.reshape((-1, 1))
    # only keep feature points fall in the region of interest
    p_out = oripoint[keepstatus]
    # return only the feature points falling in the region of interest
    return p_out

def maskBlock(estimate,original,blocks):
    """set the region of pitot tube to be 0 in background subtracted image
        """
    # first compute |estimate-original|
    error = cv2.absdiff(estimate,original)
    # then make the region of pitot tube to be black
    maskedError = cv2.multiply(error,blocks)
    # return maskedError
    return np.uint8(maskedError)



def PerspCompensate(Zx, H_1,Xm):
    """Given the prespective transform matrix, compute the motion compensated image
        Zx is the current frame
        Xm is the previous frame
        H_1 is the estimated perspective transform between Xm and Zx
        """
    h,w = Zx.shape
    # perspective transform warp
    compenstaedZ = cv2.warpPerspective(np.uint8(Zx), H_1, (w,h), np.uint8(Xm), borderMode=cv2.BORDER_TRANSPARENT)
    # convert the image into float64
    compenstaedZ = np.float32(compenstaedZ)
    return compenstaedZ

def computeError(preFrame,curFrame,H):
    e_back = PerspCompensate(curFrame,H,preFrame)
    error = np.float32(cv2.absdiff(e_back,preFrame))
    return error

def oneDirect_error(preFrame,curFrame,fp,blocks,lamda,lk_params,use_ransac):
    """This is the subroutine to compute the onedirectional error.
        Input: previous frame Xt-1, 
               current frame Xt, 
               fp(features in Xt-1), 
               blocks(in order to mask out pitot tube),
               lambda(to control use weighted sum or minumum)
        Output: bidirectional error
        """
    # find corresponding points in Xt-2 and Xt
    p_2, trace_status_2 = checkedTrace(np.uint8(preFrame), np.uint8(curFrame), fp,lk_params)
    
    # compute perspective transform for both directions
    H_2, status_2 = cv2.findHomography(p_2[trace_status_2], fp[trace_status_2], (0, cv2.RANSAC)[use_ransac], 10.0)
    
    
    # compute error image for both directions
    e_back = PerspCompensate(curFrame, H_2, preFrame)
    
    # mask out the pitot tube region
    error2 = np.float32(np.abs(maskBlock(e_back,preFrame,blocks)))
    
    
    # compute the final bidirectional error
    #finalError = np.float64(cv2.addWeighted(cv2.addWeighted(error1,0.5,error2,0.5,0),1,cv2.absdiff(error1,error2),-1/2*lamda,0))#cv2.addWeighted(error1,0.5,error2,0.5,0)
    
    return error2, H_2, p_2[trace_status_2]

def oneDirect(preFrame,curFrame,fp,blocks,lamda,lk_params_track,use_ransac):
    """This is the subroutine to compute the onedirectional error.
        Input: previous frame Xt-1,
               current frame Xt,
               fp(features in Xt-1),
               blocks(in order to mask out pitot tube),
               lambda(to control use weighted sum or minumum)
        Output: onedirectional error
        """
    # find corresponding points in Xt-2 and Xt
    #p_1, trace_status_1 = checkedTrace(np.uint8(preFrame), np.uint8(preFrame_1), fp)
    p_2, trace_status_2 = checkedTrace(np.uint8(preFrame), np.uint8(curFrame), fp,lk_params_track)
    
    # compute perspective transform for both directions
    #H_1, status_1 = cv2.findHomography(p_1[trace_status_1], fp[trace_status_1], (0, cv2.RANSAC)[use_ransac], 10.0)
    H_2, status_2 = cv2.findHomography(p_2[trace_status_2], fp[trace_status_2], (0, cv2.RANSAC)[use_ransac], 10.0)
    
    return H_2, p_2[trace_status_2]


def backgroundMotion(preFrame,preFrame_1,curFrame,fp,blocks,lamda,lk_params,use_ransac):
    """This is the subroutine to compute the bidirectional error.
        Input: previous frame Xt-1,
               frame before previous frame Xt-2,
               current frame Xt,
               fp(features in Xt-1),
               blocks(in order to mask out pitot tube),
               lambda(to control use weighted sum or minumum)
        Output: bidirectional error


        """
    
    # find corresponding points in Xt-2 and Xt
    #p_1, trace_status_1 = checkedTrace(np.uint8(preFrame), np.uint8(preFrame_1), fp)
    p_2, trace_status_2 = checkedTrace(np.uint8(preFrame), np.uint8(curFrame), fp,lk_params)
    
    # compute perspective transform for both directions
    #H_1, status_1 = cv2.findHomography(p_1[trace_status_1], fp[trace_status_1], (0, cv2.RANSAC)[use_ransac], 10.0)
    H_2, status_2 = cv2.findHomography(p_2[trace_status_2], fp[trace_status_2], (0, cv2.RANSAC)[use_ransac], 10.0)
    
    
    return H_2, p_2[trace_status_2]


def backgroundsubtraction(preFrame,preFrame_1,curFrame,fp,blocks,lamda,lk_params,use_ransac):
    """This is the subroutine to compute the bidirectional error.
        Input: previous frame Xt-1, 
               frame before previous frame Xt-2, 
               current frame Xt, 
               fp(features in Xt-1), 
               blocks(in order to mask out pitot tube),
               lambda(to control use weighted sum or minumum)
        Output: bidirectional error        
        """
    # find corresponding points in Xt-2 and Xt
    p_1, trace_status_1 = checkedTrace(np.uint8(preFrame), np.uint8(preFrame_1), fp,lk_params)
    p_2, trace_status_2 = checkedTrace(np.uint8(preFrame), np.uint8(curFrame), fp,lk_params)
    
    pt_status = np.logical_and(trace_status_1, trace_status_2)
    #print trace_status_1, trace_status_2, pt_status
    # compute perspective transform for both directions
    H_1, status_1 = cv2.findHomography(p_1[pt_status], fp[pt_status], (0, cv2.RANSAC)[use_ransac], 10.0)
    H_2, status_2 = cv2.findHomography(p_2[pt_status], fp[pt_status], (0, cv2.RANSAC)[use_ransac], 10.0)
    
    
    # compute error image for both directions
    e_back = PerspCompensate(curFrame, H_2, preFrame)
    e_fore = PerspCompensate(preFrame_1, H_1, preFrame)
    
    # mask out the pitot tube region
    error2 = np.abs(maskBlock(e_back,preFrame,blocks))
    error1 = np.abs(maskBlock(e_fore,preFrame,blocks))    
    
    # compute the final bidirectional error
    finalError = np.float32(cv2.addWeighted(cv2.addWeighted(error1,0.5,error2,0.5,0),1,cv2.absdiff(error1,error2),-1/2*lamda,0))#cv2.addWeighted(error1,0.5,error2,0.5,0)    
    
    return finalError, H_2, p_2[pt_status]





def generatePatches(frameidx, gray, Xt, weightedError, centers, H_back, ftparmes, ftparmes_ori,lk_params_track, radius, Xt_1, Xt_color, gt_mask, gt_img):
    vis = gt_img.copy()
    h, w = gray.shape
    dt_mask = np.zeros_like(gray)
    DetectNo = 0
    HitNo = 0
    FANo = 0

    pall = []

    weightedError *= 255.0/weightedError.max()
    featuresforBackgroundSubtractedImage = cv2.cvtColor(np.uint8(weightedError),cv2.COLOR_GRAY2RGB)
    ft = cv2.goodFeaturesToTrack(np.uint8(weightedError),  **ftparmes)
    pall.append(ft)

    
    posIndex=[]
    posPatches = []
    posPatches_errImg = []
    negPatches = []
    negPatches_errImg = []
    posPatches_gt = []
    posPatches_errImg_gt = []

    posdetect=0
    for p in pall:
        if p is None:
            continue
        if len(p.shape)!=3:
            continue

        frame0, frame1 = np.uint8(gray), np.uint8(Xt)
        # find corresponding points pCur  in current frame for pPre
        pCur, st, err = cv2.calcOpticalFlowPyrLK(frame0, frame1, p, None, **lk_params_track)
        # track back pCur in previous frame
        p0Pre, st, err = cv2.calcOpticalFlowPyrLK(frame1, frame0, pCur, None, **lk_params_track)


        # compute distance between the location of original feature points and tracked feature points
        d = abs(p-p0Pre).reshape(-1, 2).max(-1)
        # keep features have good matching ones
        good_frame = d < 1
        # compute corresponding location based on perpective transform
        converted = cv2.perspectiveTransform(pCur, H_back)
        for (x, y),(xx,yy), (xhat,yhat),good_flag in zip( pCur.reshape(-1, 2),p.reshape(-1, 2),converted.reshape(-1,2), good_frame):
            #if not good_flag:
                #cv2.rectangle(vis,(np.int16(xx-1),np.int16(yy-1)),(np.int16(xx+1),np.int16(yy+1)), (255,0,0),1)
                #continue

            DetectNo +=1
                          
            xxC,yyC = boundary(xx,yy,radius, w,h)
            cv2.rectangle(vis,(np.int16(xx-1),np.int16(yy-1)),(np.int16(xx+1),np.int16(yy+1)), (0,0,255),1)
            dt_mask[yyC-2*radius:yyC+2*radius,xxC-2*radius:xxC+2*radius]=255
            datapatch = Xt_1[yyC-2*radius:yyC+2*radius,xxC-2*radius:xxC+2*radius,:]
            errorpatch = weightedError[yyC-2*radius:yyC+2*radius,xxC-2*radius:xxC+2*radius]

            #print gt_mask[y-2*radius:y+2*radius,x-2*radius:x+2*radius]
            #print 'debug_1:', gt_mask.sum()
            if gt_mask[np.int16(yy),np.int16(xx)]>0:
                posdetect+=1
                posPatches.append(datapatch)
                posPatches_errImg.append(errorpatch) 
                posIndex.append(np.trim_zeros(np.unique(gt_mask[yyC-2*radius:yyC+2*radius,xxC-2*radius:xxC+2*radius])))                                                        
            else:
                FANo+=1
                negPatches.append(datapatch)
                negPatches_errImg.append(errorpatch)
 
    if len(centers)>0:
        p_ft_gt = np.float32(np.hstack([[centers]]))
        converted = cv2.perspectiveTransform(p_ft_gt, H_back)
        for (x,y),(xx,yy) in zip(centers,converted.reshape(-1, 2)):
            xx, yy = boundary(xx,yy,radius, w,h)
            x, y = boundary(x, y,radius, w,h)
            imgpatch = Xt_color[y-2*radius:y+2*radius,x-2*radius:x+2*radius,:]
            errorpatch = weightedError[yy-2*radius:yy+2*radius,xx-2*radius:xx+2*radius]
            posPatches_gt.append(imgpatch)
            posPatches_errImg_gt.append(errorpatch)

    if len(posIndex)>0:       
        detects = np.unique(np.hstack(posIndex))
        HitNo += detects.shape[0]

    return np.array(posPatches), np.array(posPatches_errImg), np.array(negPatches), np.array(negPatches_errImg), np.array(posPatches_gt), np.array(posPatches_errImg_gt), HitNo, DetectNo, FANo, vis


def generateMV(frameidx, gray, Xt, weightedError, centers, H_back, ftparmes, ftparmes_ori,lk_params_track, radius, Xt_1, Xt_color, gt_mask, gt_img):
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
    posMV = []
    negMV = []

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
            #print gt_mask[y-2*radius:y+2*radius,x-2*radius:x+2*radius]
            #print 'debug_1:', gt_mask.sum()
            #print (datapatch)
            if gt_mask[np.int16(yy),np.int16(xx)]>0:
                #print('bingo')
                posdetect+=1
                posMV.append(datapatch)
                posIndex.append(np.trim_zeros(np.unique(gt_mask[yyC-2*radius:yyC+2*radius,xxC-2*radius:xxC+2*radius])))                                                        
            else:
                FANo+=1
                negMV.append(datapatch)               

    if len(posIndex)>0:       
        detects = np.unique(np.hstack(posIndex))
        HitNo += detects.shape[0]
    #print(np.array(posMV).shape,np.array(negMV).shape)
    return np.array(posMV), np.array(negMV), HitNo, DetectNo, FANo

def generatePatches_online(gray, Xt, weightedError, ftparmes, Xt_1,radius,lk_params_track,H_back):
    h, w = gray.shape
    dt_mask = np.zeros_like(gray)

    pall = []

    weightedError *= 255.0/weightedError.max()
    featuresforBackgroundSubtractedImage = cv2.cvtColor(np.uint8(weightedError),cv2.COLOR_GRAY2RGB)
    ft = cv2.goodFeaturesToTrack(np.uint8(weightedError),  **ftparmes)
    pall.append(ft)

    locList = []
    locList_future = []
    locList_perspect = []
    dlist = []
    Patches = []
    Patches_errImg = []
    for p in pall:
        if p is None:
            continue
        if len(p.shape)!=3:
            continue
        frame0, frame1 = np.uint8(gray), np.uint8(Xt)
        # find corresponding points pCur  in current frame for pPre
        pCur, st, err = cv2.calcOpticalFlowPyrLK(frame0, frame1, p, None, **lk_params_track)
        # track back pCur in previous frame
        p0Pre, st, err = cv2.calcOpticalFlowPyrLK(frame1, frame0, pCur, None, **lk_params_track)
        # compute distance between the location of original feature points and tracked feature points
        d = abs(p-p0Pre).reshape(-1, 2).max(-1)
        # keep features have good matching ones
        good_frame = d < 1
        # compute corresponding location based on perpective transform
        converted = cv2.perspectiveTransform(p, np.linalg.inv(H_back))

        for (xx,yy),(x,y),(xhat,yhat),dist in zip(p.reshape(-1, 2), pCur.reshape(-1,2), converted.reshape(-1,2), d):
                          
            xxC,yyC = boundary(xx,yy,radius, w,h)
            datapatch = Xt_1[yyC-2*radius:yyC+2*radius,xxC-2*radius:xxC+2*radius,:]
            errorpatch = np.zeros_like(datapatch)
            errorpatch[:,:,0] = weightedError[yyC-2*radius:yyC+2*radius,xxC-2*radius:xxC+2*radius]
            errorpatch[:,:,1] = weightedError[yyC-2*radius:yyC+2*radius,xxC-2*radius:xxC+2*radius]
            errorpatch[:,:,2] = weightedError[yyC-2*radius:yyC+2*radius,xxC-2*radius:xxC+2*radius]
            Patches.append(datapatch)
            Patches_errImg.append(errorpatch) 
            locList.append([xx,yy])
            locList_future.append([x,y])
            locList_perspect.append([xhat,yhat])
            dlist.append(dist)

    return np.array(Patches_errImg),np.array(Patches),  np.array(locList), np.array(locList_future), np.array(locList_perspect), np.array(dlist)

def visDetection(vis, ft, Patches, radius, h, w):
    pt = np.float32([tr.location for tr in ft]).reshape(-1, 2) 
    #print (pt)
    for x,y in pt:
        cv2.rectangle(vis,(np.int16(x-1),np.int16(y-1)),(np.int16(x+1),np.int16(y+1)), (0,255,0),1)
    for pat in Patches:
        x,y,m,n = pat.box
        x,y = boundary(x,y,radius, w,h)
        if pat.classType:
            cv2.rectangle(vis,(np.int16(x-radius),np.int16(y-radius)),(np.int16(x+radius),np.int16(y+radius)), (0,0,255),1)            
        else:
            cv2.rectangle(vis,(np.int16(x-radius),np.int16(y-radius)),(np.int16(x+radius),np.int16(y+radius)), (255,0,0),1)
    return vis

def computeNo(Patches, gt_mask,  R, h, w):
    hit = 0
    fa = 0
    dt = 0
    ratiothr = 0.4
    for pat in Patches:
        if not pat.classType:
            continue
        dt +=1
        x,y,m,n = pat.box
        xxC,yyC = boundary(x,y,R, w,h)
        labels = np.unique(gt_mask[yyC-R:yyC+R,xxC-R:xxC+R])
        #print(labels)
        area = 1000000
        for lb in labels:
            lbarea = (gt_mask==lb).sum()
            if lbarea<area:
                area=lbarea
        if area>4*R*R:
            area = 1600
        overlaparea = (gt_mask[yyC-R:yyC+R,xxC-R:xxC+R]>0).sum()
        if np.float(overlaparea)/np.float(area)>ratiothr:
            hit+=1
        else:
            fa+=1
    return dt, hit, fa
    

def ClasifiyVis(f, radius,bacgroundsubtractedimg, oriImage, pred_y, detectedLocs,Locs_next, Locs_pers, d_match,Scores, gt_mask):
    ratiothr = 0.4
    dt=0
    fa=0
    hit=0
    R = 20
    h, w, cha = oriImage.shape
    for (xx,yy),pred, (ofx,ofy), (px,py),dist, sc in zip(detectedLocs,pred_y, Locs_next, Locs_pers, d_match, Scores.reshape(-1, 1)):
        #print(dist,sc[0])
        ft_feature = str(xx)+', '+ str(yy)+', '+ str(ofx)+', '+', '+ str(ofy)+', '+ str(px)+', '+ str(py)+', '+ str(dist)+', '+ str(sc[0])+', '
                
        #if pred!=0:
            #xxC,yyC = boundary(xx,yy,radius, w,h)
            #cv2.rectangle(oriImage,(np.int16(xxC-R),np.int16(yyC-R)),(np.int16(xxC+R),np.int16(yyC+R)), (0,255,255),1)
            #continue
        #dt+=1
        xxC,yyC = boundary(xx,yy,radius, w,h)
        labels = np.unique(gt_mask[yyC-R:yyC+R,xxC-R:xxC+R])
        #print(labels)
        area = 1000000
        for lb in labels:
            lbarea = (gt_mask==lb).sum()
            if lbarea<area:
                area=lbarea
        overlaparea = (gt_mask[yyC-R:yyC+R,xxC-R:xxC+R]>0).sum()
        if np.float(overlaparea)/np.float(area)>ratiothr:
            ft_feature+=str(int(1))+'\n'
            #hit+=1
            cv2.rectangle(oriImage,(np.int16(xxC-R),np.int16(yyC-R)),(np.int16(xxC+R),np.int16(yyC+R)), (0,255,0),1)
            cv2.rectangle(bacgroundsubtractedimg,(np.int16(xxC-R),np.int16(yyC-R)),(np.int16(xxC+R),np.int16(yyC+R)), (0,255,0),1)
        else:
            ft_feature+=str(int(0))+'\n'
            #fa+=1
            cv2.rectangle(oriImage,(np.int16(xxC-R),np.int16(yyC-R)),(np.int16(xxC+R),np.int16(yyC+R)), (0,0,255),1)
            cv2.rectangle(bacgroundsubtractedimg,(np.int16(xxC-R),np.int16(yyC-R)),(np.int16(yyC+R),np.int16(yyC+R)), (0,0,255),1)
        
        f.write(ft_feature)
    return bacgroundsubtractedimg, oriImage, dt, hit, fa

def ClasifiyVis_ori(f, radius,bacgroundsubtractedimg, oriImage, pred_y, detectedLocs,Locs_next, Locs_pers, d_match,Scores, gt_mask):
    ratiothr = 0.4
    dt=0
    fa=0
    hit=0
    R = 20
    h, w, cha = oriImage.shape
    for (xx,yy),pred, (ofx,ofy), (px,py),dist, sc in zip(detectedLocs,pred_y, Locs_next, Locs_pers, d_match, Scores):
        ft_feature = str(xx)+', '+ str(yy)+', '+ str(ofx)+', '+', '+ str(ofy)+', '+ str(px)+', '+ str(py)+', '+ str(dist)+', '+ str(sc)+', '
                
        #if pred!=0:
            #xxC,yyC = boundary(xx,yy,radius, w,h)
            #cv2.rectangle(oriImage,(np.int16(xxC-R),np.int16(yyC-R)),(np.int16(xxC+R),np.int16(yyC+R)), (0,255,255),1)
            #continue
        #dt+=1
        xxC,yyC = boundary(xx,yy,radius, w,h)
        labels = np.unique(gt_mask[yyC-R:yyC+R,xxC-R:xxC+R])
        #print(labels)
        area = 1000000
        for lb in labels:
            lbarea = (gt_mask==lb).sum()
            if lbarea<area:
                area=lbarea
        overlaparea = (gt_mask[yyC-R:yyC+R,xxC-R:xxC+R]>0).sum()
        if np.float(overlaparea)/np.float(area)>ratiothr:
            ft_feature+=str(int(1))+'\n'
            #hit+=1
            #cv2.rectangle(oriImage,(np.int16(xxC-R),np.int16(yyC-R)),(np.int16(xxC+R),np.int16(yyC+R)), (0,255,0),1)
            #cv2.rectangle(bacgroundsubtractedimg,(np.int16(xxC-R),np.int16(yyC-R)),(np.int16(xxC+R),np.int16(yyC+R)), (0,255,0),1)
        else:
            ft_feature+=str(int(0))+'\n'
            #fa+=1
            #cv2.rectangle(oriImage,(np.int16(xxC-R),np.int16(yyC-R)),(np.int16(xxC+R),np.int16(yyC+R)), (0,0,255),1)
            #cv2.rectangle(bacgroundsubtractedimg,(np.int16(xxC-R),np.int16(yyC-R)),(np.int16(yyC+R),np.int16(yyC+R)), (0,0,255),1)
        print(ft_feature)
        f.write(ft_feature)
    return bacgroundsubtractedimg, oriImage, dt, hit, fa


def generatePatches_test(gray, Xt, weightedError, H_back, lalparmes, lk_params_track, radius, color, future_color):
    h, w = gray.shape
    weightedError *= 255.0/weightedError.max()
    featuresforBackgroundSubtractedImage = cv2.cvtColor(np.uint8(weightedError),cv2.COLOR_GRAY2RGB)
    ft = cv2.goodFeaturesToTrack(np.uint8(weightedError),  **lalparmes)
    pall=[]
    pall.append(ft)


    locList = []
    prePatch = []
    curPatch=[]
    for p in pall:
        if p is None:
            continue
        if len(p.shape)!=3:
            continue
        frame0, frame1 = np.uint8(gray), np.uint8(Xt)
        # find corresponding points pCur  in current frame for pPre
        pCur, st, err = cv2.calcOpticalFlowPyrLK(frame0, frame1, p, None, **lk_params_track)
        # track back pCur in previous frame
        p0Pre, st, err = cv2.calcOpticalFlowPyrLK(frame1, frame0, pCur, None, **lk_params_track)
        # compute distance between the location of original feature points and tracked feature points
        d = abs(p-p0Pre).reshape(-1, 2).max(-1)
        # keep features have good matching ones
        good_frame = d < 1
        # compute corresponding location based on perpective transform
        converted = cv2.perspectiveTransform(pCur, H_back)
        for (x, y),(xx,yy), (xhat,yhat),good_flag in zip( pCur.reshape(-1, 2),p.reshape(-1, 2),converted.reshape(-1,2), good_frame):
            if not good_flag:
                continue
            x,y = boundary(x,y,radius, w,h)                
            xx,yy = boundary(xx,yy,radius, w,h)

            datapatch = future_color[y-2*radius:y+2*radius,x-2*radius:x+2*radius,:]
            errorpatch = np.zeros_like(datapatch)
            errorpatch[:,:,0] = weightedError[yy-2*radius:yy+2*radius,xx-2*radius:xx+2*radius]
            errorpatch[:,:,1] = weightedError[yy-2*radius:yy+2*radius,xx-2*radius:xx+2*radius]
            errorpatch[:,:,2] = weightedError[yy-2*radius:yy+2*radius,xx-2*radius:xx+2*radius]

            locList.append([x,y])
            prePatch.append(errorpatch)
            curPatch.append(datapatch)#weightedError[yy-2*radius:yy+2*radius,xx-2*radius:xx+2*radius])


    return np.array(locList), np.array(prePatch), np.array(curPatch)







affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

def deskew(img):
    SZ = img.shape[0]
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img


def hog(img):
    SZ = img.shape[0]
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bin_n = 16
    bins = np.int32(bin_n*ang/(2*np.pi))
    bin_cells = bins[:SZ/2,:SZ/2], bins[SZ/2:,:SZ/2], bins[:SZ/2,SZ/2:], bins[SZ/2:,SZ/2:]
    mag_cells = mag[:SZ/2,:SZ/2], mag[SZ/2:,:SZ/2], mag[:SZ/2,SZ/2:], mag[SZ/2:,SZ/2:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist

def Histgram(img,N):
    image = img*255.0/(img.max()+1)
    hist = cv2.calcHist([image], [0], None, [N], [0, 256])
    #print hist
    hist = np.hstack(hist)
    #print hist
    return hist

def biDirect(preFrame,preFrame_1,curFrame,fp,blocks,lamda,lk_params,use_ransac):
    """This is the subroutine to compute the bidirectional error.
        Input: previous frame Xt-1,
               frame before previous frame Xt-2,
               current frame Xt,
               fp(features in Xt-1),
               blocks(in order to mask out pitot tube),
               lambda(to control use weighted sum or minumum)
        Output: bidirectional error


        """
    
    # find corresponding points in Xt-2 and Xt
    #p_1, trace_status_1 = checkedTrace(np.uint8(preFrame), np.uint8(preFrame_1), fp)
    p_2, trace_status_2 = checkedTrace(np.uint8(preFrame), np.uint8(curFrame), fp,lk_params)
    
    # compute perspective transform for both directions
    #H_1, status_1 = cv2.findHomography(p_1[trace_status_1], fp[trace_status_1], (0, cv2.RANSAC)[use_ransac], 10.0)
    H_2, status_2 = cv2.findHomography(p_2[trace_status_2], fp[trace_status_2], (0, cv2.RANSAC)[use_ransac], 10.0)
    
    
    return H_2, p_2[trace_status_2]


def CCL(binaryMask,annotation,Ind):
    """ This is the subrountine to find clusters for the detected feature points
        Input: binaryMask: 1 for regions with detected feature points
               annotation: clusering results
               Ind: index of cluster
        Output:clustering results
        """
    newMask = np.zeros_like(binaryMask)
    #ret,binaryM = cv2.threshold(binaryMask,127,255,0)
    #opencv3
    #im2, contours, hierarchy = cv2.findContours( np.uint8(binaryMask.copy()), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #opencv2
    contours,hierarchy = cv2.findContours(np.uint8(binaryMask),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        xs,ys,ws,hs = cv2.boundingRect(cnt)
        cv2.rectangle(newMask,(xs,ys),(xs+ws,ys+hs),255,-1)
        if Ind is 0:
            cv2.rectangle(annotation,(xs,ys),(xs+ws,ys+hs),(0,255,0),2)
        else:
            cv2.rectangle(annotation,(xs,ys),(xs+ws,ys+hs),(0,0,255),2)


    return newMask,annotation



def computeFeature(clustertr,featureM,featureX,featureY, indices):
    """This is to compute the standard derivation and mean of the motion diferences in each cluster
        stdMotionDifference: angle variance for motion differences
        meanMotionDifference: angle main for motion differences
        """
    meanX = np.mean(featureX[indices])
    meanY = np.mean(featureY[indices])
    pointCount = 0
    sumAngle = 0
    #print 'testFeaturecoputation:'
    #print 'featureX',featureX[indices]
    #print 'featureX-meanX', featureX[indices]-meanX,'featureY-meanY', featureY[indices]-meanY
    
    for yy,xx in zip(featureY[indices]-meanY,featureX[indices]-meanX):
        pointCount+=1
        sumAngle+=(np.math.atan2(yy,xx))*(np.math.atan2(yy,xx))
            #print 'sumAngle:', sumAngle,'featureNum:',pointCount,'feature:',sumAngle/pointCount
    clustertr.stdMotionDifference = sumAngle/pointCount
#print clustertr.stdMotionDifference
    #clustertr.meanMotionDifference=np.mean(featureM[indics])
    return clustertr

def forgroundCheckIni(clustertr,ID1,trID,trType,indices,thr_den,thr_stdMotionDifference):
    """ This is the subrountine to classfy each cluster based on std of motion difference and feature density
        foreground cluster: angle variance for motion differences is small and density is large
        """
    #if clustertr.stdMotionDifference<=5 and clustertr.density>0.2:
    if clustertr.stdMotionDifference<=thr_stdMotionDifference and clustertr.density>thr_den:
        clustertr.classType= 1
        ID1+=1
        clustertr.classID= ID1
        trID[indices]=ID1
        trType[indices]=1
    else:
        clustertr.classType = 2
        ID1+=1
        clustertr.classID= ID1
        trID[indices]=ID1
        trType[indices]=2

    return clustertr,ID1,trID,trType



def featureClustering(initialMask,featurePoints,startID,returnedIndex,returnedType,featureMotion,MotionX, MotionY, trackClusters,WhetherKeep,thr_den,thr_stdMotionDifference):
    """clustering for detcted features
        initialMask: detection result with 1 on regions have detected feature points and 0 for no
        featurePoints: detected feature points
        startID: largest index of current detected cluster
        returnedIndex: indices with newly detected clusters
        returnedType: cluster type 
        featureMotion: motion vector for feature point
        trackClusters: record the bbox in previous frames
        bbox: detected bounding box
        
        returns: indexReturn,typeReturn,startID,trackClusters
        
        """
    indexMask = np.zeros_like(initialMask)
    
    indexReturn = np.zeros_like(returnedIndex)
    typeReturn = np.zeros_like(returnedType)
    
    boundy,boundx = indexMask.shape

    #ret,binaryM = cv2.threshold(binaryMask,127,255,0)
    #opencv3
    #im2, contours, hierarchy = cv2.findContours( np.uint8(binaryMask.copy()), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #opencv2
    Ind = 1
    _,contours,hierarchy = cv2.findContours(np.uint8(initialMask),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        
        xs,ys,ws,hs = cv2.boundingRect(cnt)
        
        highy = max(0,ys-1)
        lowy = min(boundy-1,ys+hs+1)
        leftx = max(0,xs-1)
        rightx = min(boundx-1,xs+ws+1)
        
        indexMask[highy:lowy,leftx:rightx]=Ind
        Ind +=1
            
    startcound = 0
    
    for (x, y) in featurePoints.reshape(-1, 2):
        startcound+=1
        y1 = min(np.int16(y),boundy-1)
        x1 = min(np.int16(x),boundx-1)
        #print "index",indexMask[y1,x1]
        #print "location", y1,x1
        returnedIndex[startcound-1] = indexMask[y1,x1]
    
    indstart = 1
    while indstart<Ind:
        
        #print indstart
        indexIn = returnedIndex==indstart
        indstart+=1
        #print indexIn
        bbox=Detection()
        
        PointstoClust=featurePoints[indexIn,:]
        if WhetherKeep is False:
            if PointstoClust.shape[0]==0 or PointstoClust.shape[0]==1:
                continue
        else:
            if PointstoClust.shape[0]==0:
                continue
        bbox = computeFeature(bbox,featureMotion,MotionX,MotionY,indexIn)

        maxx,maxy = PointstoClust.max(axis=0)
        minx,miny = PointstoClust.min(axis=0)

        bbox.box = [minx,miny,maxx,maxy]
        bbox.density = np.float32(PointstoClust.shape[0])/((maxx-minx+1)*(maxy-miny+1))
        
        bbox,startID,indexReturn ,typeReturn= forgroundCheckIni(bbox,startID,indexReturn,typeReturn,indexIn,thr_den,thr_stdMotionDifference)
    
        trackClusters.append(bbox)
    
    return indexReturn,typeReturn,startID,trackClusters
######

def PointtoCluster(pointTracks_all, kernelSize,mask_Cluster,firstID,trackCluster,WhetherKeep,thr_den,thr_stdMotionDifference):
    """Assign each feature point to clusters
        Input:
            pointTracks_all: all feature points
            kernelSize: if the feature points close together then belong to the same detection
            mask_Cluster: 1 for detected region
            firstID: largest index for current detection
            trackCluster: record history of detected cluster
            WhetherKeep: whether to keep on point detection
            thr_den: threshold for density criteria
            thr_stdMotionDifference: threshold for angle variance in cluster
        Output:
            detected clusters
            updated index
            updated feature points
        """
    pointTracks=[]
    for tr in pointTracks_all:
        if tr.status:
            continue
        pointTracks.append(tr)
    
    featurePoints = np.float32([tr.location for tr in pointTracks]).reshape(-1, 1, 2)
    ClusterIndex = np.int16([tr.classID for tr in pointTracks]).reshape(-1, 1)
    ClusterType = np.int16([tr.classType for tr in pointTracks]).reshape(-1, 1)
    angleFeature = np.float32([tr.motionDifferenceA for tr in pointTracks]).reshape(-1, 1)
    xFeature = np.float32([tr.motionDifferenceX for tr in pointTracks]).reshape(-1, 1)
    yFeature = np.float32([tr.motionDifferenceY for tr in pointTracks]).reshape(-1, 1)
    newmask = np.zeros_like(mask_Cluster)
    height,width = newmask.shape
    
    for x, y in np.int16(featurePoints).reshape(-1, 2):
        highy = max(0,y-kernelSize)
        lowy = min(height-1,y+kernelSize)
        leftx = max(0,x-kernelSize)
        rightx = min(width-1,x+kernelSize)
        
        newmask[highy:lowy,leftx:rightx]=255
    
    ClusterIndex, ClusterType,firstID, trackCluster = featureClustering(newmask,featurePoints,firstID,ClusterIndex, ClusterType, angleFeature,xFeature,yFeature,trackCluster,WhetherKeep,thr_den, thr_stdMotionDifference)
    
    for clust,tp, tr in zip(ClusterIndex,ClusterType,pointTracks):
        #tr.preclassID = tr.classID
        tr.classID = clust
        tr.classType = tp

    return pointTracks_all, trackCluster,firstID


def knn_search(x, D, K_nn,Dlabel):
    """ find K nearest neighbours of data among D 
        x: unlabeled data
        D:labled data
        Dlable: labels for D"""
    ndata = D.shape[1]
    K_nn = K_nn if K_nn < ndata else ndata
    # euclidean distances from the other points
    sqd = np.sqrt(((D - x[:,:ndata])**2).sum(axis=1))
    idx = np.argsort(sqd) # sorting
    # return the indexes of K nearest neighbours
    return idx[:K_nn], sqd,sqd[idx[:K_nn]],Dlabel[idx[:K_nn]]

def CheckingTrackingStatus(cluster_pre,cluster_cur,points,UAVtracks,maxTrack1,maxRealTrack1,hist_track):
    """ check whether being tracked for each detected cluster 
        cluster_pre: clusters detected in previous frame
        cluster_cur: current frame's detection
        point: detected feature points
        UAVtracks: kalman tracked for current frame
        maxTrack1: largest index for all the track
        maxRealTrack: largest index for foreground track
        hist_track: number of detection to decide whether to track
        """
    
    for cltr in cluster_pre:
        
        [x1,y1,m1,n1] = cltr.box
        featureID = np.int16([tr.classID for tr in points if tr.preclassID==cltr.classID]).reshape(-1)
        
        if featureID.shape[0]==0:
            continue
        counts = np.bincount(featureID)
        corID = np.argmax(counts)
        corType = np.int16([tr.classType for tr in cluster_cur if tr.classID==corID]).reshape(-1)
        # if the cluster is already being tracked break
        if cltr.status:
            for xxx in cluster_cur:
                if xxx.classID==corID:
                    xxx.status = True
                    xxx.classType = cltr.classType
                break
        # find correspondence between current detection and previous detection
        for xxx in cluster_cur:
            if xxx.classID==corID:
                [x,y,m,n] = xxx.box
                xxx.history=cltr.history+1

                dumy = cltr.historybox
                dumy.append(xxx.box)
                xxx.historybox = dumy
                dumy1 = cltr.historyStatus
                dumy1.append(corType)
                xxx.historyStatus = dumy1
                break
        # start new track for detections last for long enough
        for xxx in cluster_cur:
            if xxx.classID!=corID:
                continue
            if xxx.history<hist_track:
                break
            cTrack = []
            
            xxx.status = True
            BoxHisroty = xxx.historybox
            StatusHistory = xxx.historyStatus
            
            cTrack = ClusterTrack(x1,y1,m1,n1)
            

            maxTrack1+=1
            cTrack.TrackID = maxTrack1
            cTrack.classID =corID
            for Boxes in BoxHisroty:
                x,y,m,n = Boxes
                cTrack.Kalman1.update(x,y)
                cTrack.Kalman2.update(m,n)
            
            Types = np.int16(StatusHistory).reshape(-1)
            countsT = np.bincount(Types)
            cTType = np.argmax(countsT)
            xxx.classType = cTType
            cTrack.Type = cTType
            if cTType==1:
                maxRealTrack1+=1
                cTrack.TrackIDTrue = maxRealTrack1
            cTrack.curDetection = [x,y,m,n]
            UAVtracks.append(cTrack)
                #for pts in points:
                #if pts.classID ==corID:
#tr.status=True

    return cluster_pre,cluster_cur,points,UAVtracks,maxTrack1,maxRealTrack1





def updateOnepointCase(UAVtracks,cluster_pre,cluster_cur,points,kernelSize):
    """ For detected cluster contains only one feature points:
    # 1. if if belongs to some predicted mask, same as the predicted mask fore/background
    # 2. if it is tracked from previsous detection, same as the previous frame
    # 3. Newly detected point, find neares neighbor, if too far from any cluster keep it
    UAVtracks: Kalman track for detection
    cluster_pre: detection in previous frame
    luster_cur: detection in current frame
    points: detected feature points
    kernelSize: maximum distance for feature point to be consided as belong to cluster
    """
    for cltr in cluster_cur:
        subpointsID = np.int16([tr.classID for tr in points if tr.classID == cltr.classID]).reshape(-1, 1)
        subpointsIDpre = np.int16([tr.preclassID for tr in points if tr.classID == cltr.classID]).reshape(-1, 1)
        if subpointsIDpre.shape[0]==1:
            if cltr.status:
                
                cltr.classType=np.int16([UAVt.Type for UAVt in UAVtracks if UAVt.classID==cltr.classID]).reshape(-1,1)
            else:
                if subpointsIDpre==0:
                    
                    labels = np.int16([pts.classID for pts in points if pts.classID != cltr.classID]).reshape(-1, 1)
                    bpoints = np.float32([pts.location for pts in points if pts.classID != cltr.classID]).reshape(-1, 1, 2).reshape(-1,2)
                    featu = np.float32([pts.location for pts in points if pts.classID == cltr.classID]).reshape(-1, 1, 2).reshape(-1,2)
                    idxes,dists1,dists,Label = knn_search(featu, bpoints, 1,labels)
                    if dists.shape[0]==0:
                        cltr.classType=1
                        continue
                    #Note: Modified:05212017
                    #if dists.min()<kernelSize*2+1:
                    if dists.min()<kernelSize*2+1:
                        cltr.classType = np.int16([tr.classType for tr in cluster_cur if tr.classID==Label]).reshape(-1,1)
                    else:
                        cltr.classType = 1
            
                else:
                    
                    cltr.classType = np.int16([tr.classType for tr in cluster_pre if tr.classID == subpointsIDpre]).reshape(-1, 1)
    return cluster_cur

def ComputeDetection(dt, gt, gt_mask, dt_mask):
    a = 2
    gt_split = gt.split()
    dt_split = dt.split()
    
    TotalDetected = 0
    TotalMoving = 0
    Totalhit = 0
    TotalDetection = 0
    
    length = len(gt_split)
    count = 3
    bboxgt = []
    while count<length:
        uplefty = gt_split[count]
        uplefty = np.int16(uplefty[1:-1])
        upleftx = gt_split[count+1]
        upleftx = np.int16(upleftx[0:-1])
        downrighty = gt_split[count+2]
        downrighty = np.int16(downrighty[0:-1])
        downrightx = gt_split[count+3]
        downrightx = np.int16(downrightx[0:-2])
        bboxgt.append([upleftx-a,uplefty-a,downrightx+a,downrighty+a])
        TotalMoving+=1
        count+=4
        
    count = 3
    bboxdt = []
    length = len(dt_split)
    #dt_mask = np.zeros_like(gray)
    while count<length:
        uplefty = dt_split[count]
        uplefty = np.int16(uplefty[1:-1])
        upleftx = dt_split[count+1]
        upleftx = np.int16(upleftx[0:-1])
        downrighty = dt_split[count+2]
        downrighty = np.int16(downrighty[0:-1])
        downrightx = dt_split[count+3]
        downrightx = np.int16(downrightx[0:-2])
        bboxdt.append([upleftx-a,uplefty-a,downrightx+a,downrighty+a])
        count+=4
        TotalDetection+=1
    for minx,miny,maxx,maxy in bboxgt:
        if dt_mask[miny:maxy,minx:maxx].sum()>0:
            TotalDetected+=1
        else:
            print ("Missed Detection:")

    for minx,miny,maxx,maxy in bboxdt:
        if gt_mask[miny:maxy,minx:maxx].sum()>0:
            Totalhit+=1
        else:
            print ("Falsed Alarm:")
    return TotalDetected, TotalMoving, Totalhit, TotalDetection


def Tracking(UAVtracks,curClutster,points,w,h, dilateSize,hist_track):
    """# For each detected Bounding box, use Kalman Filter to generate predicton
    # based on the prediction, modify the current detection to obtain final detection
    UAVtracks: UAV Kalman track
    curClutster: detection in current frame
    points: detected feature points
    w,h: width and height of the video
    dilateSize: dilate detection to generate enlarged mask
    hist_track: number of detection to decide whether to track
    """
    tracksUAV=[]
    
    for truav in UAVtracks:
        x,y = truav.Kalman1.getEstimate()
        m,n = truav.Kalman2.getEstimate()
        truav.prediction = [x,y,m,n]
        if m<0 or n<0 or x>w or y>h or m>w or y>h or x<0 or y<0:
            x,y,m,n = truav.curDetection
        
        subpointstr = []
        for pts in points:
            if  (pts.location[0]>x-dilateSize and pts.location[0]<m+dilateSize and pts.location[1]>y-dilateSize and pts.location[1]<n+dilateSize) or pts.preclassID==truav.classID:
                pts.status=True
                pts.classID = truav.classID
                subpointstr.append(pts)
        for cltr in curClutster:
            if cltr.classID ==truav.classID:
                cltr.status=True
                break
    
        subpoints = np.float32([pts.location for pts in subpointstr]).reshape(-1,1,2).reshape(-1,2)
        #print 'poor pts:',subpoints,subpoints.shape
        if subpoints.shape[0]==0:
            if truav.Type==1:
                truav.invisibleCount+=1
                truav.curDetection = [x,y,m,n]
                truav.Kalman2.update(m,n)
                truav.Kalman1.update(x,y)
                if truav.invisibleCount<hist_track+1:
                    tracksUAV.append(truav)
                if truav.invisibleCount>hist_track:
                    for tr in points:
                        if tr.classID == truav.classID:
                            tr.status= False

        else:
            truav.invisibleCount=0
            maxx,maxy = subpoints.max(axis=0)
            minx,miny = subpoints.min(axis=0)
            truav.curDetection=[minx,miny,maxx,maxy]
            truav.Kalman1.update(minx,miny)
            truav.Kalman2.update(maxx,maxy)
            tracksUAV.append(truav)

    UAVtracks = tracksUAV
    return UAVtracks,curClutster,points

# We'll create a Cluster class which will record clusters
class FeaturePatch:
    """Simple Track class
        status is true if the feature is being tracked
        location is x y coordinate of feature
        classID is which cluster it is belong to
        classType is 1 if it is foreground 0 is background
        box is 4 coordinate record for the center and size of the bbox
        stdMotionDifference is angle variance for motion differences for features inside cluster
        meanMotionDifference is angle main 
        density is the number of features divided by area of the cluster
        historyStatus records for/background in previous frames
        historybox is bbox in previous frames
        history is the number of frames the cluster is being detected"""
    # class initialization function
    def __init__(self):
        self.classID = 0
        self.meancorner = 0
        self.stdcorner = 0
        self.maxcorner = 0
        self.classType = False
        self.box =[]
        self.meanMotionDifferenceX=0
        self.meanMotionDifferenceY=0
        self.stdMotionDifferenceX = 0        
        self.stdMotionDifferenceY = 0
        self.historyType=[]
        self.history = 0            
        self.trackedStatus = False      
        self.invisibleCount = 0      
        self.Kalman1 = None
        self.Kalman2 = None#Kalman2D(m,n)
# We'll create a Track class which will track features
class FeatrueDots:
    """Simple Track class
        status is true if the feature is being tracked
        location is x y coordinate of feature
        classID is which cluster it is belong to
        classType is 1 if it is foreground 0 is background
        preClassID is the cluster ID in the previous frame
        motionDifferenceA is the angle of motion difference
        motionDifferenceM is the magnitude of motion difference"""
    # class initialization function
    def __init__(self,x,y):
        self.location=[x,y]
        self.classID=0
        self.cornerNess=0
        self.motionDifferenceX=0
        self.motionDifferenceY=0
# We'll create a Track class which will track features
class KalmanTrack:
    """Simple Track class
        status is true if the feature is being tracked
        location is x y coordinate of feature
        classID is which cluster it is belong to
        classType is 1 if it is foreground 0 is background
        preClassID is the cluster ID in the previous frame
        motionDifferenceA is the angle of motion difference
        motionDifferenceM is the magnitude of motion difference"""
    # class initialization function
    def __init__(self,x,y):
        self.status = False
        self.location=[x,y]
        self.classID=0
        self.classType=0
        self.preclassID=0
        self.motionDifferenceA=0
        self.motionDifferenceX=0
        self.motionDifferenceY=0
        self.motionDifferenceM=0
        self.response = 0.0