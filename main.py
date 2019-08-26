from __future__ import print_function
import numpy as np
import math
import random
import time
import sys
import operator
import os
from numpy import zeros, newaxis
import re
#import matplotlib.pyplot as plt
import glob
#import skimage
#import skimage.io
#import scipy.io as scp
from sklearn.utils import shuffle

from util.Generate_pm_pa import *
from util.UAV_subfunctions import *
from util.Extract_Patch import *
from util.Detect_Patch import *

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
#import pandas


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
from keras.callbacks import ReduceLROnPlateau,EarlyStopping, ModelCheckpoint
from keras.models import load_model

videoPath = './Data/'
app_model_path = './models/max500_1_10_threelayers/'
app_model_path_track = './models/Appearance_OriImage/'

mvmodel_path = './models/motion/'
bimodel_path = './models/Adaboost/'

bimodel_path_track = './models/Adaboost_track/'
#defult is 15, 2
trackwinS =15

video_savePath_features  = './Experiment_Results/Final/Video/'
    
if not os.path.exists(video_savePath_features):
    print ("path doesn't exist. trying to make")
    os.makedirs(video_savePath_features)
    
video_savePath_Detection  =  './Experiment_Results/Final/txt/'
    
if not os.path.exists(video_savePath_Detection):
    print ("path doesn't exist. trying to make")
    os.makedirs(video_savePath_Detection)
par = []
par.append([0.001,40])


index= list(range(50))
print(index)
from sklearn.utils import shuffle
index = shuffle(index, random_state=42)
print("complete shuffling")
print(index)
maxD=4

for ind in range(1,2):
    appmodel=load_model(app_model_path+str(ind)+'.h5')
    appmodel.summary()

    appmodel_track=load_model(app_model_path_track+str(ind)+'.h5')
    appmodel_track.summary()
    
    mvmodel = load_model(mvmodel_path+str(ind)+'.h5')
    mvmodel.summary()
    
    combinemodel = joblib.load(bimodel_path+'fold'+str(ind)+'.pkl')
    
    combinemodel_track = joblib.load(bimodel_path_track+'fold'+str(ind)+'.pkl')
    a = 0.001
    b = 50
    #test_ind = index[10*(ind-1):10*(ind-1)+10]
    test_ind = index[10*(ind-1):10*(ind-1)+1]
    objNo = 0
    dtNo = 0
    htNo = 0
    allFA = 0
    for i in test_ind:
        all_params = dict(videoName = str(i+1),
                          downrightx = 350,
                          upleftx = 0,
                          downrighty = 780,
                          uplefty = 510,
                          fileName = 'supervised_SVM_'+str(i),
                          debug = 1,
                          qualityini = 0.005,#np.float32(sys.argv[4]),#1.5#,
                          K = 1,# number of previous frames before Xt-1
                          MaxCorns = 600,#np.int16(sys.argv[5]),#200,#600/(resizeFactor*resizeFactor),
                          mindist1 = 25,#np.int16(sys.argv[6]),#15,
                          quality = a,#np.float32(sys.argv[7]),#0.001 #,
                          maxcorners = 1000,#np.int16(sys.argv[8]),#100,#/(resizeFactor*resizeFactor),
                          mindist = b,#15,#np.int16(sys.argv[9]),#1,
                          use_ransac = True,
                          track_len = 10,# track_len: maximum number of points recorded in the track
                          lamda = 0,#taking average if bidirectional error
                          detect_interval = 6,
                         )
        

        print ('Video:',all_params['videoName'])
        videoName = all_params['videoName']
        uplefty = all_params['uplefty']
        downrighty = all_params['downrighty']
        upleftx = all_params['upleftx']
        downrightx = all_params['downrightx']
        fileName = all_params['fileName']
        debug = all_params['debug']
        K = all_params['K']
        qualityini = all_params['qualityini']
        MaxCorns = all_params['MaxCorns']
        mindist1 = all_params['mindist1']
        use_ransac = all_params['use_ransac']
        track_len = all_params['track_len']
        lamda = all_params['lamda']
        quality = all_params['quality']
        maxcorners = all_params['maxcorners']
        mindist = all_params['mindist']
        detect_interval = all_params['detect_interval']

        

        ## parameter for feature detection(original image) and tracking[for background subtraction]
        feature_params = dict( maxCorners = MaxCorns,
                          qualityLevel = qualityini,
                          minDistance = mindist1,
                          blockSize = 5 )
        ## parameter for feature tracking(original image)
        lk_params = dict( winSize  = (19, 19),
                     maxLevel = 2,
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

        ## parameter for feature detection(error image) and tracking[for feature extraction and tracking]
        print (maxcorners, quality, mindist)
        feature_params_track = dict( maxCorners = 500,
                                qualityLevel = quality/20.0,
                                minDistance = mindist,
                                blockSize = 9 )
        
        feature_params_track_oriImg = feature_params_track
                                
        lk_params_track = dict( winSize  = (19, 19),
                     maxLevel = 2,
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03),
                     minEigThreshold=1e-4)
        
        lk_params_track_ori = dict( winSize  = (25, 25),
                     maxLevel = 3,
                     flags = cv2.OPTFLOW_USE_INITIAL_FLOW,
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03),
                     minEigThreshold=1e-4)

        feature_params_Detect = dict( maxCorners = 10,
                                qualityLevel = 0.00000015,
                                minDistance = 0,
                                blockSize = 3 )

        #cam_gt=cv2.VideoCapture(videoPath+ 'shor_clip_gtVideo/uav_Video_'+videoName+'_gt.mov')
        print(videoName)
        cam=cv2.VideoCapture(videoPath+ 'Videos/Clip_'+videoName+'.mov')
        gt_text = open(videoPath+ 'Annotation_update_180925/Video_'+videoName+'_gt.txt',"r")
        f_txt = open(video_savePath_Detection+ videoName+'_dt.txt','w')
        

        
        
        # read in one frame in order to obtain the video size
        frameidx = 1
        color=cam.read()[1]
        color_gt=color.copy()#cam_gt.read()[1]
        prepreFrame = np.float32(cv2.cvtColor(color, cv2.COLOR_RGB2GRAY))
        h,w,channel = color.shape
        groundtruth = gt_text.readline()
        
        outputFeature = "time_layer: "+ str(frameidx)+" detections: "
        f_txt.write(outputFeature+"\n")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        FPS = cam.get(cv2.CAP_PROP_FPS)        
        video_PosPatch = cv2.VideoWriter(video_savePath_features+videoName+'.mov', fourcc,FPS,(w,h))         
        
            # initialize feature points
        pImg = None


        #initialize H_back
        H_back = None


    
        # read in Xt-1
        color=cam.read()[1]
        color_gt =color.copy()#cam_gt.read()[1]
        groundtruth = gt_text.readline()   
        frameidx+=1
        outputFeature = "time_layer: "+ str(frameidx)+" detections: "
        #f_txt.write(outputFeature+"\n")

        Xtminus1 = np.float32(cv2.cvtColor(color, cv2.COLOR_RGB2GRAY))

        # blocks is 1 except for the pitot tube region(0)
        blocks = np.ones((h,w), dtype='float32')
        #blocks[uplefty:downrighty,upleftx:downrightx ] = 0
        # parameter for groundtruth dilation
        dt_d = 4
        radius = 10

        Dotft = []
        Patchft=[]
        maxPatchId = 0
        
        while True:

            #print(frameidx)
            ##############Start Detection Part############
            ######Background Subtraction #################
            print('frameID:', frameidx)
            gray = Xtminus1.copy()
            # read in current frame Xt
            future_color = cam.read()[1]
            if future_color is None:
                frameidx+=1
                outputFeature = "time_layer: "+ str(frameidx)+" detections: "
                f_txt.write(outputFeature)
                break
      
            frameidx+=1

            Xt = np.float32(cv2.cvtColor(future_color, cv2.COLOR_RGB2GRAY))

            ## generate groundtruth mask
            gt_split = groundtruth.split()
            length = len(gt_split)
            gt_index = 3
            gt_mask = np.zeros_like(Xt)
            gt_ft_maske = np.zeros_like(Xt)
            bbox_index = 0
            centers = []
            
            color_gt = color.copy()
            while gt_index< length:

                bbox_index+=1
                uplefty = gt_split[gt_index]
                uplefty = int(uplefty[1:-1])
                upleftx = gt_split[gt_index+1]
                upleftx = int(upleftx[0:-1])
                downrighty = gt_split[gt_index+2]
                downrighty = int(downrighty[0:-1])
                downrightx = gt_split[gt_index+3]
                downrightx = int(downrightx[0:-2])
                
                downrighty = np.min([downrighty+dt_d, h-1])
                downrightx = np.min([downrightx+dt_d, w-1])
                uplefty = np.max([uplefty-dt_d,0])
                upleftx = np.max([upleftx-dt_d,0])
                cv2.rectangle(color_gt,(np.int16(upleftx),np.int16(uplefty)),(np.int16(downrightx),np.int16(downrighty)), (255,0,0),1)
                

                gt_mask[uplefty:downrighty, upleftx:downrightx] = bbox_index#255
                gt_ft_maske[uplefty:downrighty, upleftx:downrightx] = 255
                centers.append([(upleftx+downrightx)/2,(uplefty+downrighty)/2])
                gt_index += 4
            oriImage = color_gt.copy() 
            oriImage_1 = color_gt.copy()

            
            # extract feature points for previous frame gray = Xt-1. By using maskOut function, only keep features outside the pitot tube area
            if pImg is None or frameidx % track_len == 0:
                pImg = cv2.goodFeaturesToTrack(np.uint8(gray), **feature_params)
                pImg = maskOut(blocks, pImg)

            # compute onedirectional error Et-1 using backward transform to save computational time
            if (frameidx) % detect_interval == 0:
                weightedError,H_back,pImg = backgroundsubtraction(gray,prepreFrame, Xt,pImg,blocks,lamda, lk_params,use_ransac)
            else:
                H_back,pImg = backgroundMotion(gray,prepreFrame, Xt,pImg,blocks,lamda, lk_params, use_ransac)
            
            ###########################Part I.b Feature Extraction on Background Subtracted Image for Every Other 20 Frames###############################
            #print('Start:', len(Dotft))
            #start_time = time.time()

            if len(Dotft)>0:
                #print(frameidx, 'previous:', len(Dotft))
                d1, d, p1, pPers, p0, st1, ft_mv, ft_app, gt_labels = generatePatches_MV_trackV1(Dotft, gray, Xt, H_back, lk_params_track_ori, radius,w, h, color, gt_ft_maske)
                #print(frameidx, 'after:', len(Dotft))
                score_mv = mvmodel.predict(ft_mv, batch_size = 1000000)
                #print("--- %s seconds(deeplearning_motion) ---" % (time.time() - start_time))
                #start_time1 = time.time()
                score_app = appmodel_track.predict(ft_app,batch_size= 2560)
                #print("--- %s seconds(deeplearning_app) ---" % (time.time() - start_time1))
                
                #start_time1 = time.time()
                bifeature = np.hstack([score_app[:,0].reshape(-1,1),score_mv[:,0].reshape(-1,1)])                    
                trst = combinemodel_track.predict(bifeature)                
                #print("--- %s seconds(adaboost) ---" % (time.time() - start_time1))
                
                #start_time1 = time.time()
                #Dotft, indrm = prunddt(Dotft, Patchft)
                Dotft,indrm = dotupdate(Dotft, Patchft)
                #print("--- %s seconds(dotupdate) ---" % (time.time() - start_time1))
                oriImage = visDotft(oriImage, Dotft,w, h)
                #start_time1 = time.time()
                Dotft = dottrack_detect(Dotft, p1[indrm], pPers[indrm], trst[indrm], st1[indrm], d1[indrm], d[indrm], Patchft)#updatetr(p1, st1)
                #Dotft = dottrack_detect(Dotft, p1, pPers, trst, st1, d1, d, Patchft)#updatetr(p1, st1)
                
                oriImage = visPtV1(oriImage, p0, st1, d1)
                #print("--- %s seconds(dottrack_detect) ---" % (time.time() - start_time1))
                #start_time1 = time.time()
                
                #print("--- %s seconds(visPtV1) ---" % (time.time() - start_time1))
            #print("--- %s seconds(updateDot) ---" % (time.time() - start_time))
            #print('Midd1:', len(Dotft))
            #start_time = time.time()
            if len(Patchft)>0:
                #print('hahha')
                #print('before:', len(Patchft))
                oriImage = visDetect_Kalman(Patchft, oriImage, radius, w, h)
                #print("--- %s seconds(visDetect_Kalman) ---" % (time.time() - start_time))
                #start_time1 = time.time()
                outputFeature = writeDetect(outputFeature, radius, Patchft, w, h)
                #print("--- %s seconds(writeDetect) ---" % (time.time() - start_time1))
                #start_time2 = time.time()
                Patchft = patch_KalmanTracking(Dotft, Patchft, H_back, w, h)
                #print("--- %s seconds(patch_KalmanTracking) ---" % (time.time() - start_time2))
                
                #print('after:', len(Patchft))
            #print('Midd2:', len(Patchft))
            #print("--- %s seconds(updatePatch) ---" % (time.time() - start_time))
            #start_time = time.time()
            if (frameidx) % detect_interval == 0:
                #p_pos, p_pos_errImg, p_neg, p_neg_errImg, p_pos_gt, p_pos_gt_errImg, hit,ftNo,FAno, vis_points = generatePatches(frameidx, gray, Xt, weightedError, centers, H_back, feature_params_track, feature_params_track_oriImg, lk_params_track, radius, color, future_color, gt_mask, oriImage)
                mv, detectedPatches, errorPatches, gt_labels, detectedLocs, curLocslll, hit, ftNo, FAno = Extract_Patch(frameidx, gray, Xt, weightedError, centers, H_back, feature_params_track, feature_params_track_oriImg, lk_params_track, radius, color, future_color, gt_mask, oriImage)
                
                #errorPatches, detectedPatches, detectedLocs, Locs_next, Locs_pers, d_match = generatePatches_online(gray, Xt, weightedError, feature_params_track, color, radius,lk_params_track,H_back)
                if mv.shape[0]>0:
                    errorPatches = errorPatches[:,:,:, newaxis]
                    mv = np.hstack([mv[:,4:6],mv[:,10:]])
                    #print(detectedPatches.shape, errorPatches.shape,errorPatche_.shape)
                    data_np_test = np.concatenate([detectedPatches/255.0 ,errorPatches/255.0,errorPatches/255.0,errorPatches/255.0], axis=3)#errorPatches/255.0
                    test_output_app = appmodel.predict(data_np_test,batch_size= 2560)
                    #pred_y = np.argmax(test_output, 1)
                    
                    test_output_mv = mvmodel.predict(mv, batch_size = 1000000)
                    
                    mvmafeature = np.hstack([test_output_app[:,0].reshape(-1,1),test_output_mv[:,0].reshape(-1,1)])
                    
                    dt_lable = combinemodel.predict(mvmafeature)                   
                    
                    oriImage = visPosPatch_Kalman(dt_lable, gt_labels, detectedLocs, oriImage, radius)
                    #print('frameidx', frameidx)
                    oriImage, Dotft, Patchft, maxPatchId = DetectOnX_V2(maxD, maxPatchId, oriImage, gray, Xt, lk_params_track_ori, H_back, detectedLocs, curLocslll, dt_lable, detectedPatches,feature_params_Detect, radius, Dotft, Patchft)


            #print("--- %s seconds(newDetection) ---" % (time.time() - start_time))
            
            #strat_time = time.time()
            draw_str(oriImage, 20, 60, 'frame ID: %d' % (frameidx-1))
            video_PosPatch.write(oriImage)
            prepreFrame = Xtminus1.copy()
            color = future_color.copy()
            Xtminus1 = Xt.copy()
            future_color_gt = future_color.copy()#cam_gt.read()[1]
            f_txt.write(outputFeature+"\n")
            groundtruth = gt_text.readline()
            outputFeature = "time_layer: "+ str(frameidx)+" detections: "
            #print("--- %s seconds(writeout Results) ---" % (time.time() - start_time))
        video_PosPatch.release()
        f_txt.close()


