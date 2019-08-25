# Opencv 2---Version
# -*- coding: utf-8 -*-
'''
    kalman2d - 2D Kalman filter using OpenCV
    
    Based on http://jayrambhia.wordpress.com/2012/07/26/kalman-filter/
    
    Copyright (C) 2014 Simon D. Levy
    
    This code is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.
    This code is distributed in the hope that it will be useful,
    
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU Lesser General Public License
    along with this code. If not, see <http://www.gnu.org/licenses/>.
    '''

#from cv2 import cv
import cv2
import numpy as np

class Kalman2D(object):
    '''
        A class for 2D Kalman filtering
        '''
    
    def __init__(self, processNoiseCovariance=1e-4, measurementNoiseCovariance=1e-1, errorCovariancePost=0.1):
    #def __init__(self,processNoiseCovariance=1e-1, measurementNoiseCovariance=1e1, errorCovariancePost=1e4):
        '''
            Constructs a new Kalman2D object.
            For explanation of the error covariances see
            http://en.wikipedia.org/wiki/Kalman_filter
            '''
        # state space：location--2d,speed--2d
        #self.kalman = cv.CreateKalman(4, 2, 0)
        self.kalman = cv2.KalmanFilter(4, 2, 0)
        self.kalman_measurement = np.array([[1.],[1.]],np.float32)
        
        self.kalman.transitionMatrix = np.array([[1.,0., 1.,0.], [0., 1., 0., 1.], [0., 0., 1., 0.], [0., 0., 0., 1.]],np.float32)
        

        
        self.kalman.measurementMatrix = np.array([[1.,0.,0.,0.],[0.,1.,0.,0.]],np.float32)
        
        self.kalman.processNoiseCov = processNoiseCovariance * np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]],np.float32)
        self.kalman.measurementNoiseCov = np.array([[1.,0.],[0.,1.]],np.float32) * measurementNoiseCovariance
        self.kalman.errorCovPost = np.array([[1.,0., 0, 0],[0.,1., 0, 0],[0.,0, 1, 0],[0.,0, 0, 1]],np.float32) * errorCovariancePost
        #cv.SetIdentity(self.kalman.measurement_matrix)
        #Initialize identity matrix
        #cv.SetIdentity(self.kalman.process_noise_cov, cv.RealScalar(processNoiseCovariance))
        #cv.SetIdentity(self.kalman.measurement_noise_cov, cv.RealScalar(measurementNoiseCovariance))
        #cv.SetIdentity(self.kalman.error_cov_post, cv.RealScalar(errorCovariancePost))
        
        self.predicted = np.array((2,1), np.float32)
        self.corrected = np.zeros((2,1), np.float32)

    def update(self, x, y):
        '''
            Updates the filter with a new X,Y measurement
            '''
        self.kalman_measurement = np.array([[np.float32(x)],[np.float32(y)]])
        #self.kalman_measurement[0, 0] = x
        #self.kalman_measurement[1, 0] = y
        #print self.kalman.predict()
        self.predicted = self.kalman.predict()
        self.corrected = self.kalman.correct(self.kalman_measurement)
    #self.corrected = cv.KalmanCorrect(self.kalman, self.kalman_measurement)

    def getEstimate(self):
        '''
            Returns the current X,Y estimate.
            '''
            
        return self.corrected[0,0], self.corrected[1,0]
        
    def getPrediction(self):
        '''
            Returns the current X,Y prediction.
            '''
        
        return self.predicted[0,0], self.predicted[1,0]
