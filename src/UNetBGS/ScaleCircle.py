#!/usr/bin/env python
import cv2
import numpy as np
import scipy.spatial

class ScaleCircle(object):
    image=0
    def __init__(self,img):
        self.image=img
        
    def scale(self):  
        # scale parameter determination
        ecc=0.7;
        diameter_min=25
        diameter_max=1000
        distance_max=10;
        
        img=self.image;

        # resize image for processing
        sc=2000/float(img.shape[1])         
        imgres = cv2.resize(img, (int(img.shape[1]*sc), int(img.shape[0]*sc))) 
        # convert to grayscale image
        gray = cv2.cvtColor(imgres, cv2.COLOR_BGR2GRAY)
        
        # adaptive thresholding
        th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,201,10.0)
        
        # closing operator
        kernel = np.ones((5,5),np.uint8)
        closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
        
        # extract contours
        im2, contours, hierarchy = cv2.findContours(closing,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        
        # filter contours by bounding box and eccentricity
        Contdist=np.array([[0,0]])
        contlist=list()
        bboxlist=list()
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            # filter contours by bounding
            if (w>diameter_min and w<diameter_max and h>diameter_min and h<diameter_max ):
                # compute eccentricity
                (center,axes,orientation) = cv2.fitEllipse(cnt)
                majoraxis_length = max(axes)
                minoraxis_length = min(axes)        
                eccent = np.sqrt(1-(minoraxis_length/majoraxis_length)**2)
                
                # filter contours by eccentricity
                if eccent<ecc:
                    ar = np.array([[int(x+(w/2)),int(y+(h/2))]])
                    Contdist=np.concatenate((Contdist,ar))
                    contlist.append(cnt)
                    bboxlist.append([x,y,w,h])

        Contdist=Contdist[1:]
        # compute pairwise distance      
        dist_matrix = scipy.spatial.distance.pdist(Contdist)
        distsquare=scipy.spatial.distance.squareform(dist_matrix)
        np.fill_diagonal(distsquare, np.inf)
        
        # compute minimum distance
        min_positions = np.where(distsquare == distsquare.min())

        # find circles of scale symbol
        x,y,w,h = cv2.boundingRect(contlist[min_positions[0][0]])
        cent=[int(x+w/2),int(y+h/2)]
        contlistend=list()
        dialist=list()
        for cnt, b in zip(contlist, bboxlist):
            if(b[0]<cent[0] and b[1]<cent[1] and b[0]+b[2]>cent[0] and b[1]+b[3]>cent[1]):
                contlistend.append(cnt)
                d=(b[2]+b[3])/2
                dialist.append(d)
        # find circle with maximum diameter
        dmax=max(dialist)
        cnt=contlistend[dialist.index(dmax)]
        # compute scale resolution
        if distance_max>dmax:
            scale_factor=False
        else:
            d = dmax/sc;
            scale_factor=25.86*(d/198);
        if scale_factor<10 or scale_factor>150:
            scale_factor=False
            
        #cv2.drawContours(closing,cnt,-1,(0,255,0),cv2.cv.CV_FILLED)       
        #cv2.namedWindow('win1',cv2.cv.CV_NORMAL)
        #cv2.imshow('win1',closing*255)
        #cv2.waitKey()
        #print('scale_factor: ' + str(scale_factor))
        return scale_factor
        
        
#I=cv2.imread('/media/bernifoellmer/1TBDisk/Masterarbeit/Images/PCI/SAM_7067.JPG')   
#sc=ScaleCircle(I).scale()  
