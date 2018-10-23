# TrainModel.py
from __future__ import print_function, division, absolute_import, unicode_literals

import cv2
import numpy as np
from enum import Enum
from BackgroundDetector import BackgroundDetector, BGClass
from unet.image_util import BaseDataProvider
import math
import random

from random import randint

#from TrainModel import DLModel


class BGDataCreatorMethod(Enum): 
    RANDOMSELECTION = 0

    
class BGDataGenerator(object): 
    name = 'CNN'     
    BGD = BackgroundDetector()
    method = BGDataCreatorMethod.RANDOMSELECTION
    data_train = ([],[])
    data_test = ([],[])
    data_valid = ([],[])

    def __init__(self, name):
        self.name = name
        
    def loadBGModel(self, filepath):
        self.BGD.read_zipdb(filepath)
        
    def histogram_equalize(self, img):
        b, g, r = cv2.split(img)
        red = cv2.equalizeHist(r)
        green = cv2.equalizeHist(g)
        blue = cv2.equalizeHist(b)
        return cv2.merge((blue, green, red))
        
    def createData(self, N_train, N_test, N_valid, N_img_train, N_img_test, N_img_valid, nx, ny, N_img):
        
        if self.method == BGDataCreatorMethod.RANDOMSELECTION:
            
            IdxImg = random.sample(list(range(0, N_img)), len(list(range(0, N_img))))
            IdxTrain = IdxImg[0:N_img_train]
            IdxTest = IdxImg[N_img_train:N_img_train+N_img_test]
            IdxValid = IdxImg[N_img_train+N_img_test:N_img_train+N_img_test+N_img_valid]
            
            Images=[]
            # Get images ans create image masks
            for i in range(N_img):
                imd = self.BGD.Imagelist[i]
                im = self.histogram_equalize(imd.image)
                Images.append(im)
                self.BGD.RegionsList[i] = self.BGD.createRegions(self.BGD.RegionsMap, i)
            
            dims_img = Images[0].shape

            # Create masks       
            maskList=[]           
            for i in range(N_img):
                print('i:', i)
                regions = self.BGD.RegionsList[i]
                dims = Images[i].shape
                mask = np.zeros((dims[0], dims[1]), np.uint16)
                for j, reg in enumerate(regions):                   
                    if self.BGD.RegionsClass[i][j]==BGClass.BACKGROUND:
                        #print('j:', j)
                        #print('reg shape:', reg.shape)
                        image_reg = reg
                        image_reg_res = cv2.resize(image_reg, dsize=(dims[1], dims[0]))
                        
                        #print('image_reg_res shape:', image_reg_res.shape)
                        
                        th, image_reg_thr = cv2.threshold(image_reg_res, 127, 255, cv2.THRESH_BINARY)
                        
                        
                        
                        #print('image_reg_thr shape:', image_reg_thr.shape)
                        #print('mask shape:', mask.shape)
                        
                        #cv2.namedWindow( "image_reg_thr", cv2.WINDOW_NORMAL )
                        #cv2.imshow('image_reg_thr', image_reg_thr*255)
                        #cv2.waitKey(0)
                        
                        image_reg_thr = image_reg_thr.astype(np.uint16)
                        
                        mask = mask + image_reg_thr
                thr, mask_thr = cv2.threshold(mask, 150, 1, cv2.THRESH_BINARY)
                maskList.append(mask_thr)

            # Create random crops    
            # Select random midpoint
            nx2 = math.ceil(nx/2.0)
            ny2 = math.ceil(ny/2.0)
            bx1 = nx2 + 1
            bx2 = dims_img[0] - nx2 - 1
            by1 = ny2 + 1
            by2 = dims_img[1] - ny2 - 1

            # Create train data
            for i in range(N_train):
                #print('N_img_train', N_img_train)
                #print('IdxTrain', IdxTrain)
                n = IdxTrain[randint(0, N_img_train-1)]
                #print('n_train', n)
                x = randint(bx1, bx2)
                y = randint(by1, by2)
                im = Images[n]
                crop_img = im[x-nx2:x+nx2, y-ny2:y+ny2]
                self.data_train[0].append(crop_img)                
                mask = maskList[n]
                crop_mask = mask[x-nx2:x+nx2, y-ny2:y+ny2]
                self.data_train[1].append(crop_mask)
                
#                print('crop_mask shape', crop_mask.shape)
#                print('crop_mask type', crop_mask.dtype)
#                
#                m=crop_mask*255
#                m=m.astype(np.uint8)
#                cv2.namedWindow("crop_img", cv2.WINDOW_NORMAL)
#                cv2.imshow('crop_img', crop_img) 
#                cv2.namedWindow("crop_mask", cv2.WINDOW_NORMAL)
#                cv2.imshow('crop_mask', m) 
#                cv2.waitKey(0)
            
            # Create test data
            for i in range(N_test):
                
                n = IdxTest[randint(0, N_img_test-1)]
                #print('n_test', n)
                x = randint(bx1, bx2)
                y = randint(by1, by2)
                im = Images[n]
                crop_img = im[x-nx2:x+nx2, y-ny2:y+ny2]
                self.data_test[0].append(crop_img)
                
                mask = maskList[n]
                crop_mask = mask[x-nx2:x+nx2, y-ny2:y+ny2]
                self.data_test[1].append(crop_mask)
                
            # Create valid data
            for i in range(N_valid):
                
                n = IdxValid[randint(0, N_img_valid-1)]
                #print('n_valid', n)
                x = randint(bx1, bx2)
                y = randint(by1, by2)
                im = Images[n]
                crop_img = im[x-nx2:x+nx2, y-ny2:y+ny2]
                self.data_valid[0].append(crop_img)
                
                mask = maskList[n]
                crop_mask = mask[x-nx2:x+nx2, y-ny2:y+ny2]
                self.data_valid[1].append(crop_mask)
            

class BGDataProvider(BaseDataProvider):
    channels = 3
    n_class = 2
    BGDataGen = BGDataGenerator('BGDataGenerator')
    
    NumSample_train = 0
    NumSample_test = 0
    NumSample_valid = 0
    
    def __init__(self, nx, ny, filepath, loadData=True):
        super(BGDataProvider, self).__init__()
        self.nx = nx
        self.ny = ny
        #self.kwargs = kwargs
        
        if loadData:
            self.BGDataGen.loadBGModel(filepath)
            N_train = 2500
            N_test = 100
            N_valid = 1500
            
            N_img = 6
            N_img_train = 3
            N_img_test = 1
            N_img_valid = 2
            self.BGDataGen.createData(N_train, N_test, N_valid, N_img_train, N_img_test, N_img_valid, nx, ny, N_img)


    def _next_data(self, dataset='train'):
        data, label = self.create_image_and_label(self.nx, self.ny, dataset)
        return data, label

    def create_image_and_label(self, nx, ny, dataset='train'):
        
        print('dataset: ', dataset)
        
        if dataset == 'train':
            image = self.BGDataGen.data_train[0][self.NumSample_train]
            label = self.BGDataGen.data_train[1][self.NumSample_train]
            self.NumSample_train = self.NumSample_train + 1    
            label = label.astype(bool)
            
        if dataset == 'test':
            image = self.BGDataGen.data_test[0][self.NumSample_test]
            label = self.BGDataGen.data_test[1][self.NumSample_test]
            self.NumSample_test = self.NumSample_test + 1    
            label = label.astype(bool)
            
        if dataset == 'valid':
            image = self.BGDataGen.data_valid[0][self.NumSample_valid]
            label = self.BGDataGen.data_valid[1][self.NumSample_valid]
            self.NumSample_valid = self.NumSample_valid + 1    
            label = label.astype(bool)
        
        #print('label5 shape', label.shape)
        #print('label5 shape', label.shape)
        
        return image, label


def to_rgb(img):
    img = img.reshape(img.shape[0], img.shape[1])
    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    blue = np.clip(4*(0.75-img), 0, 1)
    red  = np.clip(4*(img-0.25), 0, 1)
    green= np.clip(44*np.fabs(img-0.5)-1., 0, 1)
    rgb = np.stack((red, green, blue), axis=2)
    return rgb
     
if __name__ == '__main__':
    
    BGGenerator = BGDataGenerator('BGDataGenerator01')
    BGGenerator.loadBGModel('H:/Projects/SearchPartPython/SearchPartPython/SearchPart/data/background/BG6.zip')
    N_train = 3
    N_test = 3
    N_valid = 3
    nx = 572 
    ny = 572 
    nimg = 2
    BGGenerator.createData(N_train, N_test, N_valid, nx, ny, nimg)
    
    model = DLModel()
    nx = 572
    ny = 572
    model.init(nx, ny)
    model.train()
    model.test()