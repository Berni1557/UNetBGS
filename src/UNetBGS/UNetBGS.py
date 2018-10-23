# BGDLModel.py
import os
import sys
#fileDir = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(fileDir + '\\unet')

sys.path.append('H:/Projects/SearchPartPython/SearchPartPython/SearchPart/src/SearchPart/unet')

import cv2
import numpy as np
from BGDataGenerator import BGDataProvider
from unet import unet
#from util import to_rgb

import importlib
importlib.reload(unet)
#importlib.reload(util)

import matplotlib.pyplot as plt
import timeit

class UNetBGS(object): 
    """
    Model for background segmentation
    """
    modelname = 'CNN'     
    generator = None
    path = None
    net = None
    prediction = None
    prediction_img = None
    testvalue = 3
    
    def init(self, nx_input, ny_input, filepath, loadData=True):
        
        if loadData:
            self.generator = BGDataProvider(nx_input, ny_input, filepath, loadData)       
            x_test, y_test = self.generator(1)
            
            # Create image plot properties
            fig, ax = plt.subplots(1,2, sharey=True, figsize=(8,4))
            ax[0].imshow(x_test[0,...,0], aspect="auto")
            ax[1].imshow(y_test[0,...,1], aspect="auto")
        
        self.net = unet.Unet(channels=self.generator.channels, n_class=self.generator.n_class, layers=3, features_root=16) 
    
    def train(self):
        #self.net = unet.Unet(channels=self.generator.channels, n_class=self.generator.n_class, layers=3, features_root=16)       
        trainer = unet.Trainer(self.net, batch_size=3, optimizer="momentum", opt_kwargs=dict(momentum=0.2, learning_rate = 0.05))       
        self.path = trainer.train(self.generator, "./unet_trained", training_iters=20, epochs=40, display_step=2)
        #self.path = trainer.train(self.generator, "./unet_trained", training_iters=50, epochs=10, display_step=2)
        
    def test(self):
        x_test, y_test = self.generator(1,'test') 
        print('x_test', x_test.shape)
        self.prediction = self.net.predict("./unet_trained/model.cpkt", x_test)
        self.prediction_img = x_test[0,:,:,:]
        
    def predict(self, image, nx_in, ny_in, nx_out, ny_out):

        width_crop = nx_out
        height_crop = ny_out
        width_sel = nx_in
        height_sel = ny_in
        crops_struct = self.cropImage(image, width_crop, height_crop, width_sel, height_sel)  
        crops=crops_struct[0]
        
        dims = image.shape
        image_width = dims[1]
        image_height = dims[0]
        
        #crops_small=[]
        #for c in crops:
        #    crops_small.append(c[:, 20:20+width_crop, 20:20+height_crop, :])
        
#        n = len(crops)
#        #n=30
#        im = np.zeros((n, height_sel, width_sel, dims[2]), np.uint8)
#        for i, c in enumerate(crops):
#            if i<n:
#                im[i,:,:,:] = c
#            
#        print('im shape', im.shape)
#        
#        start = timeit.timeit()
#        self.prediction = self.net.predict("./unet_trained/model.cpkt", im)  
#        end = timeit.timeit()
#        print('Time: ', end - start)
#
#        crops_pred=[]
#        for i, c in enumerate(crops):
#            if i<n:
#                imagedata = self.prediction[i,:,:,:]
#                image=imagedata[:,:,1]*255
#                image8 = image.astype(np.uint8)
#                crops_pred.append(image8)
            
        crops_pred=[]
        for im in crops:
            self.prediction = self.net.predict("./unet_trained/model.cpkt", im)
            imagedata = self.prediction
            image=imagedata[0,:,:,1]*255
            image8 = image.astype(np.uint8)
            crops_pred.append(image8)
    
        crops_struct_pred = (crops_pred, crops_struct[1], crops_struct[2])
        image_pred = self.stichImage(crops_struct_pred, image_width, image_height) 
        return image_pred
    
    def cropImage(self, image, width_crop, height_crop, width_sel, height_sel):
        
        dw=width_sel - width_crop
        dh=height_sel - height_crop
    
        dims = image.shape
        
        w_rest = dims[1] % width_crop
        h_rest = dims[0] % height_crop
        w1 = dims[1] - w_rest + width_crop
        h1 = dims[0] - h_rest + height_crop
        nw = int(w1 / width_crop)
        nh = int(h1 / height_crop)
        
        w2 = w1 + dw
        h2 = h1 + dh
        
        image_crop = np.zeros((1, h2, w2, dims[2]), np.uint8)
        x1 = int(dh/2)
        x2 = int((dh/2)+dims[0])
        y1 = int(dw/2)
        y2 = int((dw/2)+dims[1])
        image_crop[0,x1:x2, y1:y2,:] = image
        
        Crops = []
        CropsPosX = []
        CropsPosY = []

        for i in range(nw):
            for j in range(nh):
                y11 = j*height_crop + dh/2
                y12 = (j+1)*height_crop + dh/2
                x11 = i*width_crop + dw/2
                x12 = (i+1)*width_crop + dw/2
                
                y21 = int(y11 - dh/2)
                y22 = int(y12 + dh/2)
                x21 = int(x11 - dw/2)
                x22 = int(x12 + dw/2)
                crop = image_crop[:, y21:y22, x21:x22, :]
                Crops.append(crop)
                CropsPosX.append(i)
                CropsPosY.append(j)
        crops_struct=(Crops, CropsPosX, CropsPosY)
        return crops_struct
    
    
    def stichImage(self, crops_struct, image_width, image_height):
        
        Crops = crops_struct[0]
        CropsPosX = crops_struct[1]
        CropsPosY = crops_struct[2]
        
        x_max = max(CropsPosX)
        y_max = max(CropsPosY)
        
        #dw=width_sel - width_crop
        #dh=height_sel - height_crop
    
        dims = Crops[0].shape
        
        width_crop=dims[0]
        height_crop=dims[1]
        
        nw = int(x_max+1)
        nh = int(y_max+1)
        
        w = nw * dims[0]
        h = nh * dims[1]
        
        image_stitch = np.zeros((h, w), np.uint8)

        k=0
        for i in range(nw):
            for j in range(nh):
                y1 = j*height_crop
                y2 = (j+1)*height_crop
                x1 = i*width_crop
                x2 = (i+1)*width_crop

                image_stitch[y1:y2, x1:x2] = Crops[k]
                k=k+1
        image = image_stitch[0:image_height, 0:image_width] 
        return image
        
        #cv2.namedWindow("Prediction", cv2.WINDOW_NORMAL)
        #cv2.imshow('Prediction', image_concat) 
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        
        #x_test, y_test = self.generator(1,'test')        
        #self.prediction = self.net.predict("./unet_trained/model.cpkt", x_test)
        
if __name__ == '__main__':
    
    #from TrainModel import DLModel
    model = UNetBGS()
    image = cv2.imread('H:/Projects/SearchPartPython/SearchPartPython/SearchPart/data/background/SAM_0703.JPG')
    width_crop = 532
    height_crop = 532
    width_sel = 572
    height_sel = 572
    crops_struct = model.cropImage(image, width_crop, height_crop, width_sel, height_sel)  
    crops=crops_struct[0]
    
    image_width = 4000
    image_height = 3000
    
    crops_small=[]
    for c in crops:
        crops_small.append(c[:, 20:20+width_crop, 20:20+height_crop, :])
    

    crops_struct_small = (crops_small, crops_struct[1], crops_struct[2])
    image_out = model.stichImage(crops_struct_small, image_width, image_height) 
    

    cv2.namedWindow("Prediction", cv2.WINDOW_NORMAL)
    cv2.imshow('Prediction', image_out) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()      