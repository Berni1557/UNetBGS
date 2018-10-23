# main.py

# Import libraries
from __future__ import print_function, division, absolute_import, unicode_literals
import sys
import os
import importlib
import imp
import cv2
import numpy as np

fileDir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(fileDir + '/unet')
import UNetBGS
imp.reload(UNetBGS)
importlib.reload(UNetBGS)

# main function
if __name__ == '__main__':
    
    # Create model
    model = UNetBGS.UNetBGS()
    nx_input = 572
    ny_input = 572
    nx_output = 532
    ny_output = 532
    loadData=True
    #filepath_model = 'H:/Projects/SearchPartPython/SearchPartPython/SearchPart/data/background/BGBlack.zip'
    filepath_model = 'H:/Projects/SearchPartPython/SearchPartPython/SearchPart/data/background/BG01.zip'
    model.init(nx_input, ny_input, filepath_model, loadData)
    model.train()
    model.test()
    
    image = cv2.imread('H:/Projects/SearchPartPython/SearchPartPython/SearchPart/data/background/BGBlack/SAM_0625.JPG')
    image = cv2.equalizeHist(image)
    image_pred = model.predict(image, nx_input, ny_input, nx_output, ny_output)
    
    
    
    
    
    # Test image
    image = cv2.imread('H:/Projects/SearchPartPython/SearchPartPython/SearchPart/data/background/SAM_0614.JPG')
    image_pred = model.predict(image, nx_input, ny_input, nx_output, ny_output)
    
    cv2.imwrite('tmp.png', image_pred)
 
    cv2.namedWindow("Prediction", cv2.WINDOW_NORMAL)
    cv2.imshow('Prediction', image_pred) 
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow('Image', image) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    # Test image subsets
#    for i in range(10):
#        model.test()
#        imagedata = model.prediction
#        image=imagedata[0,:,:,1]*255
#        image8 = image.astype(np.uint8)
#        cv2.namedWindow("Prediction", cv2.WINDOW_NORMAL)
#        cv2.imshow('Prediction', image8) 
#        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
#        cv2.imshow('Image', model.prediction_img) 
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()

