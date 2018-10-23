# UNetBGS - UNet for PCB surface segmentation

## Overview

UNetBGS is a deep learning network (U-Net) to segment electronic parts from PCB boards.
PCB background segmentation is as sub project of the SearchPartPython project [SearchPartPython](SearchPartPython), solved with a deep learning network. 
The code was developed based on https://github.com/jakeret/tf_unet
The architecture was inspired by U-Net: Convolutional Networks for Biomedical Image Segmentation.

## Code
https://github.com/Berni1557/UNetBGS

## Data set

The dataset was created from different PCB boards. All PCB board are used and therefore show signs of wear.
Region growing was used to localize regions with identical color in the image.
Regions corresponding to backgroud are labled manually by clicking in the corresponding PCB region.

Download dataset of colored PCB boards (colored PCB surface): 
Download dataset of black PCB boards (black PCB surface): 

## Predicted image example

Original PCB image             |  Predicted PCB background
:-------------------------:|:-------------------------:
![image01_org](image01_org.PNG)  |  ![image01_predict](image01_predict.PNG)

## Results
It could be shown taht the segmentation with a UNet does segment PCB surface very successfull.

UNetBGS
A detailed evaluation is not possible yet, due to the inaccurate labeled PCB backgroung.

## Further development

## GitHub
[UNetBGS-Webside](http://www.bernifoellmer.com/wordpress/unetbgs/)

## Developers

Bernhard Föllmer, berniweb@posteo.de
