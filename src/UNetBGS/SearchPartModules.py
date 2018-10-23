# Searchpart library

#from gi.repository import Gtk
#from gi.repository import GdkPixbuf
import cv2
import numpy as np
from xml.dom.minidom import *
import sys
import time
import os
import zipfile
import shutil
import ScaleCircle as SC
from scipy import ndimage
import math
#from mlabwrap import mlab


class imagecounter(object): 
    imagenumber = -1     
    imagenumber_max = -1;
    
    def __init__(self, imagenumber = -1, imagenumber_max = -1):
        self.imagenumber = imagenumber;
        self.imagenumber_max = imagenumber_max;
        
    def tostring(self):
        if self.imagenumber_max>=0:
            imstr='Imagenumber: ' + str(self.imagenumber+1) + " / " + str(self.imagenumber_max+1)
        else:
            imstr="Imagenumber: 0 / 0"
        return imstr
    
    def valid(self):
        return self.imagenumber <= self.imagenumber_max and self.imagenumber_max>-1
    
class OCRdata(object):
    #OCRrotation = [False,False,False,False]
    OCR = False
    OCRlib = False
    charsubset = ''
    OCRborder_Top = 0
    OCRborder_Right = 0
    OCRborder_Bottom = 0
    OCRborder_Left = 0
    OCRText = ''
    OCROnlineDataBase = ''
    OCRAxialSymmetricHorizontal = True
    OCRAxialSymmetricVertical = True
    
class Component(object):
    Creation_date=''
    Componentname=''
    ComponentID=0
    Componentheight=10
    Componentwidth=20
    Componentborder=0
    #Componentrotation=[False,False,False,False]
    Componentdescription=''
    CompOCRdata=OCRdata()
    Imagename=list()
    Imagelist=list()
    dom=''
    Componentmean=0
    Decisionmodel=None
    AxialSymmetricHorizontal = True
    AxialSymmetricVertical = True
    
    def __init__(self, parent, filename,):
        # create component from file
        #if isinstance (filename, basestring):
        if(False):
            self.parent=parent
            self=read_zipdb(self, filename)
            if (parent!=None):
                self.parent.imagecounter.imagenumber=0;
                self.parent.imagecounter.imagenumber_max=len(self.Imagelist)-1;    
        # create new component                                             
        else:
            self.parent=parent
            self.Creation_date=time.strftime("%c")
            self.Componentname=''
            self.ComponentDatasetPath=''
            self.ComponentID=0
            #self.path=''
            self.Imagename=list()
            self.Top=list()
            self.Bottom=list()
            self.Left=list()
            self.Right=list()
            #self.Componentrotation=[False,False,False,False]
            self.AxialSymmetricHorizontal = True
            self.AxialSymmetricVertical = True
    
    def create_mean(self):
        #self.Componentmean=0
        sz=self.scale_corr()
        size=[sz[0],sz[1],3]
        k=0
        Componentmean=np.zeros(size, dtype=np.float)
        for Im in self.Imagelist:
            #imagesc = cv2.resize(Im.image, (self.parent.imsize[0], self.parent.imsize[1])) 
            for b in Im.Top:
                Imcomp = Im.image[int(b[1]):int(b[1]+b[3]), int(b[0]):int(b[0]+b[2])]
                Compcorr = cv2.resize(Imcomp,(size[1],size[0]))
                Componentmean=Componentmean+Compcorr
                k+=1
            for b in Im.Right:
                Imcomp = Im.image[int(b[1]):int(b[1]+b[3]), int(b[0]):int(b[0]+b[2])]
                Compcorr = ndimage.rotate(Imcomp, 90)
                Compcorr = cv2.resize(Compcorr,(size[1],size[0]))
                Componentmean=Componentmean+Compcorr
                k+=1
            for b in Im.Bottom:
                Imcomp = Im.image[int(b[1]):int(b[1]+b[3]), int(b[0]):int(b[0]+b[2])]
                Compcorr = ndimage.rotate(Imcomp, 180)
                Compcorr = cv2.resize(Compcorr,(size[1],size[0]))
                Componentmean=Componentmean+Compcorr
                k+=1
            for b in Im.Left:
                Imcomp = Im.image[int(b[1]):int(b[1]+b[3]), int(b[0]):int(b[0]+b[2])]
                Compcorr = ndimage.rotate(Imcomp, -90)
                Compcorr = cv2.resize(Compcorr,(size[1],size[0]))
                Componentmean=Componentmean+Compcorr
                k+=1

        if k>0:
            Componentmean=Componentmean/k
            self.Componentmean = np.array(Componentmean, dtype = np.uint8)
        #self.Imagelist[0].image=self.Componentmean 
        return self.Componentmean 
    

    def corr(self):
        Componentmean=self.create_mean()
        sc=self.scale_corr()
        sc_corr=sc[2]

        thr = 0.4
        n_max=1
        k=0
        for Im in self.Imagelist:
            k=k+1
            print('Image ' + str(k) + '/' + str(len(self.Imagelist)))
            del Im.Topcorr[:]
            del Im.Rightcorr[:]
            del Im.Bottomcorr[:]
            del Im.Leftcorr[:]
            
            #scale image
            scale=sc_corr/Im.scale_factor
            imagesc = cv2.resize(Im.image, (int(Im.image.shape[1]*scale), int(Im.image.shape[0]*scale)))
            
            # Top correlation
            h=sc[0];hl=int(h/2);hr=h-hl
            w=sc[1];wl=int(w/2);wr=w-wl
            # create template
            Compcorr = Componentmean
            n=0
            maxVal=1000
            
            # RGB to HSV
            imagehsv=cv2.cvtColor(imagesc, cv2.cv.CV_BGR2HSV);
            Comphsv=cv2.cvtColor(Compcorr, cv2.cv.CV_BGR2HSV);
            # correlation
            corr = cv2.matchTemplate(imagehsv,Comphsv,cv2.TM_CCOEFF_NORMED)
            
            while maxVal>thr and n<n_max:
                # determine maximum
                (minVal,maxVal,minLoc,maxLoc) = cv2.minMaxLoc(corr)
                x=maxLoc[0];y=maxLoc[1]
                if (x-wl>0 and y-hl>0 and x+wr<corr.shape[1] and y+hr<corr.shape[0]):
                    # remove correlation maximum
                    roi=np.zeros([h,w],np.uint8)
                    corr[y-hl:y+hr,x-wl:x+wr]=roi
                    #create rectangle
                    b=[x,y,w,h]
                    borg = [i / scale for i in b]
                    Im.Topcorr.append(borg)  
                else:
                    corr[y,x]=0
                n=n+1  
                
            # Right correlation
            h=sc[1];hl=int(h/2);hr=h-hl
            w=sc[0];wl=int(w/2);wr=w-wl
            # create template
            Compcorr = ndimage.rotate(Componentmean, -90)
            n=0
            maxVal=1000
            # RGB to HSV
            Comphsv=cv2.cvtColor(Compcorr, cv2.cv.CV_BGR2HSV);
            # correlation
            corr = cv2.matchTemplate(imagehsv,Compcorr,cv2.TM_CCOEFF_NORMED)
            while maxVal>thr and n<n_max:
                # determine maximum
                (minVal,maxVal,minLoc,maxLoc) = cv2.minMaxLoc(corr)
                x=maxLoc[0];y=maxLoc[1]
                if (x-wl>0 and y-hl>0 and x+wr<corr.shape[1] and y+hr<corr.shape[0]):
                    # remove correlation maximum
                    roi=np.zeros([h,w],np.uint8)
                    corr[y-hl:y+hr,x-wl:x+wr]=roi
                    #create rectangle
                    b=[x,y,w,h]
                    borg = [i / scale for i in b]
                    Im.Rightcorr.append(borg)  
                else:
                    corr[y,x]=0
                n=n+1  
                
            # Bottom correlation
            h=sc[0];hl=int(h/2);hr=h-hl
            w=sc[1];wl=int(w/2);wr=w-wl
            # create template
            Compcorr = ndimage.rotate(Componentmean, 180)
            n=0
            maxVal=1000
            # RGB to HSV
            Comphsv=cv2.cvtColor(Compcorr, cv2.cv.CV_BGR2HSV);
            # correlation
            corr = cv2.matchTemplate(imagehsv,Compcorr,cv2.TM_CCOEFF_NORMED)
            while maxVal>thr and n<n_max:
                # determine maximum
                (minVal,maxVal,minLoc,maxLoc) = cv2.minMaxLoc(corr)
                x=maxLoc[0];y=maxLoc[1]
                if (x-wl>0 and y-hl>0 and x+wr<corr.shape[1] and y+hr<corr.shape[0]):
                    # remove correlation maximum
                    roi=np.zeros([h,w],np.uint8)
                    corr[y-hl:y+hr,x-wl:x+wr]=roi
                    #create rectangle
                    b=[x,y,w,h]
                    borg = [i / scale for i in b]
                    Im.Bottomcorr.append(borg)  
                else:
                    corr[y,x]=0
                n=n+1  
                
            # Left correlation
            h=sc[1];hl=int(h/2);hr=h-hl
            w=sc[0];wl=int(w/2);wr=w-wl
            # create template
            Compcorr = ndimage.rotate(Componentmean, 90)
            n=0
            maxVal=1000
            # RGB to HSV
            Comphsv=cv2.cvtColor(Compcorr, cv2.cv.CV_BGR2HSV);
            # correlation
            corr = cv2.matchTemplate(imagehsv,Compcorr,cv2.TM_CCOEFF_NORMED)
            while maxVal>thr and n<n_max:
                # determine maximum
                (minVal,maxVal,minLoc,maxLoc) = cv2.minMaxLoc(corr)
                x=maxLoc[0];y=maxLoc[1]
                if (x-wl>0 and y-hl>0 and x+wr<corr.shape[1] and y+hr<corr.shape[0]):
                    # remove correlation maximum
                    roi=np.zeros([h,w],np.uint8)
                    corr[y-hl:y+hr,x-wl:x+wr]=roi
                    #create rectangle
                    b=[x,y,w,h]
                    borg = [i / scale for i in b]
                    Im.Leftcorr.append(borg)  
                else:
                    corr[y,x]=0
                n=n+1                     
        self.parent.update_componentdata()
              
    def scale_corr(self):
        x=float(self.Componentwidth)*float(self.Componenthight)
        res=15*math.exp( -( x -1)*0.05 )+5;
        return [int(float(self.Componenthight)*res),int(float(self.Componentwidth)*res),res]
          
          
class Imagedata(object):
    improbabilitymap=list()
    image=np.zeros((3000,4000,3), np.uint8)
    def __init__(self,path):
        self.Imagepath=path
        self.Imagepath_relative=os.path.basename(path)
        self.Imagename=os.path.basename(path)
        self.image=cv2.imread(path)    
        self.Top=list()
        self.Right=list()
        self.Bottom=list()
        self.Left=list()
        
        self.Topcorr=list()
        self.Rightcorr=list()
        self.Bottomcorr=list()
        self.Leftcorr=list()

        Circle=SC.ScaleCircle(self.image)
        sc=Circle.scale()
        self.scale_factor=sc
        
        #self.scale_factor=sc[0][0]

    def corr(self,Compscale):
        sc=Compscale/self.scale_factor;
        dst=self.image.copy()
        cv2.resize(self.image, dst, 0, 0.5, 0.5, cv2.INTER_NEAREST);
        
        #Icorr=cv2.resize(self.image)
        
def convertCV2GTK(image):
    image_gtk = Gtk.Image()
    src = cv2.cv.fromarray(image)
    cv2.cv.SaveImage('/home/bernifoellmer/workspace/SearchPartPython_V01/tempimage.png', src)
    image_gtk.set_from_file("/home/bernifoellmer/workspace/SearchPartPython_V01/tempimage.png")
    return image_gtk


def convertCV2GTK_v01(image,image_gtk):
    
    height, width, depth = image.shape
    imtemp = np.zeros((height, width, depth), np.uint8) 
    
    src = cv2.cv.fromarray(image)
    dst = cv2.cv.fromarray(imtemp)
    
    cv2.cv.CvtColor(src, dst, cv2.cv.CV_RGB2BGR)
    #dst=src;
    
    #image_gtk = Gtk.Image()
    img_pixbuf = GdkPixbuf.Pixbuf.new_from_data(dst.tostring(),GdkPixbuf.Colorspace.RGB,False,8,width,height,width*depth) 
    image_gtk.set_from_pixbuf(img_pixbuf)


def Imagecopy(imageorg,imagecopy):    
    p=imageorg.get_pixbuf()   
    imagecopy.set_from_pixbuf(p)

def create_dom(Component):
    
    dom = Document();
    base = dom.createElement('datasetstruct')
    dom.appendChild(base)
    
    node1 = dom.createElement('Creation_date')
    text1 = dom.createTextNode(str(Component.Creation_date))
    node1.appendChild(text1)
    dom.childNodes[0].appendChild(node1)
    
    node1 = dom.createElement('Componentname')
    text1 = dom.createTextNode(str(Component.Componentname))
    node1.appendChild(text1)
    dom.childNodes[0].appendChild(node1)    

    node1 = dom.createElement('ComponentID')
    text1 = dom.createTextNode(str(Component.ComponentID))
    node1.appendChild(text1)
    dom.childNodes[0].appendChild(node1)    
        
    node1 = dom.createElement('Componentheight')
    text1 = dom.createTextNode(str(Component.Componentheight))
    node1.appendChild(text1)
    dom.childNodes[0].appendChild(node1)
    
    node1 = dom.createElement('Componentwidth')
    text1 = dom.createTextNode(str(Component.Componentwidth))
    node1.appendChild(text1)
    dom.childNodes[0].appendChild(node1)  
    
    node1 = dom.createElement('Componentborder')
    text1 = dom.createTextNode(str(Component.Componentborder))
    node1.appendChild(text1)
    dom.childNodes[0].appendChild(node1)       
    
    #node1 = dom.createElement('Componentrotation')
    #text1 = dom.createTextNode(str(Component.Componentrotation))
    #node1.appendChild(text1)
    #dom.childNodes[0].appendChild(node1)   
    
    node1 = dom.createElement('AxialSymmetricHorizontal')
    text1 = dom.createTextNode(str(Component.AxialSymmetricHorizontal))
    node1.appendChild(text1)
    dom.childNodes[0].appendChild(node1)       

    node1 = dom.createElement('AxialSymmetricVertical')
    text1 = dom.createTextNode(str(Component.AxialSymmetricVertical))
    node1.appendChild(text1)
    dom.childNodes[0].appendChild(node1)          

    node1 = dom.createElement('Componentdescription')
    text1 = dom.createTextNode(str(Component.Componentdescription))
    node1.appendChild(text1)
    dom.childNodes[0].appendChild(node1)  

    node1 = dom.createElement('CompOCRdata')
    
    #node2 = dom.createElement("OCRrotation")
    #text2 = dom.createTextNode(str(Component.CompOCRdata.OCRrotation))
    #node2.appendChild(text2)
    #node1.appendChild(node2)
    
    node2 = dom.createElement("OCRborder")
    text2 = dom.createTextNode(str([Component.CompOCRdata.OCRborder_Top,Component.CompOCRdata.OCRborder_Right,Component.CompOCRdata.OCRborder_Bottom,Component.CompOCRdata.OCRborder_Left]))
    node2.appendChild(text2)
    node1.appendChild(node2)
    
    
    node2 = dom.createElement("OCR")
    text2 = dom.createTextNode(str(Component.CompOCRdata.OCR))
    node2.appendChild(text2)
    node1.appendChild(node2)
    node2 = dom.createElement("OCRlib")
    text2 = dom.createTextNode(str(Component.CompOCRdata.OCRlib))
    node2.appendChild(text2)
    node1.appendChild(node2)
    node2 = dom.createElement("charsubset")
    text2 = dom.createTextNode(str(Component.CompOCRdata.charsubset))
    node2.appendChild(text2)
    node1.appendChild(node2)
    
    node2 = dom.createElement("OCRText")
    text2 = dom.createTextNode(str(Component.CompOCRdata.OCRText))
    node2.appendChild(text2)
    node1.appendChild(node2)
    
    node2 = dom.createElement("OCROnlineDataBase")
    text2 = dom.createTextNode(str(Component.CompOCRdata.OCROnlineDataBase))
    node2.appendChild(text2)
    node1.appendChild(node2)
        
    node2 = dom.createElement("OCRAxialSymmetricHorizontal")
    text2 = dom.createTextNode(str(Component.CompOCRdata.OCRAxialSymmetricHorizontal))
    node2.appendChild(text2)
    node1.appendChild(node2)
    
    node2 = dom.createElement("OCRAxialSymmetricVertical")
    text2 = dom.createTextNode(str(Component.CompOCRdata.OCRAxialSymmetricVertical))
    node2.appendChild(text2)
    node1.appendChild(node2)
    
    dom.childNodes[0].appendChild(node1)      
        
    for Im in Component.Imagelist:
        node1 = dom.createElement('Image')
        
        node2 = dom.createElement("Imagename")
        text2 = dom.createTextNode(Im.Imagename)
        node2.appendChild(text2)
        node1.appendChild(node2)

        node2 = dom.createElement("Imagepath_relative")
        #text2 = dom.createTextNode(Im.Imagepath)
        text2 = dom.createTextNode(Im.Imagepath_relative)      
        node2.appendChild(text2)
        node1.appendChild(node2)
        
        node2 = dom.createElement("Scale_factor")
        text2 = dom.createTextNode(str(Im.scale_factor))
        node2.appendChild(text2)
        node1.appendChild(node2)
                
        node2 = dom.createElement("Top")
        for c in Im.Top:
            node3 = dom.createElement("item")
            text3 = dom.createTextNode(str(c))
            node3.appendChild(text3)
            node2.appendChild(node3)
            node1.appendChild(node2)
            
        node2 = dom.createElement("Right")
        for c in Im.Right:
            node3 = dom.createElement("item")
            text3 = dom.createTextNode(str(c))
            node3.appendChild(text3)
            node2.appendChild(node3)
            node1.appendChild(node2)

        node2 = dom.createElement("Bottom")
        for c in Im.Bottom:
            node3 = dom.createElement("item")
            text3 = dom.createTextNode(str(c))
            node3.appendChild(text3)
            node2.appendChild(node3)
            node1.appendChild(node2)

        node2 = dom.createElement("Left")
        for c in Im.Left:
            node3 = dom.createElement("item")
            text3 = dom.createTextNode(str(c))
            node3.appendChild(text3)
            node2.appendChild(node3)
            node1.appendChild(node2)   
        dom.childNodes[0].appendChild(node1)                                   
        
               
    return dom

#class ProgressBarWindow(Gtk.Window):
#
#    def __init__(self):
#        #Gtk.Window.__init__(self, title="ProgressBar Demo")
#        self.set_window(Gtk.WindowType.POPUP)
#        self.set_border_width(20)
#
#        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
#        self.add(vbox)
#
#        self.progressbar = Gtk.ProgressBar()
#        vbox.pack_start(self.progressbar, True, True, 0)
#        self.activity_mode = False
#
#    def set_value(self, new_value):
#        self.progressbar.set_fraction(new_value)
#        
#    def add_value(self, value):
#        new_value=value + self.progressbar.get_fraction()
#        self.progressbar.set_fraction(new_value)
#        



def write_zipdb(Component, filepath_ext):
    
    filepath, file_extension = os.path.splitext(filepath_ext)    
    print('filepath: ' + filepath)
    
    # Create  dom
    dom=create_dom(Component)
    st=dom.toprettyxml()
    
    #Create xml file
    filepathxml=filepath + '.xml'

    f = open(filepathxml, 'w')
    f.write(st)
    f.close()

    # Create zip path
    zipname=filepath + '.zip'
    zipf = zipfile.ZipFile(zipname, 'w')

    # Change to filepathxml folder and add xml-file to zip-file
    di, base_filename = os.path.split(filepathxml)
    os.chdir(di)
    zipf.write(base_filename)
    
    # Remove zip-file
    os.remove(filepathxml)
    
    # Add imagefiles to zipImagepath
    for Im in Component.Imagelist:
        di, base_filename = os.path.split(Im.Imagepath)
        os.chdir(di)
        zipf.write(base_filename) 
    zipf.close()
    
def read_zipdb(Component, filepath, StatusLine):
    
    #print('filepath1: ' + filepath)
    
    base_folder, base_filename = os.path.split(filepath)
    
    
    str1=base_filename.split('.')
    filename = str1[0]
    xmlname=str1[0] + '.xml'
    zipf = zipfile.ZipFile(filepath, 'r')
    zipf.extract(xmlname,base_folder)
    zipf.extractall(base_folder + '/' + filename)
    
    str1=base_filename.split('.')
    
    xmlpath=base_folder + '/'+ xmlname
    
    #print('xmlpath: ' + xmlpath)
    
    #print(xmlpath)
    dom = parse(xmlpath)
    Component.dom = dom
    os.remove(xmlpath)
    
    st=dom.toprettyxml()
    #print(st)
    
    
    Component.Creation_date=Component.dom.getElementsByTagName('Creation_date').item(0).firstChild.nodeValue
    
    if Component.dom.getElementsByTagName('Componentname').item(0).childNodes:
        Component.Componentname=Component.dom.getElementsByTagName('Componentname').item(0).firstChild.nodeValue
    if Component.dom.getElementsByTagName('ComponentID').item(0).childNodes:
        Component.ComponentID=Component.dom.getElementsByTagName('ComponentID').item(0).firstChild.nodeValue
    if Component.dom.getElementsByTagName('Componentheight').item(0).childNodes:
        Component.Componentheight=float(Component.dom.getElementsByTagName('Componentheight').item(0).firstChild.nodeValue)
    if Component.dom.getElementsByTagName('Componentwidth').item(0).childNodes:
        Component.Componentwidth=float(Component.dom.getElementsByTagName('Componentwidth').item(0).firstChild.nodeValue)
    if Component.dom.getElementsByTagName('Componentborder').item(0).childNodes:
        Component.Componentborder=Component.dom.getElementsByTagName('Componentborder').item(0).firstChild.nodeValue
        
    
    
    #s=Component.dom.getElementsByTagName('Componentrotation').item(0).firstChild.nodeValue
    #s1=s.split('[')
    #s2=s1[1].split(']')
    #str1=s2[0].split(',')
    #b=list()
    #for i in str1:
    #    i=i.replace(" ", "")
    #    b.append(str_to_bool(i))
    #Component.Componentrotation=b
    if Component.dom.getElementsByTagName('AxialSymmetricHorizontal').item(0).childNodes:
        Component.AxialSymmetricHorizontal=str_to_bool(Component.dom.getElementsByTagName('AxialSymmetricHorizontal').item(0).firstChild.nodeValue)
    
    if Component.dom.getElementsByTagName('AxialSymmetricVertical').item(0).childNodes:
        Component.AxialSymmetricVertical=str_to_bool(Component.dom.getElementsByTagName('AxialSymmetricVertical').item(0).firstChild.nodeValue)
    
    
    
    if Component.dom.getElementsByTagName('Componentdescription').item(0).childNodes:
        Component.Componentdescription=Component.dom.getElementsByTagName('Componentdescription').item(0).firstChild.nodeValue

    if Component.dom.getElementsByTagName('OCR').item(0).childNodes:
        s=Component.dom.getElementsByTagName('OCR').item(0).firstChild.nodeValue
        Component.CompOCRdata.OCR=str_to_bool(s)
    if Component.dom.getElementsByTagName('OCRlib').item(0).childNodes:
        s=Component.dom.getElementsByTagName('OCRlib').item(0).firstChild.nodeValue
        Component.CompOCRdata.OCRlib=str_to_bool(s)
    if Component.dom.getElementsByTagName('charsubset').item(0).childNodes:
        s=Component.dom.getElementsByTagName('charsubset').item(0).childNodes[0].nodeValue
        Component.CompOCRdata.charsubset=s
    if Component.dom.getElementsByTagName('OCRText').item(0).childNodes:
        s=Component.dom.getElementsByTagName('OCRText').item(0).childNodes[0].nodeValue
        Component.CompOCRdata.OCRText=s
    if Component.dom.getElementsByTagName('OCROnlineDataBase').item(0).childNodes:
        s=Component.dom.getElementsByTagName('OCROnlineDataBase').item(0).childNodes[0].nodeValue
        Component.CompOCRdata.OCROnlineDataBase=s
        
    if Component.dom.getElementsByTagName('OCRAxialSymmetricHorizontal').item(0).childNodes:
        Component.CompOCRdata.OCRAxialSymmetricHorizontal=str_to_bool(Component.dom.getElementsByTagName('OCRAxialSymmetricHorizontal').item(0).childNodes[0].nodeValue)
        
    if Component.dom.getElementsByTagName('OCRAxialSymmetricVertical').item(0).childNodes:
        Component.CompOCRdata.OCRAxialSymmetricVertical=str_to_bool(Component.dom.getElementsByTagName('OCRAxialSymmetricVertical').item(0).childNodes[0].nodeValue)
      
    # get OCRrotation
    #s=Component.dom.getElementsByTagName('OCRrotation').item(0).firstChild.nodeValue
    #s1=s.split('[')
    #s2=s1[1].split(']')
    #str1=s2[0].split(',')
    #b=list()
    #for i in str1:
    #    i=i.replace(" ", "")
    #    b.append(str_to_bool(i))
    #Component.CompOCRdata.OCRrotation=b
    # get OCRborder
    s=Component.dom.getElementsByTagName('OCRborder').item(0).firstChild.nodeValue
    s1=s.split('[')
    s2=s1[1].split(']')
    str1=s2[0].split(',')
    b=list()
    for i in str1:
        i=i.replace(" ", "")
        b.append(float(i))
    Component.CompOCRdata.OCRborder_Top=b[0]
    Component.CompOCRdata.OCRborder_Right=b[1]
    Component.CompOCRdata.OCRborder_Bottom=b[2]
    Component.CompOCRdata.OCRborder_Left=b[3]
        
    images=Component.dom.getElementsByTagName('Image')
    for image in images:
       
        node=image.getElementsByTagName('Imagepath_relative')
        Imagepath_relative=node[0].childNodes[0].nodeValue
        
        #print(Imagepath)
        Imagepath = base_folder + '/' + filename + '/' + Imagepath_relative
        #print('Imagepath: ' + Imagepath)
        Im=Imagedata(Imagepath)
        
        node=image.getElementsByTagName('Imagename')
        Im.Imagename=node[0].childNodes[0].nodeValue
        
        StatusLine.append('reading image: ' + Im.Imagename)
        print('reading image: ' + Im.Imagename);
        
        n1=image.getElementsByTagName('Top')
        for n2 in n1:
            it=n2.getElementsByTagName('item')
            for i in it:
                s=i.childNodes[0].nodeValue
                s1=s.split('[')
                s2=s1[1].split(']')
                str1=s2[0].split(',')
                b=list()
                for i in str1:
                    b.append(float(i))
                Im.Top.append(b)


        n1=image.getElementsByTagName('Right')
        for n2 in n1:
            it=n2.getElementsByTagName('item')
            for i in it:
                s=i.childNodes[0].nodeValue
                s1=s.split('[')
                s2=s1[1].split(']')
                str1=s2[0].split(',')
                b=list()
                for i in str1:
                    b.append(float(i))
                Im.Right.append(b)

        n1=image.getElementsByTagName('Bottom')
        for n2 in n1:
            it=n2.getElementsByTagName('item')
            for i in it:
                s=i.childNodes[0].nodeValue
                s1=s.split('[')
                s2=s1[1].split(']')
                str1=s2[0].split(',')
                b=list()
                for i in str1:
                    b.append(float(i))
                Im.Bottom.append(b)   
    
        n1=image.getElementsByTagName('Left')
        for n2 in n1:
            it=n2.getElementsByTagName('item')
            for i in it:
                s=i.childNodes[0].nodeValue
                s1=s.split('[')
                s2=s1[1].split(']')
                str1=s2[0].split(',')
                b=list()
                for i in str1:
                    b.append(float(i))
                Im.Left.append(b)
                      
        Component.Imagelist.append(Im)
        Component.Imagename.append(Im.Imagename)
    return Component

#   
#class DSclass(object):
#    
#    selectbbox=False
#    selectOCRborder=False
#    deletebbox=False
#    compmean=False
#    addimages=False
#    deleteimage=False
#    #DSComponent=Component(None,None)
#    scale=0.2;
#    imsize=[int(4000*scale),int(3000*scale)]
#    bboxrot=[True,False,False,False]
#    
#    def __init__(self):
#        #self.builder = Gtk.Builder()
#        #self.builder.add_from_file("/home/bernifoellmer/Studium/SearchPartPython/SearchPartPython/SearchPartPython/SearchPartPython/glade/SearchPartGlade.glade")
#        #self.builder.connect_signals(Handler(self))
#        
##
##        self.window = self.builder.get_object("window1")
##        self.windowbox = self.builder.get_object("windowbox")
##        self.height = self.builder.get_object("height")
##        self.width = self.builder.get_object("width")
##        self.CompID = self.builder.get_object("CompID")
##        self.Compborder = self.builder.get_object("Compborder")
##        self.Compname = self.builder.get_object("Compname")
##        self.Comppath = self.builder.get_object("Comppath")
##        
##        self.Comp_top = self.builder.get_object("Comp_top")
##        self.Comp_right = self.builder.get_object("Comp_right")
##        self.Comp_bottom = self.builder.get_object("Comp_bottom")
##        self.Comp_left = self.builder.get_object("Comp_left")
##        
##        self.OCR = self.builder.get_object("OCR")
##        self.Octopart = self.builder.get_object("Octopart")
##        
##        self.OCR_top = self.builder.get_object("OCR_top")
##        self.OCR_right = self.builder.get_object("OCR_right")
##        self.OCR_bottom = self.builder.get_object("OCR_bottom")
##        self.OCR_left = self.builder.get_object("OCR_left")
##        
##        self.OCRborder_Top = self.builder.get_object("OCRborder_Top")
##        self.OCRborder_Right = self.builder.get_object("OCRborder_Right")
##        self.OCRborder_Bottom = self.builder.get_object("OCRborder_Bottom")
##        self.OCRborder_Left = self.builder.get_object("OCRborder_Left")
##        
##        self.charsubset = self.builder.get_object("charsubset")
##        self.Compdescription = self.builder.get_object("Compdescription")
##        self.selectbbox = self.builder.get_object("selectbbox")
##        self.selectOCRborder = self.builder.get_object("selectOCRborder")
##        
##        self.Imscale = self.builder.get_object("Imscale")
##        self.Imnumber = self.builder.get_object("Imnumber")
##        
##        self.Imageback = self.builder.get_object("Imageback")
##        self.Imagenext = self.builder.get_object("Imagenext")
##        
##        self.drawarea=self.builder.get_object("drawingarea")
##        
##        self.progressbar=self.builder.get_object("progressbar")
##        self.progressbar.set_fraction(0.5)
##        
##        self.DSComponent=SPM.Component(self,None)
##
##        self.imagecounter=imagecounter();
##        self.window.set_size_request(250, 150)
##        self.window.show_all()
#        self.reset()
#
#    def update_componentdata(self):
#        self.imagecounter.imagenumber_max=len(self.DSComponent.Imagelist)-1
#        
#        if(self.imagecounter.imagenumber>self.imagecounter.imagenumber_max):
#            self.imagecounter.imagenumber=self.imagecounter.imagenumber_max
#            
#        if(self.imagecounter.imagenumber<0):
#            self.imagecounter.imagenumber=0
#        
#        self.height.set_text(str(self.DSComponent.Componenthight))
#        self.width.set_text(str(self.DSComponent.Componentwidth))
#        self.CompID.set_text(str(self.DSComponent.ComponentID))
#        self.Compborder.set_text(str(self.DSComponent.Componentborder))
#        self.Compname.set_text(str(self.DSComponent.Componentname))
#        
#        self.Comp_top.set_active(self.DSComponent.Componentrotation[0])
#        self.Comp_right.set_active(self.DSComponent.Componentrotation[1])
#        self.Comp_bottom.set_active(self.DSComponent.Componentrotation[2])
#        self.Comp_left.set_active(self.DSComponent.Componentrotation[3])
#        
#        self.OCR_top.set_active(self.DSComponent.CompOCRdata.OCRrotation[0])
#        self.OCR_right.set_active(self.DSComponent.CompOCRdata.OCRrotation[1])
#        self.OCR_bottom.set_active(self.DSComponent.CompOCRdata.OCRrotation[2])
#        self.OCR_left.set_active(self.DSComponent.CompOCRdata.OCRrotation[3])
#        
#        self.Octopart.set_active(self.DSComponent.CompOCRdata.OCRlib)
#        self.OCR.set_active(self.DSComponent.CompOCRdata.OCR)
#        self.Octopart.set_active(self.DSComponent.CompOCRdata.OCRlib)
#
#        self.OCRborder_Top.set_text(str(self.DSComponent.CompOCRdata.OCRborder_Top))
#        self.OCRborder_Right.set_text(str(self.DSComponent.CompOCRdata.OCRborder_Right))
#        self.OCRborder_Bottom.set_text(str(self.DSComponent.CompOCRdata.OCRborder_Bottom))
#        self.OCRborder_Left.set_text(str(self.DSComponent.CompOCRdata.OCRborder_Left))
#                
#        self.charsubset.set_text(self.DSComponent.CompOCRdata.charsubset)
#        self.Compdescription.set_text(self.DSComponent.Componentdescription)
#        
#        
#        self.Imnumber.set_label(self.imagecounter.tostring()) 
#
#        if len(self.DSComponent.Imagelist)>0:
#            self.Imscale.set_label(str(self.DSComponent.Imagelist[self.imagecounter.imagenumber].scale_factor) + ' [p/mm]') 
#        self.drawarea.queue_draw()
#
#    def reset(self):
#        height=self.builder.get_object('height')
#        height.set_text('0')
#        width=self.builder.get_object('width')
#        width.set_text('0')            
#        CompID=self.builder.get_object('CompID')
#        CompID.set_text('0')        
#        Compborder=self.builder.get_object('Compborder')
#        Compborder.set_text('0') 
#        Compname=self.builder.get_object('Compname')
#        Compname.set_text('')        
#        #Comppath=self.builder.get_object('Comppath')
#        #Comppath.set_text('') 
#                
#        Comp_top=self.builder.get_object('Comp_top')
#        Comp_top.set_active(False)        
#        Comp_right=self.builder.get_object('Comp_right')
#        Comp_right.set_active(False)  
#        Comp_bottom=self.builder.get_object('Comp_bottom')
#        Comp_bottom.set_active(False)  
#        Comp_left=self.builder.get_object('Comp_left')
#        Comp_left.set_active(False)   
#
#        OCRborder_Top=self.builder.get_object('OCRborder_Top')
#        OCRborder_Top.set_text('Top')  
#        OCRborder_Right=self.builder.get_object('OCRborder_Right')
#        OCRborder_Right.set_text('Right')  
#        OCRborder_Bottom=self.builder.get_object('OCRborder_Bottom')
#        OCRborder_Bottom.set_text('Bottom')  
#        OCRborder_Left=self.builder.get_object('OCRborder_Left')
#        OCRborder_Left.set_text('Left')     
#
#        OCR_top=self.builder.get_object('OCR_top')
#        OCR_top.set_active(False)        
#        OCR_right=self.builder.get_object('OCR_right')
#        OCR_right.set_active(False)  
#        OCR_bottom=self.builder.get_object('OCR_bottom')
#        OCR_bottom.set_active(False)  
#        OCR_left=self.builder.get_object('OCR_left')
#        OCR_left.set_active(False)      
#        charsubset=self.builder.get_object('charsubset')
#        charsubset.set_text('ABCDEFGHIJKLMONOPQRSTUVWXYZ123456789/')   
#
#        Compdescription=self.builder.get_object('Compdescription')
#        Compdescription.set_text('')  
#        
#        selectbbox=self.builder.get_object('selectbbox')
#        selectbbox.set_active(False) 
#        
#        selectOCRborder=self.builder.get_object('selectOCRborder')
#        selectOCRborder.set_active(False) 
#        
#        Imscale=self.builder.get_object('Imscale')
#        Imscale.set_label('0.0 [p/mm]')   
#        
#        Imnumber=self.builder.get_object('Imnumber')
#        Imnumber.set_label(self.imagecounter.tostring()) 
#        
#        self.DSComponent=SPM.Component(self,None)


def str_to_bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError # evil ValueError that doesn't tell you what the wrong value was

ERROR_INVALID_NAME = 123
def is_pathname_valid(pathname: str) -> bool:
    '''
    `True` if the passed pathname is a valid pathname for the current OS;
    `False` otherwise.
    '''
    # If this pathname is either not a string or is but is empty, this pathname
    # is invalid.
    try:
        if not isinstance(pathname, str) or not pathname:
            return False

        # Strip this pathname's Windows-specific drive specifier (e.g., `C:\`)
        # if any. Since Windows prohibits path components from containing `:`
        # characters, failing to strip this `:`-suffixed prefix would
        # erroneously invalidate all valid absolute Windows pathnames.
        _, pathname = os.path.splitdrive(pathname)

        # Directory guaranteed to exist. If the current OS is Windows, this is
        # the drive to which Windows was installed (e.g., the "%HOMEDRIVE%"
        # environment variable); else, the typical root directory.
        root_dirname = os.environ.get('HOMEDRIVE', 'C:') \
            if sys.platform == 'win32' else os.path.sep
        assert os.path.isdir(root_dirname)   # ...Murphy and her ironclad Law

        # Append a path separator to this directory if needed.
        root_dirname = root_dirname.rstrip(os.path.sep) + os.path.sep

        # Test whether each path component split from this pathname is valid or
        # not, ignoring non-existent and non-readable path components.
        for pathname_part in pathname.split(os.path.sep):
            try:
                os.lstat(root_dirname + pathname_part)
            # If an OS-specific exception is raised, its error code
            # indicates whether this pathname is valid or not. Unless this
            # is the case, this exception implies an ignorable kernel or
            # filesystem complaint (e.g., path not found or inaccessible).
            #
            # Only the following exceptions indicate invalid pathnames:
            #
            # * Instances of the Windows-specific "WindowsError" class
            #   defining the "winerror" attribute whose value is
            #   "ERROR_INVALID_NAME". Under Windows, "winerror" is more
            #   fine-grained and hence useful than the generic "errno"
            #   attribute. When a too-long pathname is passed, for example,
            #   "errno" is "ENOENT" (i.e., no such file or directory) rather
            #   than "ENAMETOOLONG" (i.e., file name too long).
            # * Instances of the cross-platform "OSError" class defining the
            #   generic "errno" attribute whose value is either:
            #   * Under most POSIX-compatible OSes, "ENAMETOOLONG".
            #   * Under some edge-case OSes (e.g., SunOS, *BSD), "ERANGE".
            except OSError as exc:
                if hasattr(exc, 'winerror'):
                    if exc.winerror == ERROR_INVALID_NAME:
                        return False
                elif exc.errno in {errno.ENAMETOOLONG, errno.ERANGE}:
                    return False
    # If a "TypeError" exception was raised, it almost certainly has the
    # error message "embedded NUL character" indicating an invalid pathname.
    except TypeError as exc:
        return False
    # If no exception was raised, all path components and hence this
    # pathname itself are valid. (Praise be to the curmudgeonly python.)
    else:
        return True
    # If any other exception was raised, this is an unrelated fatal issue
    # (e.g., a bug). Permit this exception to unwind the call stack.
    #
    # Did we mention this should be shipped with Python already?
                
def is_path_creatable(pathname: str) -> bool:
    '''
    `True` if the current user has sufficient permissions to create the passed
    pathname; `False` otherwise.
    '''
    # Parent directory of the passed path. If empty, we substitute the current
    # working directory (CWD) instead.
    dirname = os.path.dirname(pathname) or os.getcwd()
    return os.access(dirname, os.W_OK)

def is_path_exists_or_creatable(pathname):
    '''
    `True` if the passed pathname is a valid pathname for the current OS _and_
    either currently exists or is hypothetically creatable; `False` otherwise.

    This function is guaranteed to _never_ raise exceptions.
    '''
    try:
        # To prevent "os" module calls from raising undesirable exceptions on
        # invalid pathnames, is_pathname_valid() is explicitly called first.
        return is_pathname_valid(pathname) and (
            os.path.exists(pathname) or is_path_creatable(pathname))
    # Report failure on non-fatal filesystem complaints (e.g., connection
    # timeouts, permissions issues) implying this path to be inaccessible. All
    # other exceptions are unrelated fatal issues and should not be caught here.
    except OSError:
        return False
