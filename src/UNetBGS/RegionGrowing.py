import cv2
import numpy as np
from random import randint
import sys
from scipy.misc import imresize

class RegionGrowing(object): 
    
    # RegionGrowing members
    m_threshold = 5
    m_reg_size_min = 1000
    m_show = True
    m_scale = 1.0
    
    def __init__(self, threshold=5, reg_size_min=100, show = True, scale = 1.0):
        self.m_threshold = threshold
        self.m_reg_size_min = reg_size_min
        self.m_show = show
        self.m_scale = scale
        
    def drawContours(self, regions, show = True, scale = 1.0):
        """
        Draw contours
        """
        
        regions_res = imresize(regions[0][:,:,0], scale)
        dims = regions_res.shape
        image_cont = np.zeros((dims[0],dims[1],1), np.uint8)
        image_region_cont = []
        if show:
            cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)
        for im in regions:
            im_res = imresize(im[:,:,0], self.m_scale)
            _, contours, hierarchy = cv2.findContours(im_res.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(image_cont, contours, -1, 255, 1); 
            
            region_cont = np.zeros((dims[0],dims[1],1), np.uint8)
            cv2.drawContours(region_cont, contours, -1, 255, 1);
            image_region_cont.append(region_cont)
        if show:
            cv2.imshow('Contours', image_cont) 
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return image_cont, image_region_cont
        
    def simple_region_growing(self, img, seed, threshold, mask):
        """
        A implementation of region growing.
        """  
        try:
            #dims = cv2.GetSize(img)
            dims = img.shape
        except TypeError:
            raise TypeError("(%s) img : IplImage expected!" % (sys._getframe().f_code.co_name))
    
        #print('dims ',dims)
        # img test
        if not(img.dtype == np.uint8):
            raise TypeError("(%s) 8U image expected!" % (sys._getframe().f_code.co_name))
        #elif not(img.nChannels is 1):
        #    raise TypeError("(%s) 1C image expected!" % (sys._getframe().f_code.co_name))
        # threshold tests
        if (not isinstance(threshold, int)) :
            raise TypeError("(%s) Int expected!" % (sys._getframe().f_code.co_name))
        elif threshold < 0:
            raise ValueError("(%s) Positive value expected!" % (sys._getframe().f_code.co_name))
        # seed tests
        if not((isinstance(seed, tuple)) and (len(seed) is 2) ) :
            raise TypeError("(%s) (x, y) variable expected!" % (sys._getframe().f_code.co_name))
    
        if (seed[0] or seed[1] ) < 0 :
            raise ValueError("(%s) Seed should have positive values!" % (sys._getframe().f_code.co_name))
        elif ((seed[0] > dims[0]) or (seed[1] > dims[1])):
            print('seed err: ', seed)
            raise ValueError("(%s) Seed values greater than img size!" % (sys._getframe().f_code.co_name))
    
        #reg = cv2.CreateImage( dims, cv2.IPL_DEPTH_8U, 1)
        reg = np.zeros((dims[0], dims[1], 1), np.uint8)
        #cv2.Zero(reg)
    
        #parameters
        mean_reg = img[seed[0], seed[1]].astype(float)
       
        size = 0
        pix_area = dims[0]*dims[1]
    
        contour = [] # will be [ [[x1, y1], val1],..., [[xn, yn], valn] ]
        contour_val = []
        dist = 0
        # TODO: may be enhanced later with 8th connectivity
        orient = [(1, 0), (0, 1), (-1, 0), (0, -1)] # 4 connectivity
        cur_pix = [seed[0], seed[1]]
        #Spreading
        n=0
        HasNeighbors = True
        while(dist<threshold and size<pix_area and HasNeighbors):
            n += 1
        #adding pixels
            for j in range(4):
                #select new candidate
                temp_pix = [cur_pix[0] +orient[j][0], cur_pix[1] +orient[j][1]]
    
                #check if it belongs to the image
                is_in_img = dims[0]>temp_pix[0]>0 and dims[1]>temp_pix[1]>0 #returns boolean
                #candidate is taken if not already selected before
                if (is_in_img and (reg[temp_pix[0], temp_pix[1]]==0) and (mask[temp_pix[0], temp_pix[1]]==0)):
                    contour.append(temp_pix)
                    pix = img[temp_pix[0], temp_pix[1]]
                    contour_val.append(pix.astype(float))
                    reg[temp_pix[0], temp_pix[1]] = 255
                    mask[temp_pix[0], temp_pix[1]] = 255
                    
                    
            if len(contour_val) > 0:
                #add the nearest pixel of the contour in it
                dist3 = abs(int(np.mean(contour_val)) - mean_reg)
                dist = dist3.mean()
        
                dist_list3 = [abs(i - mean_reg) for i in contour_val ]
                dist_list = [i.mean() for i in dist_list3 ]
                dist = min(dist_list)    #get min distance
                index = dist_list.index(min(dist_list)) #mean distance index
                
                reg[cur_pix[0], cur_pix[1]] = 255
                mask[cur_pix[0], cur_pix[1]] = 255
        
                #updating mean MUST BE FLOAT
                size += 1 # updating region size
                if size==1:
                     mean_reg = contour_val[index]
                else:
                     m = ((size-1)/size) * mean_reg
                     p = (1/size) * contour_val[index]
                     mean_reg = m.astype(float) + p.astype(float)                
        
                #updating seed
                cur_pix = contour[index]
        
                #removing pixel from neigborhood
                del contour[index]
                del contour_val[index]

            else:
                HasNeighbors = False
                reg[cur_pix[0], cur_pix[1]] = 255
                mask[cur_pix[0], cur_pix[1]] = 255
    
        return reg, mask
    
    def region_growing(self, image):
        """
        An implementation of region growing.
        """
        
        print('region_growing scale: ', self.m_scale)
        
        image = imresize(image, self.m_scale)
        reg_size = self.m_reg_size_min        
        dims = image.shape
        mask = np.zeros((dims[0],dims[1],1), np.uint8)
        #mask_seed = np.zeros((dims[0],dims[1],1), np.uint8)
        
        print('dims: ', dims)
        
        if self.m_show:
            cv2.namedWindow( "mask", cv2.WINDOW_NORMAL )
            cv2.namedWindow( "reg", cv2.WINDOW_NORMAL )
        
        #hierarchy_all = np.array([])
        regions = [];
        
        while reg_size >= self.m_reg_size_min:
            
            # Create mask inverse
            mask_invert = 255-mask
            
            # Extract biggest contout and draw in mask_seed
            _, contours, hierarchy = cv2.findContours(mask_invert, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            area_max=0;
            for c in contours:
                area = cv2.contourArea(c)
                if area > area_max:
                    area_max = area
                    c_max = c
            
            contours_draw=[];
            contours_draw.append(c_max);
            mask_seed = np.zeros((dims[0], dims[1], 1), np.uint8)
            cv2.drawContours(mask_seed, contours_draw, 0, 255, cv2.FILLED);        
            
            # Extract random seed
            IndexSeed = np.where(np.equal(mask_seed, 255))
            Index = np.where(np.equal(mask, 0))
            
            pos = randint(0, len(Index[0])-1)
            #seed = (Index[1][pos], Index[0][pos])
            seed = (Index[0][pos], Index[1][pos])
            
            # Do region growing
            reg, mask = self.simple_region_growing(image, seed, self.m_threshold, mask)
            
            # Update reg_size
            #reg_size = len(Index[0])
            reg_size = len(IndexSeed[0])
            
            if self.m_show:
                cv2.imshow('mask', mask)
                cv2.imshow('reg', reg)
                cv2.waitKey(10)

            regions.append(reg)
        
        if self.m_show:
            cv2.destroyAllWindows()
            
        regionsMap = np.zeros((dims[0],dims[1],1), np.uint16)
        
        
        print('regions len1', len(regions))
        
        for i in range(1, len(regions)):
            im = (i+1)*(regions[i] / 255)
            regionsMap = regionsMap + im.astype(np.uint16)
        
        
        
        print('regionsMap max1', np.amax(regionsMap))
        
        biggest = np.amax(regionsMap)
        print('biggest: ', biggest)
        return regions, regionsMap
                   
