import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

class Loader:

    def __init__(self, dataset :str, datapath :str, bin_num :int =256, use_gradient :bool =False, use_sharpness :bool = True, use_luminance :bool =True):
        self.datapath=datapath

        if(not self.datapath.endswith('/')):
            self.datapath=self.datapath+'/'

        self.datasets=['MWD','ACDC','UAVid','syndrone']
        self.dataset=dataset
        self.bin_num=bin_num

        if(self.bin_num <= 0 or bin_num > 256):
            raise Exception('Invalid number of bins {}'.format(bin_num))

        if(dataset not in self.datasets):
            raise Exception('Wrong dataset provided, chose from MWD,ACDC,UAVid,syndrone')

        if dataset == self.datasets[0]:
            self.classes=['cloudy','rain','shine','sunrise']
            self.candidates=['cloudy','rain','shine','sunrise']
        elif dataset == self.datasets[1]:
            self.classes=['clear','fog','night','rain','snow']
            self.candidates=['clear','fog','night','rain','snow']
        elif dataset == self.datasets[2]:
            self.classes=['clear','fog','night','rain']
            self.candidates=['day','fog','night','rain']
        elif dataset == self.datasets[3]:
            self.classes=['clear','fog','night','rain']
            self.candidates=['ClearNoon','MidFoggyNoon','ClearNight','HardRainNoon']


        self.use_gradient=use_gradient
        self.use_sharpness=use_sharpness
        self.use_luminance=use_luminance



    def get_classes(self):
        return self.classes


    def get_class(self,image_name :str):
        

        for i in range(len(self.candidates)):
            if(self.candidates[i] in image_name):
                return self.classes[i]

        return None






    def load_images(self):
        if self.dataset == 'MWD':
            return self.load_mwd()
        else:
            return self.load_other()









#TBD: da personalizzare i vari metodi per una corretta classificazione.




    def load_mwd(self):
        X=np.array([])
        y=np.array([])
        
        for cls in self.classes:
            
            tmp_path=self.datapath+cls
            files_list = os.listdir(tmp_path)
            for file in files_list:
                #print(file)
                image=cv.imread(tmp_path+'/'+file)
                if(image is not None) :
                    #COMPUTE FEATURES AND APPEND DATA TO X and y
                    #cv.imshow('test0',image)
                    #cv.waitKey(1)
                    image_feat=self.get_features(image)
                    #print(image_feat)
                    #print(image_feat.shape)
                    image_cls=self.get_class(file)
                    #print(image_cls)


                    y=np.append(y,image_cls)

                    if(len(X)==0):
                        X=image_feat
                    else:
                        X=np.vstack((X,image_feat))

                else:
                    print('Found a non image file {} at: {}'.format(file,tmp_path))
                    
        return X,y


    def load_other(self):
        X=np.array([])
        y=np.array([])
        path=self.datapath
        files_list = os.listdir(path)
        for file in files_list:
            #print(file)
            image=cv.imread(path+'/'+file)
            if(image is not None) :
                #COMPUTE FEATURES AND APPEND DATA TO X and y 
                #cv.imshow('test0',image)
                #cv.waitKey(1)
                image_feat=self.get_features(image)
                #print(image_feat)
                #print(image_feat.shape)
                image_cls=self.get_class(file)
                #print(image_cls)



                y=np.append(y,image_cls)

                if(len(X)==0):
                    X=image_feat
                else:
                    X=np.vstack((X,image_feat))
                
            else:
                print('Found a non image file {} at: {}'.format(file,path))
            
                    
        return X,y



    def get_features(self,image :np.ndarray):
        feat=self.compute_histograms(image)
        if(self.use_sharpness):
            feat=np.append(feat,self.compute_sharpness(image))
        if(self.use_gradient):
            feat=np.append(feat,self.compute_gradient(image))
        if(self.use_luminance):
            feat=np.append(feat,self.compute_luminance(image))
        return feat

    def compute_histograms(self,image: np.ndarray):
        
        if image is None:
            raise Exception('Invalid None image')

        image_features=np.array([])
        channel_0 = cv.calcHist([image[:,:,0]],[0],None,[self.bin_num],[0,256]).squeeze()
        channel_1 = cv.calcHist([image[:,:,1]],[0],None,[self.bin_num],[0,256]).squeeze()
        channel_2 = cv.calcHist([image[:,:,2]],[0],None,[self.bin_num],[0,256]).squeeze()
        image_features=np.append(image_features,channel_0)
        image_features=np.append(image_features,channel_1)
        image_features=np.append(image_features,channel_2)
        return image_features


    def compute_sharpness(self,image: np.ndarray):
        laplacian=cv.Laplacian(image,-1)
        sharpness=np.mean(laplacian)
        return sharpness


    def compute_gradient(self,image: np.ndarray):
        
        dx=cv.Sobel(image,cv.CV_64F,1,0)
        dy=cv.Sobel(image,cv.CV_64F,0,1)
        mag=cv.magnitude(dx,dy)

        return np.mean(mag)

    def compute_luminance(self,image: np.ndarray):
        Lab_image=cv.cvtColor(image, cv.COLOR_BGR2LAB)
        luminance=np.mean(Lab_image[:,:,0])
        return luminance
        

