import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


"""
Loader: class used to load a dataset composed of images. it loads the images, computes and returns a descriptor for each image along with the corresponding class labels

""" 
class Loader:


    def __init__(self, dataset :str, datapath :str, bin_num :int =256, use_gradient :bool =False, use_sharpness :bool = True, use_luminance :bool =True):
        """
        Creates an image loader.
    
        Args:
            dataset (str): name of the dataset to be used eg. MWD
            datapath (str): path of the data to be loaded eg. ./data/MWD/training
            bin_num (int): number of bins that will be used in the histogram computation (quantize the level axis of the histogram)
            use_gradient (bool): flag to choose to use or note the gradient in the descriptors
            use_sharpness (bool): flag to choose to use or note the sharpness in the descriptors
            use_luminance (bool): flag to choose to use or note the luminance in the descriptors

        """ 


        self.datapath=datapath

        #ADDING a slash at the end of the path if not present
        if(not self.datapath.endswith('/')):
            self.datapath=self.datapath+'/'

        self.datasets=['MWD','ACDC','UAVid','syndrone']
        self.dataset=dataset
        self.bin_num=bin_num

        #checking that the provided number of bins is valid
        if(self.bin_num <= 0 or bin_num > 256):
            raise Exception('Invalid number of bins {}'.format(bin_num))

        #checking that the provided dataset is valid
        if(dataset not in self.datasets):
            raise Exception('Wrong dataset provided, chose from MWD,ACDC,UAVid,syndrone')


        #selecting the classes based on the chosen dataset
        #ATTENTION: classes represents the label that will be given to the images while candidates represents the strings that will be compared with the image names to assing the labels
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


        #setting flags
        self.use_gradient=use_gradient
        self.use_sharpness=use_sharpness
        self.use_luminance=use_luminance



    def getClasses(self) -> list :
        """
        Returns the classes of the dataset.
    
        Returns:
            classes (list): list of the distinct classes in the dataset

        """ 
        return self.classes


    def getClass(self,image_name :str) -> str:
        """
        Returns the class given the name of an image.
    
        Returns:
            class (str): class of the image based on the image name

        """ 

        for i in range(len(self.candidates)):
            if(self.candidates[i] in image_name):
                return self.classes[i]

        return None






    def loadImages(self) -> tuple[np.array,np.array]:
        """
        loads the images returning  the arrays corresponfing to the image descriptors and the classes for each image
    
        Returns:
            X (numpy.array): matrix having in each row the descriptor of an image
            y (numpy.array): array having in each position the label of the corresponding image in X
        """ 

        #Choosing which method to used based on the dataset
        if self.dataset == 'MWD':
            return self.loadMwd()
        else:
            return self.loadOther()













    def loadMwd(self) -> tuple[np.array,np.array]:

        """
        loads the images returning  the arrays corresponfing to the image descriptors and the classes for each image.
        method used only for the MWD dataset since it has a particular folder structure.
    
        Returns:
            X (numpy.array): matrix having in each row the descriptor of an image
            y (numpy.array): array having in each position the label of the corresponding image in X
        """ 
        X=np.array([])
        y=np.array([])
        

        for cls in self.classes:

            #for each class, load all the images in the corresponfing folder

            tmp_path=self.datapath+cls
            files_list = os.listdir(tmp_path)

            #load each file
            for file in files_list:
                #load image
                image=cv.imread(tmp_path+'/'+file)

                #computing descriptors only if the file is actually an image
                if(image is not None) :
                    
                    #getting descriptors
                    image_feat=self.getFeatures(image)
                    
                    #getting class label
                    image_cls=self.getClass(file)
                    

                    #appending label and descriptor to the matrices

                    y=np.append(y,image_cls)

                    if(len(X)==0):
                        X=image_feat
                    else:
                        X=np.vstack((X,image_feat))

                else:
                    print('Found a non image file {} at: {}'.format(file,tmp_path))
                    
        return X,y


    def loadOther(self) -> tuple[np.array,np.array]:
        """
        loads the images returning  the arrays corresponfing to the image descriptors and the classes for each image
    
        Returns:
            X (numpy.array): matrix having in each row the descriptor of an image
            y (numpy.array): array having in each position the label of the corresponding image in X
        """ 

        X=np.array([])
        y=np.array([])

        path=self.datapath
        files_list = os.listdir(path)

        #load each file
        for file in files_list:
            
            #load image
            image=cv.imread(path+'/'+file)

            #computing descriptors only if the file is actually an image
            if(image is not None) :

                #getting descriptors
                image_feat=self.getFeatures(image)
                
                #getting class label
                image_cls=self.getClass(file)
                

                #appending label and descriptor to the matrices

                y=np.append(y,image_cls)

                if(len(X)==0):
                    X=image_feat
                else:
                    X=np.vstack((X,image_feat))
                
            else:
                print('Found a non image file {} at: {}'.format(file,path))
            
                    
        return X,y



    def getFeatures(self,image :np.ndarray) -> np.array:
        """
        Computes the descriptor of a given image

        Args:
            image (numpy.ndarray): the image to be described 
    
        Returns:
            feat (numpy.array): array rapresenting the image descriptor containing the features
        """ 

        #computing the features based on the histograms
        feat=self.computeHistograms(image)

        #computing other auxiliary features based on the provided flags
        if(self.use_sharpness):
            feat=np.append(feat,self.computeSharpness(image))
        if(self.use_gradient):
            feat=np.append(feat,self.computeGradient(image))
        if(self.use_luminance):
            feat=np.append(feat,self.computeLuminance(image))
        return feat







    def computeHistograms(self,image: np.ndarray) -> np.array:
        """
        Computes the histograms of a given image

        Args:
            image (numpy.ndarray): the image  
    
        Returns:
            image_features (numpy.array): array rapresenting the histogram of the image concatenated in a single array
        """ 
        
        if image is None:
            raise Exception('Invalid None image')

        #computing histograms based on the number of bins provided as a parameter of the class
        image_features=np.array([])
        channel_0 = cv.calcHist([image[:,:,0]],[0],None,[self.bin_num],[0,256]).squeeze()
        channel_1 = cv.calcHist([image[:,:,1]],[0],None,[self.bin_num],[0,256]).squeeze()
        channel_2 = cv.calcHist([image[:,:,2]],[0],None,[self.bin_num],[0,256]).squeeze()

        #concatenating the histograms of the different channels
        image_features=np.append(image_features,channel_0)
        image_features=np.append(image_features,channel_1)
        image_features=np.append(image_features,channel_2)

        return image_features






    def computeSharpness(self,image: np.ndarray):
        """
        Computes the sharpness of a  given image exploiting the laplacian

        Args:
            image (numpy.ndarray): the image to be described 
    
        Returns:
            sharpness (float): the average value of the sharpness of the image
        """ 

        #Converting image to grayscale
        image=cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        #computing the laplacian
        laplacian=cv.Laplacian(image,-1)

        #computing the sharpness as mean of the laplacian over the entire image (anc over all the channels)
        sharpness=np.mean(laplacian)

        return sharpness


    def computeGradient(self,image: np.ndarray):
        """
        Computes the gradient of a  given image exploiting the Sobel masks

        Args:
            image (numpy.ndarray): the image to be described 
    
        Returns:
            mag (float): the average value of the gradient of the image
        """ 
        
        #Converting image to grayscale
        image=cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        #Computing the derivatives in both direction exploiting the sobel mask
        dx=cv.Sobel(image,cv.CV_64F,1,0)
        dy=cv.Sobel(image,cv.CV_64F,0,1)

        #computing the gradient magnitude based on the derivatives
        mag=cv.magnitude(dx,dy)

        return np.mean(mag)

    def computeLuminance(self,image: np.ndarray):
        """
        Computes the luminance of a  given image exploiting the LAB representation

        Args:
            image (numpy.ndarray): the image to be described 
    
        Returns:
            luminance (float): the average value of the luminance of the image
        """
        #converting image
        Lab_image=cv.cvtColor(image, cv.COLOR_BGR2LAB)

        #computing average of the L (luminance) channel
        luminance=np.mean(Lab_image[:,:,0])

        return luminance
        

