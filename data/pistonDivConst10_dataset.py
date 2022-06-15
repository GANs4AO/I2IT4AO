# pistonDivConst10 - Modified Aligned Dataset for Formatting COMPASS AO data. Used for GAOL experiments (UAI2022)
# Jeffrey Smith
import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy
import scipy.interpolate as sc
import copy

class pistonDivConst10Dataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        #assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        #print(str(opt.dataroot))
        self.pupilMask = numpy.load(opt.dataroot + 'sPupil.npy')

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        ImageSize = 512
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = numpy.load(AB_path)
        A = AB['arr_0']
        A = numpy.squeeze(A)
        A = A / 1200000
        ApadVal = int(ImageSize/2-A.shape[0]/2)
        A = numpy.pad(A, ((ApadVal, ApadVal), (ApadVal, ApadVal)), 'constant')
        A = numpy.expand_dims(A, axis=0)
        #Added phase details:
        #print('phaseplate size: ' + str(AB['arr_2'].shape))
        B = AB['arr_2']
        BpadVal = int(ImageSize/2-B.shape[0]/2)

        C = copy.deepcopy(B)
        #Posssibly remove the mask for C
        C = numpy.multiply(C, self.pupilMask)
        C[C==0.]=numpy.nan
        B = numpy.subtract(B, numpy.nanmean(C))
        B = B * self.pupilMask

        #B = numpy.divide(B, numpy.multiply(numpy.nanmax(numpy.abs(B)), 2))
        B = B/5
        ##print(numpy.nanmax(numpy.abs(B)))
        
        # 90% of window:
        #B = numpy.multiply(B, 0.8)
        B = B + 0.5 #adjust to 0.1 - 0.9
        B = numpy.multiply(B, self.pupilMask)

        B = numpy.pad(B, ((BpadVal, BpadVal), (BpadVal, BpadVal)), 'constant')
        B = numpy.expand_dims(B, axis=0)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
        

    
