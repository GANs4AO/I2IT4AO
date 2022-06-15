import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy


class npAlignedDataset(BaseDataset):
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
        #print(str(self.AB_paths))

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
        
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        #AB = Image.open(AB_path).convert('RGB')
        AB = numpy.load(AB_path)
        # split AB image into A and B
        #w, h = AB.size
        #w2 = int(w / 2)
        #A = AB.crop((0, 0, w2, h))
        #B = AB.crop((w2, 0, w, h))
        #print(str(AB['arr_0'].shape))
        A = AB['arr_0']
        A = numpy.squeeze(A)
        A = numpy.expand_dims(A, axis=0)
        #A = numpy.unsqueeze(A)
        
        #Added phase details:
        #print('phaseplate size: ' + str(AB['arr_2'].shape))
        B = AB['arr_2'][66:194, 66:194]
        #print('max: ' + str(numpy.amax(B)) + 'min: ' + str(numpy.amin(B)) + 'shape: ' + str(B.shape))
        B = numpy.subtract(B, numpy.amin(B))
        B = numpy.divide(B, numpy.amax(B))
#        print('max: ' + str(numpy.amax(B)) + 'min: ' + str(numpy.amin(B)) + 'shape: ' + str(B.shape))
        #B = numpy.squeeze(B)
        B = numpy.expand_dims(B, axis=0)
        #B = numpy.unsqueeze(B)
        #B = numpy.empty((128, 128, 1))
        #B[..., 0] = AB['arr_2']
        #print(str(A.shape))
        #print(str(B.shape))
        #print(str(A.dtype))
        # apply the same transform to both A and B
        #transform_params = get_params(self.opt, A.size)
        #A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        #B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        #A = A_transform(A)
        #B = B_transform(B)
        #print('testing image count: ' + str(len))
        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
