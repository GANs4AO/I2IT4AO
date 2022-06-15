"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import numpy
import scipy.interpolate as sc
import copy
from shesha.config import ParamConfig
import torch

import numpy
from docopt import docopt
#from shesha.util.utilities import load_config_from_file
from shesha.supervisor.compassSupervisor import CompassSupervisor as Supervisor
from shesha.ao.basis import compute_btt, compute_cmat_with_Btt
from scipy.sparse import csr_matrix

def MakePSF(Phase):
    sPupil = sup.get_s_pupil()
    WFS = numpy.multiply(Phase, sPupil)
    #save the Loop state before changes
    temp = sup.target.get_tar_phase(0)
    sup.target.set_tar_phase(0, WFS)
    sup.target.comp_tar_image(0, compLE=False)
    PSF = sup.target.get_tar_image(0, expo_type="se")
    #set the phase back
    sup.target.set_tar_phase(0, temp)
    sup.target.comp_tar_image(0, compLE=False)
    return {'PSF': PSF}

def SetData():
    A = copy.deepcopy(sup.wfs.get_wfs_image(0)) / 1200000
    ApadVal = int(uNetsize / 2 - A.shape[0] / 2)
    A = numpy.pad(A, ((ApadVal, ApadVal), (ApadVal, ApadVal)), 'constant')
    A = numpy.expand_dims(A, axis=0)
    A = numpy.expand_dims(A, axis=0)
    A = torch.from_numpy(A)
    B = copy.deepcopy(sup.target.get_tar_phase(0))
    B = B * sup.get_s_pupil()
    BpadVal = int(uNetsize / 2 - B.shape[0] / 2)
    B[B == 0] = numpy.nan
    temp_min = numpy.nanmin(B)
    temp_max = numpy.nanmax(B)
    temp_mean = numpy.nanmean(B)
    temp_absMax = numpy.nanmax(numpy.abs(B - temp_mean))
    B = numpy.subtract(B, temp_mean)
    B = numpy.divide(B, temp_absMax * 2)
    B = numpy.multiply(B, 0.8)
    B = numpy.add(B, 0.5)
    B = numpy.nan_to_num(B)
    B = numpy.pad(B, ((BpadVal, BpadVal), (BpadVal, BpadVal)), 'constant')
    B = numpy.expand_dims(B, axis=0)
    B = numpy.expand_dims(B, axis=0)
    B = torch.from_numpy(B)
    AB_path = ''
    return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}
    
if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    #dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.print_networks(False)

    windspeeds = [5]
    for x in windspeeds:


        # Initialise COMPASS



        param_file = "/home/jeff/GITs/COMPASS/Sim_param_r0_012_W" + str(x) + "_N2.py"
        #config = load_config_from_file(param_file)
        config = ParamConfig(param_file)
        sup = Supervisor(config) # Instantiates your class and load parameters
        n = config.p_loop.niter
        sup.loop(1000) #dump the first 1000 iterations
        pupilMask = sup.get_s_pupil()
        #init for cmat


        uNetsize = 512
        """
        nslopes = sup.rtc.get_slopes(0).size
        slopes = numpy.zeros((n, nslopes), dtype=numpy.float32)
        nfiltered = 20
        IFpzt = sup.rtc._rtc.d_control[1].d_IFsparse.get_csr()
        TT = numpy.array(sup.rtc._rtc.d_control[1].d_TT)
        Btt, P = compute_btt(IFpzt.T, TT)
        tmp = Btt.dot(Btt.T)
        sup.rtc._rtc.d_control[1].load_Btt(tmp[:-2, :-2], tmp[-2:, -2:])
        compute_cmat_with_Btt(sup.rtc._rtc, Btt, nfiltered)
        cmat = sup.rtc.get_command_matrix(0)
        """
        trueDim = int(pupilMask.shape[0])

        debug = False
        WFSmax = 0
        #print(pupilMask)
        PSF_LE = numpy.multiply(sup.target.get_tar_image(0), 0)
        PSF_LE_Sim = numpy.multiply(sup.target.get_tar_image(0), 0)
        # test with eval mode. This only affects layers like batchnorm and dropout.
        # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
        # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
        if opt.eval:
            model.eval()

        #for i, data in enumerate(dataset):
        for i in range(n):
            if i >= opt.num_test:  # only apply our model to opt.num_test images.
                break
            sup.loop(1)#advance simulation

            #set data object here: 'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path
            #if (numpy.amax(sup.wfs.get_wfs_image(0)) > WFSmax):
            #   WFSmax = numpy.amax(sup.wfs.get_wfs_image(0))

            data = SetData()
            #print(data['A'].shape)
            #print(data['B'].shape)

            model.set_input(data)  # unpack data from data loader

            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()     # get image paths
            """calc the Btt surface"""
            """
            slopes = sup.rtc.get_slopes(0)
            vTT = cmat[-2:,:].dot(slopes)
            sup.dms.set_command(vTT, dm_index=1)# <-- needed?
            TTphase = copy.deepcopy(sup.dms.get_dm_shape(1))
    
            offset = int((TTphase.shape[0] - trueDim)/2)
            TTphase = TTphase[offset:offset+trueDim, offset:offset+trueDim]
            TTphase = TTphase*pupilMask
            true_max = (numpy.amax(TTphase)*2*numpy.pi)/1.65
            true_min = (numpy.amin(TTphase)*2*numpy.pi)/1.65
            #print('TTphaseMax: ' + str(true_max))
            #print('TTphaseMin: ' + str(true_min))
            #then probably do micron to rad here
            """

            fileName = 'newname' + str(i)
            """get the phase max and min"""

            #FullSizePhase = numpy.load(fileName)['arr_2']
            FullSizePhase = sup.target.get_tar_phase(0)
            FullSizePhase = numpy.multiply(FullSizePhase, pupilMask)
            FullSizePhase[FullSizePhase==0]=numpy.nan
            #print('TrueMax: ' + str(numpy.nanmax(FullSizePhase)))
            #print('TrueMin: ' + str(numpy.nanmin(FullSizePhase)))
            TruePhaseMax = numpy.nanmax(FullSizePhase)
            TruePhaseMin = numpy.nanmin(FullSizePhase)
            TruePhaseMean = numpy.nanmean(FullSizePhase)
            TruePhaseAbsMax = numpy.nanmax(numpy.abs(FullSizePhase - TruePhaseMean))



            """
            if(debug == True):
               numpy.savez_compressed("truePhase", FullSizePhase)
            true_max = numpy.amax(FullSizePhase)
            true_min = numpy.amin(FullSizePhase)
            if(true_max == 0):
                FullSizePhase[FullSizePhase == 0] = true_min
                true_max = numpy.amax(FullSizePhase)
            elif(true_min == 0):
                FullSizePhase[FullSizePhase == 0] = true_max
                true_min = numpy.amin(FullSizePhase)
            if(debug == True):    
               print("truth max: "+ str(true_max))
               print("truth min: "+ str(true_min))
            """
            #fileName = fileName.split('/')
            #splitlen = len(fileName)
            #fileName = fileName[len(fileName)-1]
            #print(fileName)
            for label, im_data in visuals.items():
                if label == 'fake_B':# or label == 'real_B':
                   im = im_data.cpu()
                   im = im.detach().numpy()
                   im = numpy.squeeze(im)
                   #name = 'test'
                   #image_name = '%s_%s' % (label, fileName)
                   #image_name = os.path.join('test/', image_name)
                   #print(str(total_iters) + ' ' + str(epoch))
                   #image_name = 'results/' + opt.name +'/' + image_name

                   """turn this off"""
                   #numpy.savez_compressed(image_name, im)

                   #print(im.shape)
                   #print(pupilMask.shape)
                   #im = numpy.squeeze(im)
                   #im = interpolate(pupilMask, im)['B']

                   offset = int((im.shape[0] - trueDim)/2)
                   im = im[offset:offset+trueDim, offset:offset+trueDim]
                   im = numpy.multiply(im, pupilMask)

                   """CHECK preadjustment predicted phase amplitude"""

                   ampCheck = copy.deepcopy(im)
                   c_max = numpy.amax(ampCheck)
                   c_min = numpy.amin(ampCheck)
                   if(c_max == 0):
                      ampCheck[ampCheck == 0] = c_min
                      c_max = numpy.amax(ampCheck)
                   elif(c_min == 0):
                      ampCheck[ampCheck == 0] = c_max
                      c_min = numpy.amin(ampCheck)

                   if(debug == True):
                      print("Precicted UNadj max: "+ str(c_max))
                      print("Predicted UNadj min: "+ str(c_min))

                   """adjust output"""
                   #im = numpy.subtract(im, c_min)
                   im = numpy.subtract(im, 0.5)
                   #im = numpy.divide(im, c_max-c_min)
                   im = numpy.divide(im, 0.8)
                   im = numpy.multiply(im, pupilMask)
                   im = im * 2 * TruePhaseAbsMax

                   if(debug == True):
                      ampCheck = copy.deepcopy(im)
                      d_max = numpy.amax(ampCheck)
                      d_min = numpy.amin(ampCheck)
                      if(d_max == 0):
                         ampCheck[ampCheck == 0] = d_min
                         d_max = numpy.amax(ampCheck)
                      elif(d_min == 0):
                         ampCheck[ampCheck == 0] = d_max
                         d_min = numpy.amin(ampCheck)
                      print("Precicted MIDadj max: "+ str(d_max))
                      print("Predicted MIDadj min: "+ str(d_min))

                   #im = numpy.multiply(im, (TruePhaseMax-TruePhaseMin))
                   #im = numpy.add(im, TruePhaseMin)
                   ### subtract piston
                   #im_copy = copy.deepcopy(im)
                   #im_copy = im_copy + 1000
                   #im_copy = im_copy * pupilMask
                   #im_copy[im_copy == 0] = numpy.nan
                   #im_copy = im_copy - 1000
                   #im_mean = numpy.nanmean(im_copy)
                   #im = im - im_mean
                   #im = numpy.multiply(im, pupilMask)

                   """CHECK final Predicted phase amplitude"""
                   if(debug == True):
                      ampCheck = copy.deepcopy(im)
                      c_max = numpy.amax(ampCheck)
                      c_min = numpy.amin(ampCheck)
                      if(c_max == 0):
                         ampCheck[ampCheck == 0] = c_min
                         c_max = numpy.amax(ampCheck)
                      elif(c_min == 0):
                         ampCheck[ampCheck == 0] = c_max
                         c_min = numpy.amin(ampCheck)
                      print("Precicted adj max: "+ str(c_max))
                      print("Predicted adj min: "+ str(c_min))

                   thePSF = MakePSF(im)['PSF']
                   PSF_LE = numpy.add(PSF_LE, thePSF)
                   PSF_LE_Sim = numpy.add(PSF_LE_Sim, sup.target.get_tar_image(0))
                   #print(PSFname)
                   #numpy.savez_compressed(PSFname, thePSF)
                   if(debug == True):
                      if label == 'fake_B':
                         numpy.savez_compressed("phase_pred", im)
                      else:
                         numpy.savez_compressed("phase_realProcessed", im)

        PSF_LE = numpy.divide(PSF_LE, n)
        PSF_LE_Sim = numpy.divide(PSF_LE_Sim, n)
        image_name = 'results/' + opt.name +'/LE_PSF_012_W' + str(x) + '_N2.npz'
        numpy.savez_compressed(image_name, PSF_LE)
        image_name = 'results/' + opt.name +'/LE_PSF_Sim_012_W' + str(x) + '_N2.npz'
        numpy.savez_compressed(image_name, PSF_LE_Sim)
    
    

