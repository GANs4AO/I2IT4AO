# Image to Image Translation for Adaptive Optics (I2IT4AO) 
# *** Under Construction***

Supporting code for conference proceedings:

UAI 2022 paper "**Enhanced Adaptive Optics Control with Image to Image Translation**"  and

SPIE AO VIII paper "**Image to image translation for wavefront and PSF estimation**" 

**by Jeffrey Smith, Jesse Craney, Charles Gretton and Damien Gratadour.**
Please cite these papers if you use this code in your research.

This code was written / adapted by Jeffrey Smith and Jesse Cranney.

To view analysis workflow click on the GAOL_control.ipynb file which is pre calculated for a typical set of parameters.

To use pretrained model from paper after downloading the repo, go into the checkpoint folder and extract the generator network from zips (see the 'read me' in the folder).

To train your own model, run the generator script to build a dataset with COMPASS, then train using the command (feel free to vary parameters):

python I2IT_train.py --dataroot ./r0_093_W10_p512_s9999/ --load_size=512 --netG=unet_256 --input_nc=1 --output_nc=1 --dataset_mode pistonDivConst10 --name test_GAOL --model pix2pixExM --direction AtoB --ngf 64 --ndf 64 --lambda_L1 150 --lambda_Ex 30

Animation for one of our more recent simulated experiments - using SPHERE parameters and including the DM shape with the Wavefront Sensor image:

![animated_from_images](https://user-images.githubusercontent.com/104841506/178138053-ff97923d-7a43-4487-b406-d550143b9194.gif)

*note that while this looks impresive, we are just subtracting the inferred wavefront from the closed loop output and calculating the PSF from this difference in the case of 'AI on'. While this gives a good look at the instataneous correction this example is not a real time control experiment.

#
#
#

Requires working COMPASS tools to run. For installation of COMPASS go to:
https://anr-compass.github.io/compass-new/install.html

Original code repo used as base for research work here:
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

Project based on this paper:
Image-to-Image Translation with Conditional Adversarial Networks.
Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros. In CVPR 2017. [Bibtex]

This work was supported in part by Oracle Cloud credits
and related resources provided by the Oracle for Research
program.

This research was undertaken with the assistance of re-
sources from the National Computational Infrastructure
(NCI Australia), an NCRIS enabled capability supported by
the Australian Government.

