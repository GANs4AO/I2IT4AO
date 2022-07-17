# Image to Image Translation for Adaptive Optics (I2IT4AO) 
## *** Under Construction***
**This project aims to improve the science return of future and existing large, ground-based optical telescopes by improving estimation of the wavefront phase. Our approach is to apply image translation using CNNs, specifically conditional Generative Adversarial Networks (cGANSs) to interprete wavefront information from a Shack-Hartmann Wavefront Sensor (SH-WFS) and generate accurate, high fidelity wavefront estimates using Adaptive Optics (AO) control loop data.**

This github repo contains supporting code for conference proceedings:

UAI 2022 paper "**Enhanced Adaptive Optics Control with Image to Image Translation**"  and

SPIE AO VIII paper "**Image to image translation for wavefront and PSF estimation**" 

**by Jeffrey Smith, Jesse Craney, Charles Gretton and Damien Gratadour.**
Please cite these papers if you use this code in your research.

This code was written / adapted by Jeffrey Smith and Jesse Cranney.

To view analysis workflow click on the GAOL_control.ipynb file which is pre calculated for a typical set of parameters.

To use pretrained model from paper after downloading the repo, go into the checkpoint folder and extract the generator network from zips (see the 'read me' in the folder).

python I2IT_train.py --dataroot ./r0_093_W10_p512_s9999/ --load_size=512 --netG=unet_256 --input_nc=1 --output_nc=1 --dataset_mode pistonDivConst10 --name test_GAOL --model pix2pixExM --direction AtoB --ngf 64 --ndf 64 --lambda_L1 150 --lambda_Ex 30

Please see the 'Instructions' for more detail.
### Animation for one of our more recent simulated experiments - using SPHERE parameters and including the DM shape with the Wavefront Sensor image:

![animated_from_images](https://user-images.githubusercontent.com/104841506/178396080-5f5ce8a9-7679-4fd3-bc94-da9fc105f0b3.gif)

*While this gives a good look at the instataneous correction, this example is not a real time control experiment. SPHERE instrument detail: https://www.eso.org/sci/facilities/paranal/instruments/sphere.html

### Instantaneous comparison for simulated control with GAOL AO vs purely linear control. 

![3_grey](https://user-images.githubusercontent.com/104841506/178396370-af214a9c-bc33-473a-9e19-29a86c257d73.png)

#
#
#

Requires working COMPASS tools to run. For installation of COMPASS go to:
https://anr-compass.github.io/compass-new/install.html

Original code repo used as base for research work here:
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

Project based on this paper:
_Image-to-Image Translation with Conditional Adversarial Networks.
Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros. In CVPR 2017. [Bibtex]_

Our work relies heavily of Numerical methods (and code / methods) developed by this work:
_F Ferreira, E Gendron, G Rousset, and D Gratadour. Numerical estimation of wavefront error breakdown in adaptive optics. 
Astronomy and astrophysics (Berlin), 616:A102, 2018a_

This work was supported in part by Oracle Cloud credits
and related resources provided by the Oracle for Research
program.

This research was undertaken with the assistance of re-
sources from the National Computational Infrastructure
(NCI Australia), an NCRIS enabled capability supported by
the Australian Government.

