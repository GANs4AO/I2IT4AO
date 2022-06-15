# Image to Image Translation for Adaptive Optics (I2IT4AO)

Code support UAI 2022 paper "Enhanced Adaptive Optics Control with Image to Image Translation" by Jeffrey Smith, Jesse Craney, Charles Gretton and Damien Gratadour.
Please cite this paper if you use this code in your research.

This code was written / adapted by Jeffrey Smith and Jesse Cranney.

To use pretrained model from paper  -after downloading the repo, go into the checkpoint folder and extract the generator network from zips (see the read me in the folder).

To train your own model, run the generator script to build a dataset with COMPASS, then train using the command (feel free to vary parameters):
python I2IT_train.py --dataroot ./r0_093_W10_p512_s9999/ --load_size=512 --netG=unet_256 --input_nc=1 --output_nc=1 --dataset_mode pistonDivConst10 --name test_GAOL --model pix2pixExM --direction AtoB --ngf 64 --ndf 64 --lambda_L1 150 --lambda_Ex 30


#
#
#

Requires working COMPASS tools to run. For installation of COMPASS go to:
https://anr-compass.github.io/compass-new/install.html

Original code repo used as base for research work here:
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

Image-to-Image Translation with Conditional Adversarial Networks.
Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros. In CVPR 2017. [Bibtex]
