# Image to Image Translation for Adaptive Optics (I2IT4AO) 

**This project aims to improve the science return of future and existing large, ground-based optical telescopes by improving estimation of the wavefront phase. Our approach is to apply image translation using CNNs, specifically conditional Generative Adversarial Networks (cGANs) to interpret wavefront information from a Shack-Hartmann Wavefront Sensor (SH-WFS) image and generate accurate, high fidelity wavefront estimates using Adaptive Optics (AO) control loop data.**

**We demonstrate that cGAN wavefront estimation produces excellent results in simulation for PSF reconstruction and with the GAN Assisted Open Loop methodology there is potential for real time control applications. Other possible benefits not discussed in papers are detection of waffle mode and good low strehl performance in high wind (50m/s)**

This github repo contains supporting code for conference proceedings:

UAI 2022 paper "**Enhanced Adaptive Optics Control with Image to Image Translation**"  and
Jeffrey Smith, Jesse Cranney, Charles Gretton, Damien Gratadour, "Enhanced adaptive optics control with image to image translation," Proceedings of the Thirty-Eighth Conference on Uncertainty in Artificial Intelligence, PMLR 180:1846-1856, 2022. https://proceedings.mlr.press/v180/smith22a.html


SPIE AO VIII paper "**Image to image translation for wavefront and PSF estimation**" 
Jeffrey Smith, Jesse Cranney, Charles Gretton, Damien Gratadour, "Image-to-image translation for wavefront and PSF estimation," Proc. SPIE 12185, Adaptive Optics Systems VIII, 121852L (29 August 2022); https://doi.org/10.1117/12.2629638

**by Jeffrey Smith, Jesse Craney, Charles Gretton and Damien Gratadour.**
Please cite these papers if you use this code in your research.

This code was written / adapted by Jeffrey Smith and Jesse Cranney.
Code is functional but is in the process of cleaning up!

Please see the 'Instructions' for how to use this repo.

### Animation for one of our more recent simulated experiments - using SPHERE parameters and including the DM shape with the Wavefront Sensor image:

![animated_from_images](https://user-images.githubusercontent.com/104841506/178396080-5f5ce8a9-7679-4fd3-bc94-da9fc105f0b3.gif)

*While this gives a good look at the instantaneous correction, this example is not a real time control experiment. SPHERE instrument detail: https://www.eso.org/sci/facilities/paranal/instruments/sphere.html

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
Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros. In CVPR 2017.

Our work relies heavily of Numerical methods (and code / methods) developed by this work:
_F Ferreira, E Gendron, G Rousset, and D Gratadour. Numerical estimation of wavefront error breakdown in adaptive optics. 
Astronomy and astrophysics (Berlin), 616:A102, 2018a_

.
.
.

**UPDATE: 
This initial work lead to some great results for PSF Reconstruction, AO real time control and atmospheric parameter estimation, and Free Space Optical Communication. Please see the following papers that this work facilitated:**

B. Pou, J. Smith, E. Quinones, M. Martin, D. Gratadour, **"Model-free reinforcement learning with a non-linear reconstructor for closed-loop adaptive optics control with a pyramid wavefront sensor,"** Proc. SPIE 12185, Adaptive Optics Systems VIII, 121852U (29 August 2022); https://doi.org/10.1117/12.2627849

Jeffrey Smith, Jesse Cranney, Charles Gretton, Damien Gratadour, **"Image-to-image translation for wavefront and point spread function estimation,"** J. Astron. Telesc. Instrum. Syst. 9(1) 019001 (19 January 2023) https://doi.org/10.1117/1.JATIS.9.1.019001

Jeffrey Peter Smith, Jesse Cranney, Charles Gretton, Damien Gratadour. **"A Study of Network-based Wavefront Estimation with Noise,"** Adaptive Optics for Extremely Large Telescopes 7th Edition, ONERA, Jun 2023, Avignon, France. https://dx.doi.org/10.13009/AO4ELT7-2023-084

B. Pou, J. Smith, E. Quinones, M. Martin, and D. Gratadour, **"Integrating supervised and reinforcement learning for predictive control with an unmodulated pyramid wavefront sensor for adaptive optics,"** Opt. Express 32, 37011-37035 (2024). https://doi.org/10.1364/OE.530254

Smith, J., Fujii, T., Craney, J., & Gretton, C. (2025). **"Fried Parameter Estimation from Single Wavefront Sensor Image with Artificial Neural Networks,"** https://doi.org/10.48550/arXiv.2504.17029

.
.
.

**Acknowledgements:**
This work was supported in part by Oracle Cloud credits
and related resources provided by the Oracle for Research
program.

This research was undertaken with the assistance of re-
sources from the National Computational Infrastructure
(NCI Australia), an NCRIS enabled capability supported by
the Australian Government.

**Contact**
I've finished my PhD, and after quite a bit of searching have not found anywhere to continue this work professionally, although I am aware it is being used in several areas of research. It remains a hobby where time permits, so if you would like to discuss applications of adaptations please feel free to contact me at (Dr.Smith.Prime @ Gmail).

Oh, the pain. 
