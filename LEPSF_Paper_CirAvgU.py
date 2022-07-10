import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as tick

#Roket_PSF = np.load('/home/jeff/roket_out/PSFs/roket_012_W5.npz')['arr_0']
Pred_PSF = np.load('/home/jeff/GITs/wavefrontestimation_i2it/results/test_UNet2/LE_PSF_r_093_W10_s3333.npz')['arr_0']
GAM_PSF = np.load('/home/jeff/GITs/wavefrontestimation_i2it/results/test_UNet2/GAM_093.npz')['arr_0']
COMPASS_PSF = np.load('/home/jeff/GITs/wavefrontestimation_i2it/results/test_UNet2/LE_PSF_Sim_r_093_W10_s3333.npz')['arr_0']
GAM_log = np.log(GAM_PSF)
#Roket_Log = np.log(Roket_PSF)
COMPASS_Log = np.log(COMPASS_PSF)
GAM_Log = np.log(GAM_PSF)
P2S_Log = np.log(Pred_PSF)

#blah = np.abs(COMPASS_PSF - Pred_PSF)
#blah[blah==0.0]=0.00000000000000001


diff_GAM_Log = np.log(np.abs(COMPASS_PSF - GAM_PSF))
diff_P2S_Log = np.log(np.abs(COMPASS_PSF - Pred_PSF))
#diff_P2S_Log = np.log(np.abs(blah))

pix = 96
x = -15
imsize = 2048
im_mid = imsize/2

radial_GAM_Diff = diff_GAM_Log[1024-pix:1024+pix, 1024-pix:1024+pix]
radial_P2S_Diff = diff_P2S_Log[1024-pix:1024+pix, 1024-pix:1024+pix]



# Adjust plot parameters
#mpl.rcParams['font.family'] = 'Avenir'
#mpl.rcParams['font.size'] = 16
#mpl.rcParams['axes.linewidth'] = 2
#mpl.rcParams['axes.spines.top'] = False
#mpl.rcParams['axes.spines.right'] = False
#mpl.rcParams['xtick.major.size'] = 7
#mpl.rcParams['xtick.major.width'] = 2
#mpl.rcParams['ytick.major.size'] = 7
#mpl.rcParams['ytick.major.width'] = 2  # Create figure and add subplot

fig = plt.figure(figsize=(6.5,8))
gs = GridSpec(3, 120)
p1 = fig.add_subplot(gs[0, 0:60])  # ROKET PSF
p2 = fig.add_subplot(gs[1, 0:60])  # GAM PSF
p3 = fig.add_subplot(gs[2, 0:60])  # GAM Diff
p4 = fig.add_subplot(gs[1, 63:120])  # P2S PSF
p5 = fig.add_subplot(gs[2, 63:120])  # P2S Diff
pCirc = fig.add_subplot(gs[0, 66:108]) #table
plt.subplot(p1)
plt.axis('off')
plt.title('Simulation')
plt.imshow(COMPASS_Log[1024-pix:1024+pix, 1024-pix:1024+pix], label="Simulation", cmap=plt.cm.binary)
#plt.colorbar()
plt.colorbar(format=tick.FormatStrFormatter('%.0f'))
plt.clim(0, x)
plt.subplot(pCirc)
plt.title('Circular Avg')
cen_x = pix
cen_y = pix  # Get image parameters
a = radial_GAM_Diff.shape[0]
b = radial_GAM_Diff.shape[1]  # Find radial distances
[X, Y] = np.meshgrid(np.arange(b) - cen_x, np.arange(a) - cen_y)
R = np.sqrt(np.square(X) + np.square(Y))
rad = np.arange(1, np.max(R), 1)
#rad = np.arange(1, 64, 1)
intensity = np.zeros(len(rad))
intensity2 = np.zeros(len(rad))
index = 0
bin_size = 1
for i in rad:
    mask = (np.greater(R, i - bin_size) & np.less(R, i + bin_size))
    values = radial_GAM_Diff[mask]
    intensity[index] = np.mean(values)
    index += 1
index = 0
bin_size = 1
for j in rad:
    mask = (np.greater(R, j - bin_size) & np.less(R, j + bin_size))
    values = radial_P2S_Diff[mask]
    intensity2[index] = np.mean(values)
    index += 1
#fig = plt.figure()
ax = plt.subplot(pCirc)
ax.plot(rad, intensity, linewidth=1, label='reference error')  # Edit axis labels
ax.plot(rad, intensity2, linewidth=1, label='inferred error')  # Edit axis labels
ax.set_xlabel('Radial Distance')
ax.set_ylabel('Average Intensity')
ax.axes.get_xaxis().set_visible(False)
#plt.subplot(pCirc).scale(0.75,0.75)
plt.xlim([0, 88])
plt.ylim([x, 0])
ax.yaxis.tick_right()
ax.legend()
plt.subplot(p2)
plt.axis('off')
plt.title('Reference')
plt.imshow(GAM_Log[1024-pix:1024+pix, 1024-pix:1024+pix], label="Prediction", cmap=plt.cm.binary)
plt.colorbar(format=tick.FormatStrFormatter('%.0f'))
plt.clim(0, x)
plt.subplot(p3)
plt.axis('off')
plt.title('Reference Error')
plt.imshow(diff_GAM_Log[1024-pix:1024+pix, 1024-pix:1024+pix], label="Difference", cmap=plt.cm.binary)
plt.colorbar(format=tick.FormatStrFormatter('%.0f'))
plt.clim(0, x)
plt.subplot(p4)
plt.axis('off')
plt.title('Inferred')
plt.imshow(P2S_Log[1024-pix:1024+pix, 1024-pix:1024+pix], label="Difference", cmap=plt.cm.binary)
plt.colorbar(format=tick.FormatStrFormatter('%.0f'))
plt.clim(0, x)
plt.subplot(p5)
plt.axis('off')
plt.title('Inferred Error')
plt.imshow(diff_P2S_Log[1024-pix:1024+pix, 1024-pix:1024+pix], label="Difference", cmap=plt.cm.binary)
plt.colorbar(format=tick.FormatStrFormatter('%.0f'))
plt.clim(0, x)
#plt.show()
plt.savefig("LEPSFs.svg", format="svg", bbox_inches='tight', dpi=300)
plt.savefig("LEPSFs.png", format="png", bbox_inches='tight', dpi=300)
