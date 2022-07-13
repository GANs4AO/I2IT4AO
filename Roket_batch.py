"""
Script to create roket buffer files from a directory of COMPASS paramater files
!!! Update for you local paths etc. !!!
Written By Jeffrey Smith

"""

import os
from tqdm import tqdm
import subprocess

directory = './COMPASS_PARAM/r0_batch/'
for filename in os.listdir(directory):
    if filename.endswith(".py"):
        print(os.path.join(directory, filename))
        #os.system("python3 /home/jeff/shesha/guardians/roket.py /home/jeff/GITs/COMPASS/r0_batch/r0_35_W10_p512_s9999.py")# + os.path.join(directory, filename))
        os.system('python3 ~/shesha/guardians/roket.py ' + os.path.join(directory, filename))
        os.rename('~/roket_output/roket_default.h5', '~/roket_output/batch/roket_' + filename + '.h5')

    else:
        continue

