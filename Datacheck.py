import numpy
import os
from tqdm import tqdm
max = 0
for filename in tqdm(os.listdir("/media/jeff/Data/TrainingData/COMPASS/r0_093_W10_p512_s9999/train/")):
    if filename.endswith(".npz"):
         blah = numpy.load("/media/jeff/Data/TrainingData/COMPASS/r0_093_W10_p512_s9999/train/" +filename)
         if numpy.amax(blah['arr_0']) > max:
             max = numpy.amax(blah['arr_0'])
             print(max)
         continue
    else:
        continue
