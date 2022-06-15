"""
Generate training data for GAOL experiments 
Jeffrey Smith & Jesse Craney

"""
import numpy
from docopt import docopt
#from shesha.util.utilities import load_config_from_file
from shesha.config import ParamConfig
from shesha.supervisor.compassSupervisor import CompassSupervisor as Supervisor
import os

#arguments = docopt(__doc__)
#param_file = arguments["<parameters_filename>"]
#save_file = arguments["<save_filename>"]
param_file = "./r0_093_W10_p512_s9999.py"
save_root =  "r0_093_W10_p512_s9999"
os.makedirs(save_root)
save_file_test = save_root + "/test/"
os.makedirs(save_file_test)
save_file_train = save_root + "/train/"
os.makedirs(save_file_train)

config = ParamConfig(param_file)
#if arguments["--r0"]: # Check if r0 has been given as arguments
#config.p_atmos.set_r0(float(arguments["--r0"])) # Change the r0 before initalization

sup = Supervisor(config) # Instantiates your class and load parameters
pupil_save = save_root + '/mPupil'
numpy.save(pupil_save, sup.get_m_pupil())
pupil_save = save_root + '/sPupil'
numpy.save(pupil_save, sup.get_s_pupil())
dummyarray = numpy.array([0])
#sup.loop(30000)
turbs = [0.093, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
for x in turbs:
  sup.atmos.set_r0(x)
  for i in range(50000):
    sup.loop(1)
    Image1 = sup.wfs.get_wfs_image(0)
    theWFSstack = numpy.array([Image1])
    theWFSstack = numpy.transpose(theWFSstack, (1, 2, 0))
    numpy.savez_compressed(save_file_train + str(x) + "_" +str(i), theWFSstack, dummyarray, sup.target.get_tar_phase(0))
