# directory structure:
# AmbiQuant/
#     |-- AmbiQuantFunctions.py // this is the main import module so that all function can be called from this module
#     |-- calculation.py // some calculation related functions
#     |-- data_processing.py
#     |-- plot_quality_score.py // the functions that generate the qc report lots
#     |-- quality_control_function.py //function that calculate qc metrics that will be used to make the qc report plots 
#     |-- QCPipe_dir // a directory that contains the QCPipe module which is a module already published in our previous publication
#         |-- QCPipe.py // the main import module to call all functions in fcc_utils and qc_utils 
#         |-- fcc_utils.py
#         |-- qc_utils.py


from  calculation import *
from data_processing import *
from plot_quality_score import *
from quality_control_function import *

from QCPipe_dir import QCPipe
