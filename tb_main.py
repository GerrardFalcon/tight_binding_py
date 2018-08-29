import os, sys, traceback, time, h5py

os.environ['MKL_NUM_THREADS'] = '4'

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

from scipy.io import savemat # Need to edit this out - deprecated

from utility import *
from recursive_methods import *

from devices import device, device_finite, plot_xyz
from potentials import potential

from sys_funcs_infinite import *
from sys_funcs_finite import *


def __main__():

    # Use the tb_utility module to print the current date to our output file

    file_out_name = 'out_zz_width_500.txt'

    create_out_file(file_out_name)

    ################################ POTENTIAL #################################

    pot_type = 'well'

    # Dictionary of paramters used to define the potential
    pot_kwargs = {
        'gap_val'           :   0.150,  # 100meV delta0
        'offset'            :   0,      # 0eV

        'well_depth'        :   -0.02,  # -20meV U0
        'gap_relax'         :   0.3,    # dimensionless beta
        'channel_width'     :   500,    # 850A L

        # Select if the well depth is modulated along the channel
        'is_const_channel'  :   True,

        'channel_depth'     :   -0.04,  # -40meV U0
        'channel_length'    :   1000,   # 1000A
        'channel_relax'     :   200     # 200A
        }

    ################################ SUPERCELL #################################

    cell_func = BLG_cell # Cell function to use (BLG vs MLG)

    # Define the number of cells either side of whatever interface we are using
    cell_num_L = 500          # 400
    cell_num_R = None       # If None this is set to equal cell_num_L

    if cell_num_R is None: cell_num_R = cell_num_L

    cell_num = (cell_num_L, cell_num_R)

    orientation = 'zz'

    # Dictionary of paramters used to define the dev (no potential)
    dev_kwargs = {
        'is_gamma_3'    :   True    # On/off gamma 3 coupling in the BLG system
        }

    ################################ SIMULATION ################################

    # Parameters related to the running of the programme itself
    prog_kwargs = {
        'is_main_task'  :   False,          # False parallelise over fewer cores
        'max_cores'     :   20,             # 20, Max cores to parallelise over
        'is_parallel'   :   True,           # If True, parallelise
        'is_plot'       :   False
        }

    ############################################################################

    is_finite = True

    ############################################################################

    # int_norm -  Vector normal to the potential interface    

    # Define the potential's orientation. This is switched by 90 degrees if we
    # are wanting to study transport along the interface (infinite system)
    if is_finite: int_norm = [0,1,0] ;

    else: int_norm = [1,0,0]

    int_loc = [0, 0, 0]

    # Create the potential
    pot = potential(pot_type, int_loc, int_norm, **pot_kwargs)

    ############################################################################

    start = time.time()

    if is_finite:

        sys_finite(cell_func, orientation, cell_num, pot,
            pot_kwargs, dev_kwargs, prog_kwargs)

    else:

        sys_infinite(cell_func, orientation, cell_num, pot, 
            pot_kwargs, dev_kwargs, prog_kwargs)

    print_out('Complete. Total elapsed time : ' +
        time_elapsed_str(time.time() - start))


if __name__ == '__main__':

    try:

        __main__()

    except Exception as e:

        print_out('Caught exception in tb_main.py')

        print_out( ''.join( traceback.format_exception( *sys.exc_info() ) ) )

        raise


"""
shift = 0 # Amout to shifft the interface by (default is zero)
    # Calculate the location of the interface given cell_num etc.

    int_loc_y = cell_num[0] * np.dot(
        cell_func(index = 0, orientation = orientation, **dev_kwargs
            ).lat_vecs_sc[1], int_norm) + shift
"""