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

    file_out_name = 'out.txt'

    create_out_file(file_out_name)

    ################################ POTENTIAL #################################

    pot_type = None#'well'

    # Dictionary of paramters used to define the potential
    pot_kwargs = {
        'gap_val'           :   0.150,  # 100meV delta0
        'offset'            :   0,      # 0eV

        'well_depth'        :   -0.02,  # -20meV U0
        'gap_relax'         :   0.3,    # dimensionless beta
        'channel_width'     :   500,    # 850A / 500A

        # Select if the well depth is modulated along the channel
        'is_const_channel'  :   True,
        # If is_const_channel is True, we can also supply a y-value for which to
        # take a cut of the potential
        'cut_at'            :   0,  # -(1200, 1060, 930, 800, 0) w/ d faults

        'gap_min'           :   0.01,   # -40meV U0
        'channel_length'    :   2000,   # 2000A
        'channel_relax'     :   100     # 100A
        }

    ################################ SUPERCELL #################################

    # Define the number of cells either side of whatever interface we are using
    cell_num_L = 1          # 500
    cell_num_R = 0       # If None this is set to equal cell_num_L
    #
    #               cell_num (FINITE)       cell_num (INFINITE)
    #
    #       ZZ      (500, 500)              
    #
    #       AC      (750, 750)
    #

    if cell_num_R is None: cell_num_R = cell_num_L

    cell_num = (cell_num_L, cell_num_R)

    # Dictionary of paramters used to define the dev (no potential)
    dev_kwargs = {
        'is_gamma_3'    :   True,           # On/off gamma 3 coupling in BLG
        'latt_type'     :   BLG_cell,       # Pick a lattice type (MLG_cell,
                                            # BLG_cell) from grpahene_supercell
        'cell_func'     :   stripe, # min_ortho_cell vs stripe
        'cell_num'      :   cell_num,       # Pick the number of cells in the
                                            # transport direction
        'stripe_len'    :   1000,             # num of cells to repeat in stripe
        'is_periodic'   :   True,           # Periodic in non-trnsprt direction?
        'is_wrap_finite':   True,          # Whether to wrap the finite system
                                            # into a torus
        'orientation'   : 'zz'              # orientation of the cells
        }

    ################################ SIMULATION ################################

    # Parameters related to the running of the programme itself
    prog_kwargs = {
        'is_main_task'  :   False,          # False parallelise over fewer cores
        'max_cores'     :   20,             # 20, Max cores to parallelise over
        'is_parallel'   :   True,           # If True, parallelise
        }

    ############################################################################

    is_finite = False

    sys_kwargs = {
        'is_spectral'   :   False,           # Calc. spec. data in infinite sys
        'is_plot'       :   False,
        }

    ############################################################################

    # int_norm -  Vector normal to the potential interface    

    # Define the potential's orientation. This is switched by 90 degrees if we
    # are wanting to study transport along the interface (infinite system)
    if is_finite: int_norm = [0, 1, 0] ;

    else: int_norm = [1, 0, 0]

    int_loc = [0, 0, 0]

    # Create the potential
    pot = potential(pot_type, int_loc, int_norm, **pot_kwargs)

    ############################################################################

    start = time.time()

    if is_finite:

        sys_finite(pot, pot_kwargs, dev_kwargs, prog_kwargs, **sys_kwargs)

    else:

        sys_infinite(pot, pot_kwargs, dev_kwargs, prog_kwargs, **sys_kwargs)

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