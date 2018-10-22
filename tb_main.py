import os, sys, traceback, time, h5py

os.environ['MKL_NUM_THREADS'] = '8'

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


def scaling_prnt(SF, is_scale_CN):
    """
    Method which prints to the output file info about the scaling of the cells
    """
    if SF != 1 and not is_scale_CN:

        print_out('System scaling is not 1. Remember to change the number of '+\
            'cells in the system accordingly, or set is_scale_CN to True.\n')

    if SF != 1 and is_scale_CN:

        print_out('System scaling is not 1. Automatically scaling the number '+\
            'of cells by ' + str(SF) + '.\n')


def __main__():

    # Use the tb_utility module to print the current date to our output file

    file_out_name = 'out_ac_bands_750.txt'

    create_out_file(file_out_name)

    # ------------------------------ POTENTIAL ------------------------------- #

    pot_type = 'well'

    is_finite = True

    SF = 8 # Factor by which to scale the system

    is_scale_CN = True

    scaling_prnt(SF, is_scale_CN)
    """
    # Dictionary of paramters used to define the potential
    pot_kwargs = {
        'gap_val'           :   0.150,  # 100meV delta0
        'offset'            :   0,      # 0eV

        'well_depth'        :   -0.02,  # -20meV U0
        'gap_relax'         :   0.3,    # dimensionless beta
        'channel_width'     :   500,    # 850A / 500A

        # Select if the well depth is modulated along the channel
        'is_const_channel'  :   False,
        # If is_const_channel is True, we can also supply a y-value for which to
        # take a cut of the potential
        'cut_at'            :   0,  # -(1200, 1060, 930, 800, 0) w/ d faults

        'gap_min'           :   0.01,   # 0.01
        'lead_offset'       :   0.0,   # -0.1

        'channel_length'    :   1000,   # 1000A
        'channel_relax'     :   100     # 100A
        }
    """
    pot_kwargs = {
        'gap_val'           :   0.03,  # 100meV delta0
        'offset'            :   0,      # 0eV

        'well_depth'        :   -0.0001,  # -20meV U0
        'gap_relax'         :   0.3,    # dimensionless beta
        'channel_width'     :   500,    # 850A / 500A

        # Select if the well depth is modulated along the channel
        'is_const_channel'  :   True,
        # If is_const_channel is True, we can also supply a y-value for which to
        # take a cut of the potential
        'cut_at'            :   -750,  # -(1200, 1060, 930, 800, 0) w/ d faults

        'gap_min'           :   0.01,   # -40meV U0
        'lead_offset'       :   0.0,   # -0.1
        'channel_length'    :   1000,   # 2000A
        'channel_relax'     :   100     # 100A
        }
    

    if pot_kwargs['is_const_channel']:
        print_out('Calculating for a CONSTANT channel')
    else:
        print_out('Calculating for a VARYING channel')

    # ------------------------------ SUPERCELL ------------------------------- #

    # Define the number of cells either side of whatever interface we are using
    cell_num_L = 800        # 300 / 160 SCALES WITH POTENTIAL DIMENSIONS
    cell_num_R = None       # If None this is set to equal cell_num_L

    stripe_len = 1400       # 800 / 1400

    #   * For channel_width = 500 and channel length = 1000
    #
    #               cell_num (FINITE)       stripe (INFINITE)
    #
    #       ZZ      (300, 300)              800
    #
    #       AC      (160, 160)              1400
    #

    if cell_num_R is None: cell_num_R = cell_num_L

    if is_scale_CN:

        cell_num = (
            np.sum(np.divmod(cell_num_L, SF)),
            np.sum(np.divmod(cell_num_R, SF)))

        stripe_len = np.sum(np.divmod(stripe_len, SF))

    else: cell_num = (cell_num_L, cell_num_R)

    # Dictionary of paramters used to define the dev (no potential)
    dev_kwargs = {
        'is_gamma_3'    :   True,           # On/off gamma 3 coupling in BLG
        'latt_type'     :   BLG_cell,       # Pick a lattice type (MLG_cell,
                                            # BLG_cell) from grpahene_supercell
        'cell_func'     :   stripe,         # min_ortho_cell vs stripe
        'cell_num'      :   cell_num,       # Pick the number of cells in the
                                            # transport direction
        'stripe_len'    :   stripe_len,     # num of cells to repeat in stripe
        'is_periodic'   :   True,           # Periodic in non-trnsprt direction?

        # Whether to wrap the finite system into a torus
        'is_wrap_finite':   True,

        # orientation of the cells along the x-direction perp. to transport
        'orientation'   :   'ac',          
        'scaling'       :   SF,             # Value by which to scale the system
        }

    # ------------------------------ SIMULATION ------------------------------ #

    # Parameters related to the running of the programme itself
    prog_kwargs = {
        'is_main_task'  :   False,          # False parallelise over fewer cores
        'max_cores'     :   10,             # 20, Max cores to parallelise over
        'is_parallel'   :   True,           # If True, parallelise
        'is_save_vecs'  :   False,          # Save eigenvectors for bndstructure
        }

    # ------------------------------------------------------------------------ #

    sys_kwargs = {
        'is_spectral'   :   False,      # Calc. spec. data in infinite sys
        'is_plot'       :   False,      # Do the plotting methods?
        'is_plot_sublat':   False,      # Whether to pass sublat to plot funcs.
        'k_num'         :   400,        # No. of k-values to do calc.s for
        }

    # ------------------------------------------------------------------------ #

    # int_norm -  Vector normal to the potential interface    

    # Define the potential's orientation. This is switched by 90 degrees if we
    # are wanting to study transport along the interface (infinite system)

    # We also need to switch the orientation of the cells when we do this
    if is_finite:

        int_norm = [0, 1, 0]

        dev_kwargs['orientation_along_trnsprt_axis'] = dev_kwargs['orientation']

    else:
        int_norm = [1, 0, 0]

        dev_kwargs['orientation_along_trnsprt_axis'] = dev_kwargs['orientation']
        ori_list = ['zz', 'ac']
        if dev_kwargs['orientation_along_trnsprt_axis'] in ori_list:
            for i in range(len(ori_list)):
                if dev_kwargs['orientation_along_trnsprt_axis'] == ori_list[i]:
                    dev_kwargs['orientation'] = ori_list[-(i+1)]

    int_loc = [0, 0, 0]

    # Create the potential
    pot = potential(pot_type, int_loc, int_norm, **pot_kwargs)

    # ------------------------------------------------------------------------ #

    start = time.time()

    if is_finite:

        sys_finite(pot, pot_kwargs, dev_kwargs, prog_kwargs, **sys_kwargs)

    else:

        sys_infinite(pot, pot_kwargs, dev_kwargs, prog_kwargs, **sys_kwargs)

    print_out('Complete. Total elapsed time : ' +
        time_elapsed_str(time.time() - start))


if __name__ == '__main__':

    killer = WhoKilledMe()

    try:

        __main__()

    except Exception as e:

        print_out('Caught exception in tb_main.py')

        print_out( ''.join( traceback.format_exception( *sys.exc_info() ) ) )

        raise

        sys.exit()