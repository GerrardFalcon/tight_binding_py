import os, sys, traceback, datetime

os.environ['MKL_NUM_THREADS'] = '8'

from tb_calc import make_cell_num, do_tb_calc, get_k_params
from graphene_supercell import *


def generate_band_data(cut_vals, name_append, progress_file_name, is_finite,
    SF, is_scale_CN, dev_kwargs, prog_kwargs, sys_kwargs, pot_kwargs):

    exclude = ['pot_type', 'is_const_channel', 'cut_at', 'is_shift_channel_mid']

    replacements = [['-', 'm'], ['.', '_']]

    with open(progress_file_name, 'w') as p_file:

        p_file.write('Calculating band structures into a channel with ' +\
            'potential parameters:\n')

        # ------------ Include potential parameters in output file ----------- #

        max_len = max(len(key) for key in pot_kwargs.keys())

        for key, val in pot_kwargs.items():

            if key not in exclude:

                p_file.write('\n\t'+ key.ljust(max_len + 1)+ '\t\t'+ str(val))

        p_file.flush()

        # ------- Produce data for a range of cuts and record progress ------- #

        for i, cut in enumerate(cut_vals):

            pot_kwargs['cut_at'] = cut

            file_out_name = 'out_BANDS_' + str(dev_kwargs['orientation']) + \
                '_' + str(pot_kwargs['channel_length']) + \
                '_' + str(cut)

            for rep in replacements:

                file_out_name = file_out_name.replace(*rep)

            do_tb_calc(file_out_name + name_append + '.log', is_finite, SF,
                is_scale_CN, dev_kwargs, prog_kwargs, sys_kwargs, **pot_kwargs)

            now = datetime.datetime.now()

            p_file.write('\n\n Completed cut ' + str(i + 1) + ' of ' + \
                str(len(cut_vals)) + ' at ' + \
                str(now.strftime('\t%Y/%m/%d\t%H:%M:%S')))

            p_file.flush()

        p_file.write(" Calculation complete. ")


def __main__():

    # Use the tb_utility module to print the current date to our output file
    file_out_name = 'out_BANDS_AND_VECS_ac_small.log'
    progress_file_name = '../progress_file_ac_small.log'
    name_append = '_small'

    is_generate_many = True

    cut_vals = [-741.206, -746.231, -751.256, -756.281, -761.307, -766.332]

    # zz [-741.206, -746.231, -751.256, -756.281, -761.307, -766.332]
    # ac [-600.503, -605.528, -610.553, -615.578, -620.603, -625.628]
    #np.linspace(-1500, -500, 200)

    # ------------------------------ POTENTIAL ------------------------------- #

    is_finite = True

    is_K_plus = True
    
    # Dictionary of paramters used to define the potential
    pot_kwargs = {
        'pot_type'          :  'well',  # Type of potential to model

        'gap_val'           :   .150,  # 150meV delta0 (.06 for flat profile)
        'offset'            :   .0,      # 0eV

        'well_depth'        :   -.02,  # -20meV U0
        'gap_relax'         :   .3,    # dimensionless beta
        'channel_width'     :   500,    # 850A / 500A

        # Select if the well depth is modulated along the channel
        'is_const_channel'  :   True,
        # If is_const_channel is True, we can also supply a y-value for which to
        # take a cut of the potential
        'cut_at'            :   0,  # -(1200, 1060, 930, 800, 0) w/ d faults

        'gap_min'           :   .01,   # 0.01

        'lead_offset'       :   -.2,   # -0.2 (-.2 -> wl 157, -.5 -> wl 97)
        'channel_length'    :   2400,   # 2000A
        'channel_relax'     :   200,     # 100A (200 max)

        # Rescale the max height of the channel valley to a consistent value
        'is_shift_channel_mid'  :   True
        }

    # Long run ZZ CL 6500, CR 850, CN 2000, SN 1000

    # ------------------------------ SUPERCELL ------------------------------- #
    # Define the number of cells either side of whatever interface we are using

    is_scale_CN = True
    SF = 8 # Factor by which to scale the system

    # 500 / 750 for finite bands

    cell_num_L = 750

    cell_num_R = None       # If None this is set to equal cell_num_L

    stripe_len = 1500       # 900 / 1400

    #   * For channel_width = 500 and channel length = 1000
    #
    #               cell_num (FINITE)       stripe (INFINITE)
    #
    #       ZZ      (300, 300)              800
    #
    #       AC      (160, 160)              1400
    #
    #   Smoothing       AC cells        ZZ Cells
    #
    #   50              170             300     (1200 channel length)
    #   100             175             310
    #   120             200             330
    #   140             205             335
    #   160             210             340
    #   180             215             350
    #   200             220             360
    #
    #   All             700 ?           700         (2400 channel length)
    #
    #   100             510             800     (3000 channel length)
    #   420             560             940

    cell_num, stripe_len = make_cell_num(
        cell_num_L, cell_num_R, stripe_len, SF, is_scale_CN)

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
        'orientation'   :   'zz',          
        'scaling'       :   SF,             # Value by which to scale the system
        }

    # ------------------------------ SIMULATION ------------------------------ #

    # Parameters related to the running of the programme itself
    prog_kwargs = {
        'is_main_task'  :   False,  # False parallelise over fewer cores
        'max_cores'     :   5,      # 20, Max cores to parallelise over
        'is_parallel'   :   True,   # If True, parallelise

        'is_save_vecs'  :   True,   # Save eigenvectors for bndstructure
        'bnd_no'        :   20,     # No. of bands to save. Integer or 'All'
        }

    # ------------------------------------------------------------------------ #

    k_mid = 0       # Default for ac
    k_num = 400     # Number of k-points to sample.

    k_params = get_k_params(k_num, is_K_plus, scaling = dev_kwargs['scaling'],
        orientation = dev_kwargs["orientation"])

    print(k_params)

    sys_kwargs = {
        'is_spectral'   :   False,      # Calc. spec. data in infinite sys
        'is_plot'       :   False,      # Do the plotting methods?
        'is_plot_sublat':   False,      # Whether to pass sublat to plot funcs.

        # k range parameters [minimum, maximum, number of points]
        'k_params'      :   k_params,
        # e range parameters [minimum, maximum, number of points]
        'e_params'      :   [0.025, 0.04, 200],
        }

    if is_finite:

        if is_generate_many:

            generate_band_data(cut_vals, name_append, progress_file_name,
                is_finite, SF, is_scale_CN, dev_kwargs, prog_kwargs, sys_kwargs,
                pot_kwargs)

        else:

            do_tb_calc(file_out_name, is_finite, SF, is_scale_CN, dev_kwargs,
                prog_kwargs, sys_kwargs, **pot_kwargs)

    else:

        do_tb_calc(file_out_name, is_finite, SF, is_scale_CN, dev_kwargs,
            prog_kwargs, sys_kwargs, **pot_kwargs)


if __name__ == '__main__':

    __main__()