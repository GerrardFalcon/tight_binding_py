import os, sys, traceback, time, h5py

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

from utility import *
from recursive_methods import *

from devices import device, plot_xyz
from potentials import potential

# ------------------------------- TRANSMISSION ------------------------------- #

def get_trans_wrapper(kdp_list, energy, lead_left, lead_right, dev,
    small = 1E-6, **kwargs):

    val = [get_trans(
        lead_left, lead_right, dev, kdp, energy, small = 1E-6) \
        for kdp in kdp_list]

    return energy, sum(val) / len(kdp_list)


def get_trans(lead_left, lead_right, dev, kdp, energy, small = 1E-6):
    # Get the SE of the left lead
    lead_left_GF, lead_left_SE = lead_left.get_GF(
        kdp, energy, small, is_return_SE = True)

    lead_right_GF, lead_right_SE = lead_right.get_GF(
        kdp, energy, small, is_return_SE = True)

    if len(dev.cells) == 1:
        # If there is only one cell we do not need to do the return process from
        # the left to right
        GF_part_nn = R_to_L(lead_left, lead_right, dev, kdp, energy, small)

        # Select the fully connected GF
        g_D = GF_part_nn[0]

    else:
        # Calculate the fully connected Greens functions from left to right
        GF_full_nn, GF_full_n1 = double_folding(
            lead_left, lead_right, dev, kdp, energy, small)

        # Select the fully connected GF to the farthest right
        g_D = GF_full_n1[-1]

    # Calculate the gamma factors (imag part of self energy matrices) for the
    # left and right leads
    gamma_L = 1j * (lead_left_SE - np.conj(lead_left_SE).T)
    gamma_R = 1j * (lead_right_SE - np.conj(lead_right_SE).T)

    # Return transmission given these two terms and the device GF
    return np.trace(gamma_L @ np.conj(g_D).T @ gamma_R @ g_D)


def get_transmission(out_file, lead_left, lead_right, dev, pot, en_list, k_list,
    prog_kwargs, small = 1E-6):
    """
    Function which plots the transmission for a range of energies after
    averaging over k between plus and minus pi
    """
    small = 1E-6

    # ------------------------------------------------------------------------ #

    # Additional strings to add to file name
    size_str = 'Energy range : ' + str(min(en_list)) + ' to ' + \
        str(max(en_list)) + '\n\tK range : ' + str(min(k_list)) + ' to ' + \
        str(max(k_list))

    out_file.prnt(size_str)

    param_dict = {**dev.get_req_params(), **pot.get_req_params()}

    # Construct the file name
    file_name = make_file_name(
        pick_directory(dev.get_req_params()), 'TRANS', '.csv')

    # Initialise the file to save to so that other running instances of the
    # script dont try to save to the same place
    np.savetxt(file_name + '.csv', np.array([]), delimiter = ',')

    params_to_txt(file_name, param_dict, size_str)

    # ------------------------------------------------------------------------ #

    # Select the number of cores to parallelise over
    num_tasks = cpu_num(**prog_kwargs)

    # If kdp_list much longer than the number of cores, split kdp_list
    # into blocks of multiples of the number of cores avilable
    if len(en_list) > 2 * num_tasks:

        brk_list = [] ; i = 1

        while 2 * i * num_tasks < len(en_list):

            brk_list.append(2 * i * num_tasks) ; i += 1

        en_list = np.split(en_list, brk_list)

    # If the list is only short, wrap it in an extra set of brackets so
    # that it works correctly with the parallelisation process
    else:

        en_list = [en_list]


    out_file.prnt('Parallelising ' + str(len(en_list)) + ' blocks of ' +
        str(len(en_list[0])) + ' energies over ' + str(num_tasks) + ' of ' +
        str(mp.cpu_count()) + ' total cores.')
        
    keywords = {
        'lead_left'     :   lead_left,
        'lead_right'    :   lead_right,
        'dev'           :   dev,
        'small'         :   small,
    }

    data = np.array([], dtype=np.complex128).reshape(0,2)

    for i, en_block in enumerate(en_list):

        pool = mp.Pool(processes = num_tasks)

        data_tmp = [pool.apply_async(get_trans_wrapper, args = (k_list, en,),
            kwds = keywords) for en in en_block]

        pool.close() ; pool.join()

        # Extract data from the block and add to the list of data points
        data = np.concatenate((data, np.array([d.get() for d in data_tmp])))

        # Save all current data to the provided file path
        np.savetxt(file_name + '.csv', data, delimiter = ',')

        n = datetime.datetime.now()

        out_file.prnt('Completed energy ' + str(i + 1) + ' of ' + \
            str(len(en_list)) + ' at ' + \
            str(n.strftime('\t%Y/%m/%d\t%H:%M:%S')))


# ----------------------------------- LDOS ----------------------------------- #


def plot_LDOS_k_average(lead_left, lead_right, dev, energy, small = 1E-6,
    is_edges = True):

    # Get the recursively calculated GF for a range of kdx
    k_num = 200

    kdx_list = np.linspace(-np.pi, np.pi, k_num)

    cellno = len(dev.cells)

    atno = len(dev.cells[0].xyz)

    if is_edges:

        cell_rng = [0, cellno]

    else:

        pad = 5

        cell_rng = [pad, cellno - pad]

    GF_full_nn_cumulative = np.zeros((cellno, atno, atno), np.complex128)

    for kdx in kdx_list:

        GF_full_nn, GF_full_n1 = double_folding(lead_left, lead_right, dev,
            kdx, energy, small)

        GF_full_nn_cumulative += GF_full_nn

    GF_full_nn_cumulative /= k_num

    LDOS_data = np.array([[np.append(dev.cells[j].xyz[i,:2],
        (-1 / np.pi) * GF_full_nn_cumulative[j,i,i].imag) for i in range(atno)]\
        for j in range(cell_rng[0], cell_rng[1])]).reshape(
        ((cell_rng[1] - cell_rng[0]) * atno, 3))

    reps = 20

    LDOS_data = np.array(
        [LDOS_data + i * dev.lat_vecs_sc[0] for i in range(reps)]).reshape(
        reps * cellno * atno, 3)

    fig = plt.figure()

    all_xyz = np.array(
        [cell.xyz for cell in dev.cells[cell_rng[0] : cell_rng[1]]]
        ).reshape(((cell_rng[1] - cell_rng[0]) * atno, 3))

    pad = 2 ; padd = np.array([-pad, pad])

    xrng = [np.min(all_xyz[:,0]), np.max(all_xyz[:,0]) + (reps - 1) * \
        dev.lat_vecs_sc[0,0]]

    yrng = [np.min(all_xyz[:,1]), np.max(all_xyz[:,1])]

    ax1 = fig.add_subplot(111)

    ax1.scatter(LDOS_data[:,0], LDOS_data[:,1], c = LDOS_data[:,2],
        s = len(dev.cells) / 15)

    ax1.set_aspect('equal', adjustable = 'box')

    ax1.set_xlim(xrng + padd)
    ax1.set_ylim(yrng + padd)

    plt.show()


# --------------------------------- SPECTRAL --------------------------------- #


def spectral_wrapper(kdx, energy, lead_left, lead_right, dev, small = 1E-6):

    return [kdx, energy, (-1 / np.pi) * np.diagonal(double_folding(
        lead_left, lead_right, dev, kdx, energy, small)[0], axis1 = 1,
        axis2 = 2).imag ]


def get_spectral(out_file, lead_left, lead_right, dev, gap_val, small = 1E-6,
    **prog_kwargs):
    """
    Plots the spectral function for a single atom, selected from wihtin this
    function. The expression for the spectral function is given by:

    -1/pi * Im ( Element of the fully conected Greens function for that atom )

    """

    k_num = 2 # 400

    en_num = 2 # 300

    en_lim = 0.5 # 3.5

    kdx_list = np.linspace(-np.pi, np.pi, k_num)

    en_list = np.linspace(-en_lim, en_lim, en_num)

    pool = mp.Pool(processes = cpu_num(**prog_kwargs))

    out_file.prnt('Parallelising over ' + str(cpu_num(**prog_kwargs)) + ' of ' +
        str(mp.cpu_count()) + ' total cores.')

    data = [pool.apply_async(
        spectral_wrapper,
        args = (kdx_list[i], en_list[j]),
        kwds = {'lead_left':lead_left, 'lead_right':lead_right,
        'dev':dev, 'small':small}
        )
        for i in range(len(kdx_list)) for j in range(len(en_list))]

    pool.close()

    pool.join()

    data = [dat.get() for dat in data]

    return k_num, en_num, data


def get_spectral_notParallel(lead_left, lead_right, dev, gap_val,
    small = 1E-6):
    """
    Plots the spectral function for a single atom, selected from wihtin this
    function. The expression for the spectral function is given by:

    -1/pi * Im ( Element of the fully conected Greens function for that atom )

    """

    # Make a grid of the LDOS for that atom for a range of energy and kdx

    k_num = 400 # 300

    en_num = 300 # 200

    en_lim = 0.4

    kdx_list = np.linspace(-np.pi, np.pi, k_num)

    en_list = np.linspace(-en_lim, en_lim, en_num)

    data = [spectral_wrapper(kdx, energy, lead_left, lead_right, dev, small)
        for kdx in kdx_list for en in en_list]

    return data


def save_spectral(spec_data, dev, pot, k_num, en_num):

    # Save the spectral data

    """
    spec_dict = {}

    [spec_dict.update({'cell'+str(cell_idx)+'atom'+str(atom_idx):[ \
        [tmp[0], tmp[1], tmp[2][cell_idx, atom_idx]] for tmp in spec_data]}
        )\
        for cell_idx in range(spec_data[0][2].shape[0]) \
        for atom_idx in range(spec_data[0][2].shape[1])]
    """

    spec_dict = {
        'cell' + str(cell_idx) + 'atom' + str(atom_idx) :
        [[tmp[0], tmp[1], tmp[2][cell_idx, atom_idx]] for tmp in spec_data]
        for cell_idx in range(spec_data[0][2].shape[0]) 
        for atom_idx in range(spec_data[0][2].shape[1])}

    size_str = 'kvals_' + str(k_num) + '_evals' + str(en_num)

    file_name = make_file_name(
        pick_directory(dev.get_req_params()),
        'SPECTRAL_FUNC',
        {**dev.get_req_params(), **pot.get_req_params()},
        size_str)

    savemat(file_name, spec_dict)

    # Create an array of the xyz positions to help visualise which atom we are
    # plotting for

    pos = {}

    xyz_A = np.array([cell.xyz[cell.sublat == 0] for cell in dev.cells])
    xyz_B = np.array([cell.xyz[cell.sublat == 1] for cell in dev.cells])

    [pos.update({'cell'+str(cell_idx)+'atom'+str(atom_idx):\
        np.append(dev.cells[cell_idx].xyz[atom_idx],
            dev.cells[cell_idx].sublat[atom_idx])})\
        for cell_idx in range(spec_data[0][2].shape[0]) \
        for atom_idx in range(spec_data[0][2].shape[1])]

    pos.update({'int_loc':pot.int_loc, 'int_norm':pot.int_norm})

    file_name = make_file_name(
        pick_directory(dev.get_req_params()),
        'xyz',
        {**dev.get_req_params(), **pot.get_req_params()},
        size_str)

    savemat(file_name, pos)


# ---------------------------- PRIMARY CALL METHOD --------------------------- #


def sys_infinite(out_file, pot, pot_kwargs, dev_kwargs, prog_kwargs,
    is_plot = True, is_plot_sublat = False, is_spectral = True,
    e_params = [-0.1, 0.1, 400], **kwargs):

    if dev_kwargs['is_wrap_finite']:

        out_file.prnt('\'is_wrap_finite\' cannot be True for an infinite ' + \
            'system. Setting to False\n')

        dev_kwargs['is_wrap_finite'] = False

    # Create the dev
    dev = device(pot = pot, **dev_kwargs)

    # ------------------------------------------------------------------------ #

    # Include parameters in the output file for comparison

    param_dict = {**dev.get_req_params(), **pot.get_req_params()}

    out_file.prnt_dict(param_dict, is_newline = False)

    # ------------------------------------------------------------------------ #

    # Generate leads
    lead_left = make_lead(dev, 'L', pot = pot, **dev_kwargs)
    lead_right = make_lead(dev, 'R', pot = pot, **dev_kwargs)

    if not pot.pot_params['is_const_channel']:

        smoothing_info = pot.get_pot_smoothing_info(
            dev.get_xyz(), dev.get_sublat())

        for info in smoothing_info:
            
            out_file.prnt(info)

    if is_plot:

        # Check if we are meant to pass sublat to the plotting function
        if is_plot_sublat:
            pot.plot_pot_3D(dev.get_xyz(), dev.get_sublat())

        else:
            pot.plot_pot_3D(dev.get_xyz())

        dev.plot_interface(pot.int_loc)

        dev.plot_energies()

    param_dict = {**dev.get_req_params(), **pot.get_req_params()}

    ####                            LDOS                                ####

    energy = 0.0
    #plot_LDOS_k_average(lead_left, lead_right, dev, energy)

    ####                    SPECTRAL FUNCTION                           ####

    if is_spectral:

        str_ex = None#'_atom_2'

        start_spectral = time.time()

        k_num, en_num, spec_data = get_spectral(out_file, lead_left, lead_right,
            dev, pot_kwargs['gap_val'], **prog_kwargs)

        out_file.prnt('Time to calculate spectral data : ' +
            time_elapsed_str(time.time() - start_spectral))

        save_spectral(spec_data, dev, pot, k_num, en_num)


    out_file.prnt('Calculating transmission')

    en_list = np.linspace(*e_params)

    k_list = [0]#np.linspace(-np.pi, np.pi, 400)

    get_transmission(out_file, lead_left, lead_right, dev, pot, en_list, k_list,
        prog_kwargs)


def __main__():
    pass


if __name__ == '_main__':

    __main__()