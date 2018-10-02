import os, sys, traceback, time, h5py

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

from utility import *
from recursive_methods import *

from devices import device, plot_xyz
from potentials import potential

# ------------------------------- TRANSMISSION ------------------------------- #

def get_transmission_wrapper(kdp_list, energy, lead_left, lead_right, dev,
    small = 1E-6, **kwargs):

    val = [get_transmission(
        lead_left, lead_right, dev, kdp, energy, small = 1E-6) \
        for kdp in kdp_list]

    return energy, sum(val) / len(kdp_list)


def get_transmission(lead_left, lead_right, dev, kdp, energy, small = 1E-6):
    # Get the SE of the left lead
    lead_left_GF, lead_left_SE = lead_left.get_GF(
        kdp, energy, small, is_return_SE = True)

    # Get the GF's for the rest of the dev
    GF_part_nn = R_to_L(lead_left, lead_right, dev, kdp, energy, small)

    if len(dev.cells) == 1:

        lead_right_GF, lead_right_SE = lead_right.get_GF(
            kdp, energy, small, is_return_SE = True)

    else:

        # Calculate the SE for the cell to the right of the fully connected cell
        lead_right_SE = np.conj(dev.cells[1].get_V()).T @ GF_part_nn[1] @ \
            dev.cells[1].get_V()

    # Calculate the gamma factors (imag part of self energy matrices) for the
    # left and right leads
    gamma_L = 1j * (lead_left_SE - np.conj(lead_left_SE).T)
    gamma_R = 1j * (lead_right_SE - np.conj(lead_right_SE).T)

    # Select the fully connected GF
    g_D = GF_part_nn[0]

    # Return transmission given these two terms and the device GF
    return np.trace(gamma_L @ np.conj(g_D).T @ gamma_R @ g_D)


def plot_transmission_test(lead_left, lead_right, dev, prog_kwargs,
    small = 1E-6):
    """
    Function which plots the transmission for a range of energies after
    averaging over k between plus and minus pi
    """
    small = 1E-6
    lim = 0.1
    en_list = np.linspace(0.03, 0.05, 200)#np.linspace(-lim, lim, 1600)#

    k_list = [0]#np.linspace(-np.pi, np.pi, 400)

    print_out(
        'Energy range : ' + str(min(en_list)) + ' to ' + str(max(en_list)))

    print_out('K range : ' + str(min(k_list)) + ' to ' + str(max(k_list)))

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


    print_out('Parallelising ' + str(len(en_list)) + ' blocks of ' +
        str(len(en_list[0])) + ' energies over ' + str(num_tasks) + ' of ' +
        str(mp.cpu_count()) + ' total cores.')

    data = np.array([])
        
    keywords = {
        'lead_left'     :   lead_left,
        'lead_right'    :   lead_right,
        'dev'           :   dev,
        'small'         :   small,
    }

    i = 1
    for en_block in en_list:

        pool = mp.Pool(processes = num_tasks)

        data_tmp = [pool.apply_async(get_transmission_wrapper, args = (k_list, en,), \
            kwds = keywords) for en in en_block]

        pool.close()
        pool.join()

        data = np.append(data, data_tmp)

        print_out('Completed energy ' + str(i) + ' of ' + str(len(en_list)))
        i += 1

    d_t = list(zip(*[dat.get() for dat in data]))

    [en, data] = list(zip(*[dat.get() for dat in data]))

    #[k, en, data] = list(zip(*[dat.get() for dat in data]))

    data_save = [[en[i], data[i]] for i in range(len(data))]

    np.savetxt('test_1.csv', data_save, delimiter = ',')
    """

    small = 1E-6

    k_num = 50

    kdx_list = np.linspace(-np.pi, np.pi, k_num)

    kdx_list = [0]
    k_num = 1

    en_list = np.linspace(-1, 1, 200)

    data = []

    for en in en_list:

        # Average transmission over the k - values
        k_av = np.sum(
            np.array([get_transmission(lead_left, lead_right, dev, kdx, en,
                small) for kdx in kdx_list])) / k_num

        data.append([en, k_av])

    data = np.array(data)

    np.savetxt('test.csv', data, delimiter = ',')
    """

    """

    plt.plot(data[:, 0].real, data[:, 1].real,'k-')

    plt.show()

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
    """


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


def get_spectral(lead_left, lead_right, dev, gap_val,
    small = 1E-6, **prog_kwargs):
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

    print_out('Parallelising over ' + str(cpu_num(**prog_kwargs)) + ' of ' +
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
        pick_directory(dev.orientation),
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
        pick_directory(dev.orientation),
        'xyz',
        {**dev.get_req_params(), **pot.get_req_params()},
        size_str)

    savemat(file_name, pos)


# ---------------------------- PRIMARY CALL METHOD --------------------------- #


def sys_infinite(pot, pot_kwargs, dev_kwargs, prog_kwargs, is_plot = True,
    is_spectral = True, **kwargs):

    if dev_kwargs['is_wrap_finite']:

        print_out('\'is_wrap_finite\' cannot be True for an infinite system.' +
            ' Setting to False')

        dev_kwargs['is_wrap_finite'] = False

    # Create the dev
    dev = device(pot = pot, **dev_kwargs)

    # ------------------------------------------------------------------------ #

    # Include parameters in the output file for comparison

    param_dict = {**dev.get_req_params(), **pot.get_req_params()}

    max_len = max(len(key) for key in param_dict.keys())

    for key, val in param_dict.items():
        
        print_out('\n\t' + key.ljust(max_len + 1) + '\t\t' + str(val),
            is_newline = False)

    # ------------------------------------------------------------------------ #

    # Generate leads
    lead_left = make_lead(dev, 'L', pot = pot, **dev_kwargs)
    lead_right = make_lead(dev, 'R', pot = pot, **dev_kwargs)

    if is_plot:

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

        k_num, en_num, spec_data = get_spectral(
            lead_left, lead_right, dev, pot_kwargs['gap_val'],
            **prog_kwargs)

        print_out('Time to calculate spectral data : ' +
            time_elapsed_str(time.time() - start_spectral))

        # Construct the file name
        file_name = make_file_name(pick_directory(dev.orientation),
            'SPECTRAL', '.h5')

    print_out('Calculating transmission')

    file_name = make_file_name(pick_directory(dev.orientation),
            'TYPE', '.extension')

    params_to_txt(file_name, param_dict)
        
    #save_spectral(spec_data, dev, pot, k_num, en_num)

    plot_transmission_test(lead_left, lead_right, dev, prog_kwargs)


def __main__():
    pass


if __name__ == '_main__':

    __main__()