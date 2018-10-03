import os, sys, traceback, time, h5py

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

from utility import *
from recursive_methods import *

from devices import device_finite, plot_xyz
from graphene_supercell import min_ortho_cell, stripe
from potentials import potential


# -------------------------- NON-INF HELPER MODULES -------------------------- #


def save_band_data_nonpar(dev, kdp_list, bnd_rng, hf, is_save_vecs):
    """
    Calculates the bands for a finite system with a single periodic direction
    and saves them to a *.h5 file

    """

    # Unpack the returned list of lists into k and the eigensystem
    [kdp, eig_sys] = list( zip(*[dev.get_eig_sys(kdp) for kdp in kdp_list]) )

    vals, vecs = list( zip(*eig_sys) )

    hf['kdp' ].resize((len(kdp)), axis = 0)
    hf['vals'].resize((len(kdp)), axis = 0)

    hf['kdp' ][:] = kdp
    hf['vals'][:] = np.array(vals)[:, bnd_rng[0]:bnd_rng[1]]

    if is_save_vecs:
        # Save the transpose of the vectors such that vals[i] corresponds to vecs[i]
        # rather than the default vecs[:,i]
        hf['vecs'].resize((len(kdp)), axis = 0)
        hf['vecs'][:] = np.transpose(vecs, (0,2,1))[:, bnd_rng[0]:bnd_rng[1]]


def save_band_data_par(dev, k_num, kdp_list, bnd_rng, hf, is_save_vecs,
    **prog_kwargs):
    """
    Calculates the bands for a finite system with a single periodic direction
    and saves them to a *.h5 file

    """

    # Select the number of cores to parallelise over

    num_tasks = cpu_num(**prog_kwargs)

    # If kdp_list much longer than the number of cores, split kdp_list
    # into blocks of multiples of the number of cores avilable
    if k_num > 2 * num_tasks:

        brk_list = [] ; i = 1

        while 2 * i * num_tasks < k_num:

            brk_list.append(2 * i * num_tasks) ; i += 1

        kdp_list = np.split(kdp_list, brk_list)

    # If the list is only short, wrap it in an extra set of brackets so
    # that it works correctly with the parallelisation process
    else:

        kdp_list = [kdp_list]


    print_out('Parallelising ' + str(len(kdp_list)) + ' blocks of ' +
        str(len(kdp_list[0])) + ' k-values over ' + str(num_tasks) + ' of ' +
        str(mp.cpu_count()) + ' total cores.')

    # Iterate over the blocks of kdp values, saving to an h5 file after each
    # block returns
    i = 1
    for kdp_block in kdp_list:

        pool = mp.Pool(processes = num_tasks)

        try:

            results = [pool.apply_async(
                dev.get_eig_sys, args = (kdp,)) for kdp in kdp_block]

        except Exception as e:

            print_out('Caught exception in tb_main.py, save_band_data_par()')

            print_out(traceback.format_exception(*sys.exc_info()))

            raise

        pool.close()
        pool.join()

        # Get the results of the parallelisation and unpack the returned list of
        # lists into lists of the k-values and the eigensystem
        [kdp, eig_sys] = list( zip(*[res.get() for res in results]))

        # unpack each of the eigensystems
        vals, vecs = list( zip(*eig_sys) )

        hf['kdp' ].resize(( len(hf['kdp' ]) + len(kdp ) ), axis = 0)
        hf['vals'].resize(( len(hf['vals']) + len(vals) ), axis = 0)

        hf['kdp' ][-len(kdp):] = kdp
        hf['vals'][-len(kdp):] = np.array(vals)[:, bnd_rng[0]:bnd_rng[1]]

        if is_save_vecs:
            hf['vecs'].resize(( len(hf['vecs']) + len(vecs) ), axis = 0)

            print_out('Compressing eigenvectors')
            # Save the transpose of the vectors such that vals[i] corresponds to
            # vecs[i] rather than the default vecs[:,i]
            hf['vecs'][-len(kdp):] = np.transpose(
                vecs, (0,2,1))[:, bnd_rng[0]:bnd_rng[1]]

        print_out('Completed block ' + str(i) + ' of ' + str(len(kdp_list)))

        hf.flush()

        i += 1


def save_band_data(dev, pot, k_num, k_rng = [-np.pi, np.pi], bnd_no = 'All',
    is_save_vecs = False, **prog_kwargs):
    """ Returns an array of the eigenvalues for each k-value """

    print_out('Calculating band data.')

    bnd_no, bnd_rng = set_bnd_rng(bnd_no, len(dev.xyz))

    # Make the list of wavenumbers to iterate over
    kdp_list = np.linspace(k_rng[0], k_rng[1], k_num)

    # Make the k-range into a useable string if plotting between -pi and pi
    k_rng = get_k_rng_str(k_rng)

    # Additional strings to add to file name
    size_str = 'kvals_' + str(k_num) + \
        '_Krng_min_' + str(k_rng[0]) + '_Kmax_' + str(k_rng[1]) + \
        '_bndNo_' + str(bnd_no)

    param_dict = {**dev.get_req_params(), **pot.get_req_params()}

    # Construct the file name
    file_name = make_file_name(pick_directory(dev.orientation), 'BANDS', '.h5')

    params_to_txt(file_name, param_dict, size_str)

    # Open with h5py file so that is closes completely when we leave the method
    with h5py.File(file_name + '.h5', 'w') as hf:

        # Initialise the required datasets
        hf.create_dataset('kdp', (0,), maxshape = (None, ),
            dtype = np.float64, chunks = True,
            compression = "gzip", compression_opts = 4)

        hf.create_dataset('vals', (0, bnd_no),
            maxshape = (None, bnd_no),
            dtype = np.float64, chunks = True,
            compression = "gzip", compression_opts = 4)

        if is_save_vecs:

            hf.create_dataset('vecs', (0, bnd_no, len(dev.xyz)),
                maxshape = (None, bnd_no, len(dev.xyz)),
                dtype = np.complex128, chunks = True,
                compression = "gzip", compression_opts = 4)

        if not prog_kwargs['is_parallel']:

            save_band_data_nonpar(dev, kdp_list, bnd_rng, hf, is_save_vecs)

        else:

            save_band_data_par(dev, k_num, kdp_list, bnd_rng, hf, is_save_vecs,
                **prog_kwargs)

        hf.attrs.update({key:str(val) for key, val in param_dict.items()})


def set_bnd_rng(bnd_no, sys_sz):
    """ Checks the chosen numeber of bands to save """

    # If 'All' set bnd_no to the size of the system
    if bnd_no == 'All':

        bnd_no = sys_sz

    # If int check that it is not larger than the system
    elif type(bnd_no) == int:

        if bnd_no > sys_sz:

            bnd_no = sys_sz

    # I another data type raise an error
    else:

        err_str = self.__class__.__name__ +'(): bnd_no may be either an int or \
        \'All\', not ' + str(bnd_no)

        print_out(err_str)

        raise ValueError(err_str)

    bnd_rng = [(sys_sz - bnd_no) // 2,(sys_sz + bnd_no) // 2]

    return bnd_no, bnd_rng


def get_k_rng_str(k_rng):
    """ Returns k_rng with its elements altered to strings. e.g. pi-> 'pi' """

    for i, val in enumerate(k_rng):

        if np.abs(val) == np.pi:

            if np.sign(val) == +1: k_rng[i] = '+_pi'

            else: k_rng[i] = '-_pi'

        else:

            k_rng[i] = '%.2f' % val

    return k_rng


# ---------------------------- PRIMARY CALL METHOD --------------------------- #


def sys_finite(pot, pot_kwargs, dev_kwargs, prog_kwargs, k_num = 400,
    scaling = 1, is_plot = True, **kwargs):

    if pot_kwargs['is_const_channel'] is False:

        print_out('WARNING - channel is not constant in finite system.')

    if dev_kwargs['is_periodic'] is False:

        print_out('WARNING - system is not periodic along the interface / edge')

    if dev_kwargs['cell_func'] == stripe:

        print_out('WARNING - system is using a stripe as the unit cell, ' +
            'switching to \'min_ortho_cell\'')

        dev_kwargs['cell_func'] = min_ortho_cell

    # Create dev
    dev = device_finite(pot, **dev_kwargs)

    # ------------------------------------------------------------------------ #

    # Include parameters in the output file for comparison

    param_dict = {**dev.get_req_params(), **pot.get_req_params()}

    max_len = max(len(key) for key in param_dict.keys())

    for key, val in param_dict.items():
        
        print_out('\n\t' + key.ljust(max_len + 1) + '\t\t' + str(val),
            is_newline = False)

    # ------------------------------------------------------------------------ #

    if is_plot:

        dev.plot_interface(pot.int_loc)

        plot_xyz(dev.xyz, dev.sublat)

        dev.plot_energies()

    # ------------------------------------------------------------------------ #

    k_mid = -2.1 # Default for zz

    if dev.orientation == 'zz': k_mid = -2.1

    elif dev.orientation == 'ac': k_mid = 0

    if dev.orientation not in ['zz', 'ac']: k_rng = [-np.pi, np.pi]

    else:
        scl = dev_kwargs['scaling']
        pad = 0.1 + (scl - 1) * 0.05
        k_rng = [k_mid - pad, k_mid + pad]
    

    bnd_no = 200 # The number of bands to save. Integer or 'All'

    start_band = time.time()

    save_band_data(dev, pot, k_num, k_rng, bnd_no, **prog_kwargs)

    print_out('Time to calculate band data : ' +
        time_elapsed_str(time.time() - start_band))


def __main__():
    pass


if __name__ == '_main__':

    __main__()