import os, sys, traceback, time, h5py

os.environ['MKL_NUM_THREADS'] = '6'

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

from scipy.io import savemat # Need to edit this out - deprecated

from tb_utility import *
from recursive_methods import *
from devices import device, device_finite, plot_xyz
from potentials import potential


########################     NON-INF HELPER MODULES     ########################


def save_band_data_nonpar(dev, kdp_list, bnd_rng, hf):
    """
    Calculates the bands for a finite system with a single periodic direction
    and saves them to a *.h5 file

    """

    # Unpack the returned list of lists into k and the eigensystem
    [kdp, eig_sys] = list( zip(*[dev.get_eig_sys(kdp) for kdp in kdp_list]) )

    vals, vecs = list( zip(*eig_sys) )

    hf['kdp' ].resize((len(kdp)), axis = 0)
    hf['vals'].resize((len(kdp)), axis = 0)
    hf['vecs'].resize((len(kdp)), axis = 0)

    hf['kdp' ][:] = kdp
    hf['vals'][:] = np.array(vals)[:, bnd_rng[0]:bnd_rng[1]]
    # Save the transpose of the vectors such that vals[i] corresponds to vecs[i]
    # rather than the default vecs[:,i]
    hf['vecs'][:] = np.transpose(vecs, (0,2,1))[:, bnd_rng[0]:bnd_rng[1]]


def save_band_data_par(dev, k_num, kdp_list, bnd_rng, hf, **prog_kwargs):
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
        hf['vecs'].resize(( len(hf['vecs']) + len(vecs) ), axis = 0)

        hf['kdp' ][-len(kdp):] = kdp
        hf['vals'][-len(kdp):] = np.array(vals)[:, bnd_rng[0]:bnd_rng[1]]

        print_out('Compressing eigenvectors')
        # Save the transpose of the vectors such that vals[i] corresponds to
        # vecs[i] rather than the default vecs[:,i]
        hf['vecs'][-len(kdp):] = np.transpose(
            vecs, (0,2,1))[:, bnd_rng[0]:bnd_rng[1]]

        print_out('Completed block ' + str(i) + ' of ' + str(len(kdp_list)))

        hf.flush()

        i += 1


def save_band_data(dev, pot, k_num, k_rng = [-np.pi, np.pi], bnd_no = 'All',
    **prog_kwargs):
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
    file_name = make_file_name(
        pick_directory(dev.orientation),
        'BANDS',
        param_dict,
        size_str) + '.h5'

    # Open with h5py file so that is closes completely when we leave the method
    with h5py.File(file_name, 'w') as hf:

        # Initialise the required datasets
        hf.create_dataset('kdp', (0,), maxshape = (None, ),
            dtype = np.float64, chunks = True,
            compression = "gzip", compression_opts = 4)

        hf.create_dataset('vals', (0, bnd_no),
            maxshape = (None, bnd_no),
            dtype = np.float64, chunks = True,
            compression = "gzip", compression_opts = 4)

        hf.create_dataset('vecs', (0, bnd_no, len(dev.xyz)),
            maxshape = (None, bnd_no, len(dev.xyz)),
            dtype = np.complex128, chunks = True,
            compression = "gzip", compression_opts = 4)

        if not prog_kwargs['is_parallel']:

            save_band_data_nonpar(dev, kdp_list, bnd_rng, hf)

        else:

            save_band_data_par(dev, k_num, kdp_list, bnd_rng, hf, **prog_kwargs)

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


##########################     INF HELPER MODULES     ##########################


def plot_transmission_test(lead_left, lead_right, dev, small = 1E-6):
    """
    Function which plots the transmission for a range of energies after
    averaging over k between plus and minus pi
    """

    small = 1E-6

    k_num = 200

    kdx_list = np.linspace(-np.pi, np.pi, k_num)

    en_list = np.linspace(-1, 1, 40)

    data = []

    for en in en_list:

        # Average transmission over the k - values
        k_av = np.sum(
            np.array([get_transmission(lead_left, lead_right, dev, kdx, en,
                small) for kdx in kdx_list])) / k_num

        data.append([en, k_av])

    data = np.array(data)

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

    k_num = 200 # 400

    en_num = 200 # 300

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


#################################### UTILITY ###################################


def data_save(data, data_type_str, file_str_ex = None):

    all_params = {**dev.get_req_params(), **pot.get_req_params()}

    file_name = make_file_name(
        pick_directory(all_params['ori']),
        data_type_str,
        {**dev.get_req_params(), **pot.get_req_params()},
        file_str_ex)

    np.savetxt(file_name + '.csv', data, delimiter = ',')


def make_file_name(dir_str, data_str, param_dict, extra_str = None):
    """ Adds dictionary values to the naming string """

    str_tmp = dir_str + data_str

    for key, value in param_dict.items():
        str_tmp += '_' + key + '_' + str(value)

    if extra_str is not None:
        str_tmp += '_' + extra_str

    file_name = str_tmp.replace('.', '_').replace('-', 'm').replace('+', 'p')

    print_out(str(data_str) + ' data saving to :\n\n\t' + str(file_name))

    return file_name


def pick_directory(ori):
    # Sub-directory to save the data to

    dir_ext = '../saved_files/'

    if ori == 'zz':

        return dir_ext + 'zz/'

    elif ori == 'ac':

        return dir_ext + 'ac/'

    else:

        return dir_ext + 'other/'


def cpu_num(is_main_task, max_cores, **kwargs):

    cpu_no = mp.cpu_count()
    
    if is_main_task:

        if cpu_no < max_cores:

            return cpu_no

        else:

            return max_cores
    
    else:
    
        if cpu_no <= 3:

            return 2
    
        elif cpu_no > 3:

            if cpu_no // 2 < max_cores:

                return cpu_no // 2

            else:

                return max_cores

        else:

            return 1


def time_elapsed_str(time):
    """ Makes a formated string for the time elapsed during the calculation """

    if time < 60:

        return ' %d minutes' % time

    else:

        return ' %d hours and %d minutes' % divmod(time, 60)


############################     INF VS NON-INF     ############################

def sys_finite(cell_func, orientation, cell_num, pot, pot_kwargs, dev_kwargs,
    prog_kwargs):

    # Create dev
    dev = device_finite(cell_func, orientation, cell_num, pot, **dev_kwargs)

    #dev.plot_interface()

    #plot_xyz(dev.xyz, dev.sublat)

    #dev.plot_energies()

    ############################################################################

    k_num = 400

    if orientation == 'zz':

        k_rng = [-2.3,-1.8]

    elif orientation == 'ac':

        k_rng = [-0.25, 0.25]

    else:

        k_rng = [-np.pi, np.pi]
    

    bnd_no = 200 # The number of bands to save. Integer or 'All'

    start_band = time.time()

    save_band_data(dev, pot, k_num, k_rng, bnd_no, **prog_kwargs)

    print_out('Time to calculate band data : ' +
        time_elapsed_str(start_band - time.time()))


def sys_infinite(cell_func, orientation, cell_num, pot, pot_kwargs, dev_kwargs,
    prog_kwargs):

    # Create the dev
    dev = device(cell_func, orientation, cell_num, pot, **dev_kwargs)

    # Generate leads
    lead_left = make_lead(dev, cell_func, 'L', pot, **dev_kwargs)
    lead_right = make_lead(dev, cell_func, 'R', pot, **dev_kwargs)


    dev.plot_interface()

    dev.plot_energies()

    ####                            LDOS                                ####

    energy = 0.0
    plot_LDOS_k_average(lead_left, lead_right, dev, energy)

    ####                    SPECTRAL FUNCTION                           ####

    str_ex = None#'_atom_2'

    start_spectral = time.time()

    k_num, en_num, spec_data = get_spectral(
        lead_left, lead_right, dev, pot_kwargs['gap_val'],
        **prog_kwargs)

    print_out('Time to calculate spectral data : ' +
        time_elapsed_str(start_spectral - time.time()))
        
    save_spectral(spec_data, dev, pot, k_num, en_num)

    #plot_transmission_test(lead_left, lead_right, dev)


################################################################################


def __main__():

    # Use the tb_utility module to print the current date to our output file

    file_out_name = '../out_ac_g3_true.txt'

    if os.path.isfile(file_out_name):

        sys.exit('\n\tOutput file \'' + file_out_name + '\' already exists')

    else:

        create_out_file(file_out_name)

    ################################ POTENTIAL #################################

    pot_type = 'well'

    # Dictionary of paramters used to define the potential
    pot_kwargs = {
        'gap_val'       :   0.1,    # 100meV delta0
        'offset'        :   0,      # 0eV
        'well_depth'    :   -0.02,  # -20meV U0
        'gap_relax'     :   0.3,    # dimensionless beta
        'channel_width' :   850,    # 850A L
        }

    ################################ SUPERCELL #################################

    cell_func = BLG_cell # Cell function to use (BLG vs MLG)

    cell_num = 1

    orientation = 'ac'

    # Dictionary of paramters used to define the dev (no potential)
    dev_kwargs = {
        'is_gamma_3'    :   True    # On/off gamma 3 coupling in the BLG system
        }

    ################################ SIMULATION ################################

    # Parameters related to the running of the programme itself
    prog_kwargs = {
        'is_main_task'  :   False,          # False parallelise over fewer cores
        'max_cores'     :   12,             # Max cores to parallelise over
        'is_parallel'   :   True            # If True, parallelise
        }

    int_norm = [0,1,0] # Vector normal to the potential interface

    shift = 0 # Amout to shifft the interface by (default is zero)
    # Calculate the location of the interface given cell_num etc.

    int_loc_y = cell_num * np.dot(
        cell_func(index = 0, orientation = orientation, **dev_kwargs
            ).lat_vecs_sc[1], int_norm) + shift

    int_loc = [0, int_loc_y, 0] # Vector version of int_loc_y


    # Create the potential
    pot = potential(pot_type, int_loc, int_norm, **pot_kwargs)

    ############################################################################

    is_finite = True

    ############################################################################
    start = time.time()

    if is_finite:

        sys_finite(cell_func, orientation, cell_num, pot,
            pot_kwargs, dev_kwargs, prog_kwargs)

    else:

        sys_infinite(cell_func, orientation, cell_num, pot, 
            pot_kwargs, dev_kwargs, prog_kwargs)

    print_out('Complete. Total elapsed time : ' +
        time_elapsed_str(start_spectral - time.time()))


if __name__ == '__main__':

    try:

        __main__()

    except Exception as e:

        print_out('Caught exception in tb_main.py')

        print_out( ''.join( traceback.format_exception( *sys.exc_info() ) ) )

        raise

"""
I can also pull out the LDOS of certain atoms at the edge for each k - value
and plot this for a range of energies. This would result in a spectral plot that
should represent the band structure.

"""

"""
Change cell_num to be the total number of cells in the dev, not the number of
cells on either side. Must change this across ALL modules

"""

"""
How to get rid of the code's dependence upon cell_num as a parameter ? 

"""

"""
Code to plot the simple lattice structures and unit cells 

xyz_tmp = np.array([cell.xyz for cell in dev.cells[0:2]]
    ).reshape((len(dev.cells[0].xyz) * 2, 3))

data_save(xyz_tmp, dir_ext, 'xyz_simple_', parameter_str)
data_save(dev.lat_vecs_sc, dir_ext, 'lat_vec_sc_simple_', parameter_str)

"""

"""
Method to save both bands and eigenfunctions

bnd_struc = dev.get_band_data_full(gap_val, kvals = 1000)

# k, kval or eig, val vs vec

print(bnd_struc[500][0])

y_vals = [xyz[1] for xyz in dev.xyz]

a = bnd_struc[500][1][1][:,38]
print(bnd_struc[500][1][0][38])
plt.plot(y_vals, np.abs(a)**2, 'ko')
plt.show()

a = bnd_struc[500][1][1][:,39]
print(bnd_struc[500][1][0][39])
plt.plot(y_vals, np.abs(a)**2, 'ko')
plt.show()

#for i in range(len(bnd_struc[500][1][1])):
#   a = bnd_struc[0][1][1][:,i]
#   plt.plot(np.abs(a)**2)
#   plt.show()
"""