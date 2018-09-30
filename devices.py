import numpy as np
from numpy.linalg import norm as nrm

import matplotlib.pyplot as plt
from matplotlib import path

from potentials import potential
from graphene_supercell import *
from utility import print_out

import sys, traceback, copy

class device:

    def __init__(self, cell_num, orientation, latt_type, cell_func,
        pot = potential(), **kwargs):

        self.latt_type = latt_type

        self.cell_func = cell_func

        self.cell_num = cell_num

        self.keywords = kwargs

        # Generate each individual unit cell within the device by making a call
        # to the relevant function / class in graphene_supercell
        self.cells = np.array([
            self.cell_func(idx, latt_type, orientation, **self.keywords)
            for idx in range(-cell_num[0], cell_num[1])])

        self.lat_vecs_sc = self.cells[0].lat_vecs_sc

        self.orientation = self.cells[0].orientation

        self.rot_angle = self.cells[0].rot_angle

        if type(self.orientation) is str:

            if self.orientation == 'ac':

                for cell in self.cells:

                    # If armchair orientation, shift the lattice a little so
                    # that the domain wall lies such that the interface is
                    # geometrically symmetric
                    cell.xyz -= np.array([0, 2.46 / 4, 0])

        self.set_energy(pot)


    def get_xyz(self):

        return np.concatenate([cell.xyz for cell in self.cells])


    def get_lat_vecs_sc(self):
        """ Return lattice vectors for the fully combined supercell """

        lvsc = copy.copy(self.lat_vecs_sc)

        lvsc[1] *= sum(self.cell_num)

        return lvsc


    def get_sublat(self):

        return np.concatenate([cell.sublat for cell in self.cells])


    def get_energy(self):

        return np.concatenate([cell.energy for cell in self.cells])


    def set_energy(self, pot_func = None):

        if pot_func is None:

            for cell in self.cells:

                cell.set_energy()

        else:

            for cell in self.cells:

                cell.set_energy(pot_func)


    ################################## UTILITY #################################


    def get_req_params(self):
        """
        Returns all required paramters for the device, including both device and
        supercell requirements

        """

        # Params required for the periodic device
        req_dict = {'cell_num'  :   self.cell_num}

        # Update the required params to include the no. of cells in the stripe
        if str(self.cell_func.__name__) == 'stripe':

            if 'stripe_len' in self.keywords:

                req_dict.update({'stripe_len' : self.keywords['stripe_len']})

        # Update with cell requirements and return
        return {**self.cells[0].get_req_params(), **req_dict}


    #####################        Plotting functions      #######################

    def plot_interface(self, int_loc):

        if sum(self.cell_num) > 1:

            cells_L = self.cell_num[0]

            xyz = np.concatenate((self.cells[cells_L - 1].xyz,
                self.cells[cells_L].xyz), axis = 0)

            sublat = np.concatenate((self.cells[cells_L - 1].sublat,
                self.cells[cells_L].sublat), axis = 0)

        else:

            xyz = self.cells[0].xyz
            sublat = self.cells[0].sublat

        ax = make_plot_xyz(xyz, sublat)

        bnd = np.array([
            int_loc - self.cells[0].lat_vecs_sc[0],
            int_loc + self.cells[0].lat_vecs_sc[0]])

        ax.plot(bnd[:,0], bnd[:,1],'k-')

        plt.show()


    def plot_energies(self):

        data_list = np.array([[[cell.sublat[i], cell.xyz[i,1], cell.energy[i]]
            for i in range(len(cell.xyz))] for cell in self.cells]
            ).reshape(len(self.cells) * len(self.cells[0].xyz), 3)

        fig = plt.figure()

        ax = fig.add_subplot(111)

        ax.plot(
            data_list[data_list[:,0] == 0, 1],
            data_list[data_list[:,0] == 0, 2], 'r.')

        ax.plot(
            data_list[data_list[:,0] == 1, 1],
            data_list[data_list[:,0] == 1, 2], 'k.')

        plt.show()


class device_finite(device):
    """
    Works with a finite system. Need to shift the structure so that the
    interface lies symmetrically in the armchair case
    """

    def __init__(self, pot = potential(), **kwargs):

        super().__init__(pot = pot, **kwargs)

        is_ac = False

        if type(self.orientation) is str:

            if self.orientation == 'ac':

                is_ac = True

        self.xyz = self.get_xyz()

        self.lat_vecs_sc = self.get_lat_vecs_sc()

        self.sublat = self.get_sublat()

        self.energy = self.get_energy()

        # Save only one reference cell so that we are ont saving all of the data
        # twice (need one for pulling out values for later computation)
        self.cell_tmp = self.cells[0]

        del self.cells

        # Define the y-position of the interface for later use
        self.int_loc = pot.int_loc

        if is_ac:

            # If armchair orientation, shift the lattice a little so that the
            # interface lies such that the interface is geometrically symmetric
            #self.xyz -= np.array([0, 2.46/4, 0])
            # define a small tolerance shift to eliminate atoms directly at 0
            tol = 1E-2

            # now exclude cells outside of our desired range
            is_in = np.logical_and(
                self.xyz[:,1] >= (
                    - self.cell_num[0] * self.lat_vecs_sc[1])[1] - tol,
                self.xyz[:,1] <= (
                    self.cell_num[1] * self.lat_vecs_sc[1])[1] + tol)

            self.xyz = self.xyz[is_in]

            self.energy = self.energy[is_in]

            self.sublat = self.sublat[is_in]

        # If energies are not set in the infinite system, we would set them here


    def set_energy_finite(self, pot = potential()):

        self.energy = pot.pot_func(self.xyz, self.sublat)


    def get_sys_hamiltonian(self, kdp = 0, kdp_perp = 0):
        """
        Passes xyz, sublat, energy and lat_vecs_sc to the relevant cell

        """
        

        # Make a dictionary of the required values. Create 'is_periodic'
        # assuming initially that it is True by default
        sys_data_list = {'xyz' : self.xyz, 'sublat' : self.sublat,
            'energy' : self.energy, 'lat_vecs_sc' : self.lat_vecs_sc}

        is_wrap_finite_tmp = False      # Default behaviour

        # Check if this system is periodic in the 'transport' direction.
        if 'is_wrap_finite' in self.keywords.keys():

            is_wrap_finite_tmp = self.keywords['is_wrap_finite']

        # Get the Hamiltonian
        ham = self.cell_tmp.get_H(
            kdp, kdp_perp, sys_data_list, is_wrap_finite = is_wrap_finite_tmp)

        # # Check Hamiltonian is Hermitian (within a numerical tolerance)
        if np.allclose(ham, np.conj(ham).T): # atol = 1E-8

            return ham    

        else:

            err_str = str(self.__class__.__name__) + '(): Hamiltonian is not \
                Hermitian. Try again'

            print_out(err_str)

            # If not, give a value error and print required inputs
            raise ValueError(err_str)


    ################################## UTILITY #################################


    def get_req_params(self):
        """
        Returns all required paramters for the device, including both device and
        supercell requirements

        """

       # Params required for the periodic device
        req_dict = {'cell_num'  :   self.cell_num}

        if 'is_wrap_finite' in self.keywords.keys():

            req_dict.update(
                {'is_wrap_finite'   :   self.keywords['is_wrap_finite']})

        # Update with cell requirements and return
        return {**self.cell_tmp.get_req_params(), **req_dict}


    def get_band_data(self, kvals = 1000):
        """ Returns just the eigenvalues """

        kdp_list = np.linspace(-np.pi, np.pi, kvals)

        eig_vals = np.array([np.linalg.eigvalsh(self.get_sys_hamiltonian(kdp))
            for kdp in kdp_list])

        is_nonfinite_imag = (np.abs(eig_vals.imag) > 1E-8).any()

        if is_nonfinite_imag:

            print_out(
                'The imaginary part of an eigenvalue array element is non-zero.\
                \n\n\tSomething weird is going on.')

        else:

            eig_vals = np.array([[kdp_list[i], eig_vals[i,j]]
                for i in range(len(kdp_list)) for j in range(len(eig_vals[0]))])

            return eig_vals


    def get_eig_sys(self, kdp = 0):
        """ Returns [k, eigenvalues] for a given wavenumber """

        try:

            eig_sys = np.linalg.eigh(self.get_sys_hamiltonian(kdp))

        except Exception as e:

            print_out('Caught exception in devices.py, get_eig_sys()')

            print_out( ''.join( traceback.format_exception(*sys.exc_info()) ) )

            raise

        return [kdp, eig_sys]


    def get_band_data_full(self, kvals = 1000):
        """ Returns the bands along with the wavefunctions """

        kdp_list = np.linspace(-np.pi, np.pi, kvals)

        eig_sys = [[kdp, np.linalg.eigh(self.get_sys_hamiltonian(kdp))]
            for kdp in kdp_list]

        return eig_sys


    ####################        Plotting functions      ########################


    def plot_interface(self, int_loc):

        path_vec_list = np.array((self.lat_vecs_sc[0],  2 * self.lat_vecs_sc[1],
            -self.lat_vecs_sc[0], -2 * self.lat_vecs_sc[1]))

        # start position + small amount to correct the numebr of points selected
        cell_start = int_loc + 0.01 - self.lat_vecs_sc[1]

        cell_corners = np.array(
            [cell_start + np.sum(path_vec_list[0:i], axis = 0)
            for i in range(len(path_vec_list))])

        p = path.Path(cell_corners[:,0:2])

        is_in = p.contains_points(self.xyz[:,0:2])

        #is_in = np.logical_and(
        #        self.xyz[:,1] >= int_loc - 1.1 * self.lat_vecs_sc[1,1],
        #        self.xyz[:,1] <= int_loc + 1.1 * self.lat_vecs_sc[1,1])

        ax = make_plot_xyz(self.xyz[is_in], self.sublat[is_in])

        bnd = np.array([
            int_loc - self.lat_vecs_sc[0],
            int_loc + self.lat_vecs_sc[0]])

        ax.plot(bnd[:,0], bnd[:,1],'k-')

        plt.show()


    def plot_energies(self):

        fig = plt.figure()

        ax = fig.add_subplot(111)

        ax.plot(
            self.xyz[self.sublat == 0, 1], self.energy[self.sublat == 0], 'r.')

        ax.plot(
            self.xyz[self.sublat == 1, 1], self.energy[self.sublat == 1], 'k.')

        plt.show()
        

################################################################################
####                    Functions to test the device                        ####
################################################################################


def make_plot_xyz(xyz,sublat):

    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.scatter(*zip(*xyz[sublat == 0, 0:2]), c = 'r', label = 'A')
    ax.scatter(*zip(*xyz[sublat == 1, 0:2]), c = 'k', label = 'B')

    pd = np.array([-1,1])

    lims = np.array(
        [np.min(xyz[:,0]),np.max(xyz[:,0]), np.min(xyz[:,1]), np.max(xyz[:,1])])

    ax.set_xlim(lims[0:2] + pd)
    ax.set_ylim(lims[2:] + pd)

    ax.legend(loc='upper right');

    ax.set_aspect('equal', adjustable = 'box')

    return ax


def plot_xyz(xyz,sublat):

    ax = make_plot_xyz(xyz, sublat)

    plt.show()


def unit_vec(vec):

    return np.array(vec) / np.linalg.norm(np.array(vec))


def __main__():

    pass


if __name__ == "__main__":

    __main__()