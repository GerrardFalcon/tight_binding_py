import numpy as np

from numpy.linalg import inv

import matplotlib.pyplot as plt

import copy as cpy

from graphene_supercell import *
from potentials import potential
from utility import print_out


class lead:

    def __init__(self, cell, lead_direction):

        self.cell = cpy.deepcopy(cell)

        # Define lead direction as left or right. This determines which coupling
        # matrix acts forwards and which acts backwards. Must be 'L' or 'R'
        self.lead_dir = lead_direction

        # Define the `forward' coupling matrix based on lead direction
        if self.lead_dir == 'L':

            self.V = np.conj(self.cell.get_V()).T

        elif self.lead_dir == 'R':

            self.V = self.cell.get_V()

        else:

            err_str = str(self.__class__.__name__) +'(): Lead direction may be \
                either \'L\' or \'R\', not ' + str(self.lead_dir)

            print_out(err_str)

            raise ValueError(err_str)


    def get_GF(self, kdx, energy, small = 1E-6, is_return_SE = False):
        """
        Uses the Rubio Sancho iterative method to approximate the surface
        Green's Function of the lead

        Takes as arguments the Hamiltonian describing a single cell and the
        matrix that couples one cell forward to the next cell
        """

        # Initialise starting values for first iteration
        a_new = np.conj(self.V).T

        b_new = self.V

        H_new = self.cell.get_H(kdx)

        Hs_new = self.cell.get_H(kdx)

        g_new = inv(np.identity(H_new.shape[0]) * (energy + small* 1j) - H_new )

        # Define a counting parameter to make sure we don't end up in an
        # infinite loop
        count = 0

        # Define a matrix of small values to check convergence against
        target = np.ones(H_new.shape, np.complex128) * 1E-10

        while True if count < 1 else (np.abs(Hs_new - Hs_old) > target).any():

            count += 1

            # Set previous results to be the old ones
            a_old = a_new

            b_old = b_new

            H_old = H_new

            Hs_old = Hs_new

            g_old = g_new

            # Update to new values for this iteration
            a_new = a_old @ g_old @ a_old

            b_new = b_old @ g_old @ b_old

            H_new = H_old + a_old @ g_old @ b_old + b_old @ g_old @ a_old

            Hs_new = Hs_old + a_old @ g_old @ b_old

            g_new = inv(
                np.identity(H_new.shape[0]) * (energy + small*1j) - H_new)

            # Break loop based on upper ceiling for count
            if count > 100:

                print_out('Exceeded maximum count in make_lead(...)')

                break

        # Calculate the greens function for the surface
        gs = inv(np.identity(Hs_new.shape[0]) * (energy + small * 1j) - Hs_new)

        # return the greens function and self energy if is_return_SE is True
        if is_return_SE:

            return gs, np.conj(self.V).T @ gs @ self.V

        else:

            return gs


    ############################################################################
    ####              Functions to make and test the leads                  ####
    ############################################################################

def make_lead(device, lead_dir, cell_func, latt_type, pot = potential(),
    **kwargs):
    """
    Wrapper for the `lead' class that makes it easier to generate the leads each
    time

    """

    if lead_dir == 'L':

        idx = - device.cell_num[0] - 1

    elif lead_dir == 'R':

        idx = device.cell_num[1] + 1

    else:
        err_str = 'Lead direction may be either \'L\' or \'R\', not ' + \
            str(lead_dir)

        print_out(err_str)

        raise ValueError(err_str)

    if type(device.orientation) is str:

        lead_cell = cell_func(idx, latt_type, **kwargs)

    # Deal with the case where the orientation is arbitrary

    else:

        # Adapt the input to work with the graphene_supercell module
        ori = [int(idx) for idx in device.orientation[0]]

        # Remove orientation from kwargs so that it is not passed twice
        del kwargs['orientation']

        lead_cell = cell_func(idx, latt_type, ori, **kwargs)

    lead_cell.set_energy(pot)

    return lead(lead_cell, lead_dir)


def get_t(kdx, energy, small):
    """
    Function which calculates the transmission for a single cell between two
    leads to check if the leads are being set up correctly. No potential

    """

    # LEFT
    cell_left = graphene_cell(index = -10, orientation = 'zz')

    lead_left = lead(cell_left, 'L')

    lead_GF_left, lead_SE_left = lead_left.get_GF(
        kdx, energy, small, is_return_SE = True)

    # RIGHT
    cell_right = graphene_cell(index = +10, orientation = 'zz')

    lead_right = lead(cell_right, 'R')

    lead_GF_right, lead_SE_right = lead_right.get_GF(
        kdx, energy, small, is_return_SE = True)

    cell_device = graphene_cell(index = 0, orientation = 'zz')

    # Calculate the device GF
    H_D = cell_device.get_H(kdx)

    g_D = inv(np.identity(H_D.shape[0]) * (energy + small * 1j) \
        - H_D - lead_SE_left - lead_SE_right)

    # Calculate the gamma factors (imag part of self energy matrices) for the
    # left and right leads
    gamma_L = 1j * (lead_SE_left - np.conj(lead_SE_left).T)
    gamma_R = 1j * (lead_SE_right - np.conj(lead_SE_right).T)

    # Calculate transmission given these two terms and the device GF
    trans = np.trace(gamma_L @ np.conj(g_D).T @ gamma_R @ g_D)

    return trans


def plot_transmission():
    """
    Function which plots the transmission for a range of energies after
    averaging over k between plus and minus pi
    """
    small = 1E-6

    k_num = 200

    kdx_list = np.linspace(-np.pi, np.pi, k_num)

    en_list = np.linspace(-1, 1, 200)

    data = []

    for en in en_list:

        k_av = np.sum(
            np.array([get_t(kdx, en, small) for kdx in kdx_list])) / k_num

        data.append([en, k_av])

    data = np.array(data)

    plt.plot(data[:, 0].real, data[:, 1].real,'k-')
    
    plt.show()


def __main__():
    plot_transmission()


if __name__ == '__main__':
    __main__()