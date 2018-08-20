import numpy as np

import matplotlib.pyplot as plt

from numpy.linalg import inv, norm

from graphene_supercell import *
from tb_utility import print_out
from devices import device, device_finite, plot_xyz
from leads import make_lead
from potentials import potential


# NOTE :    May need to check indexing on R_to_L and L_to_R to make sure that
#           the hoppings between cells are correctly indexed. The code should
#           work fine at the minute since all cells have the same coupling
#           between them


def R_to_L(lead_left, lead_right, device, kdp, energy, small = 1E-6):

    atno = len(device.cells[0].xyz)

    # Calculate the Self Energy for the right lead
    SE_right_lead = np.conj(device.cells[-1].get_V()).T @ \
        lead_right.get_GF(kdp, energy, small) @ device.cells[-1].get_V()

    # Caclulate the Greens Function for the furthest right device cell which
    # is attached to the lead
    GF_part_nn = inv( np.identity(atno) * (energy + small * 1j) -
        device.cells[-1].get_H(kdp) - SE_right_lead ).reshape((1, atno, atno))

    # Iterate over the device, appending each successive GF to the array
    for i in range(1, len(device.cells) - 1):

        # Calculate the self energy all blocks up to this point. Because we will
        # be prepending arrays to the array of arrays, the GF for the previous
        # cell will always be the first in the list
        SE = np.conj(device.cells[-(1 + i)].get_V()).T @ GF_part_nn[0] @\
            device.cells[-(1 + i)].get_V()

        # Calculate the value of the Greens Function for the current cell given
        # the SE of the cumulative combined system to this point
        GF_tmp = inv( np.identity(atno) * (energy + small * 1j) -
            device.cells[-(1 + i)].get_H(kdp) - SE ).reshape((1, atno, atno))

        # Prepend the new GF to the array of arrays so that when the iterations
        # are complete, it is ordered from left to right
        GF_part_nn = np.concatenate((GF_tmp, GF_part_nn), axis = 0)

    # The calculation for the final cell must include both the self energy of
    # everything to the right (calculated so far) and also the left lead
    # Calculate the left lead SE
    SE_left_lead = device.cells[0].get_V() @ \
        lead_left.get_GF(kdp, energy, small) @ \
        np.conj(device.cells[0].get_V()).T

    # Calculate the device SE
    SE_device = np.conj(device.cells[0].get_V()).T @ GF_part_nn[0] @ \
        device.cells[0].get_V()

    # Make the Green's Function for the final cell including both SE's
    GF_tmp = inv( np.identity(atno) * (energy + small * 1j) -
        device.cells[0].get_H(kdp) - SE_left_lead - SE_device
        ).reshape((1, atno, atno))

    # Prepend the final GF to the array and return it
    return np.concatenate((GF_tmp, GF_part_nn), axis = 0)


def L_to_R(lead_left, lead_right, device, GF_part_nn, kdp, energy, small):
    # Assign the first element of both fully connected GFs, since the final
    # element of the partially connected GF is connected both forwards and
    # backwards. Indexes are from (right index) and to (left index)
    atno = len(device.cells[0].xyz)

    GF_full_nn = GF_part_nn[0].reshape((1, atno, atno))

    GF_full_n1 = GF_part_nn[0].reshape((1, atno, atno))

    # Iterate over the remaining cells to complete the fully connected GFs
    for i in range(1, len(device.cells)):

        GF_full_n1 = np.concatenate((GF_full_n1,

            (GF_part_nn[i] @ device.cells[i].get_V() @ GF_full_n1[-1]

                ).reshape((1, atno, atno))), axis = 0)

        GF_full_nn = np.concatenate((GF_full_nn,
            (GF_part_nn[i] + GF_part_nn[i] @ device.cells[i].get_V() @ \
            GF_full_nn[-1] @ np.conj(device.cells[i].get_V()).T @ \
            GF_part_nn[i]).reshape((1, atno, atno))), axis = 0)

    # Return the fully connected GFs
    return GF_full_nn, GF_full_n1


def double_folding(lead_left, lead_right, device, kdp, energy, small = 1E-6):
    # Get the partially connected Green's Function from right to left, which is
    # returned ordered so that the fully connected final cell is the first
    # element of the returned array along axis 0
    GF_part_nn = R_to_L(lead_left, lead_right, device, kdp, energy, small)

    # Use the partially connected GF to calculate the fully connected GF
    GF_full_nn, GF_full_n1 = L_to_R(lead_left, lead_right, device, GF_part_nn,
        kdp, energy, small)

    # Return both the fully connected 
    return GF_full_nn, GF_full_n1


def get_transmission(lead_left, lead_right, device, kdp, energy, small = 1E-6):
    # Get the SE of the left lead
    lead_left_GF, lead_left_SE = lead_left.get_GF(
        kdp, energy, small, is_return_SE = True)

    # Get the GF's for the rest of the device
    GF_part_nn = R_to_L(lead_left, lead_right, device, kdp, energy, small)

    # Calculate the SE for the cell to the right of the fully connected cell
    lead_right_SE = np.conj(device.cells[1].get_V()).T @ GF_part_nn[1] @ \
        device.cells[1].get_V()

    # Calculate the gamma factors (imag part of self energy matrices) for the
    # left and right leads
    gamma_L = 1j * (lead_left_SE - np.conj(lead_left_SE).T)
    gamma_R = 1j * (lead_right_SE - np.conj(lead_right_SE).T)

    # Select the fully connected GF
    g_D = GF_part_nn[0]

    # Return transmission given these two terms and the device GF
    return np.trace(gamma_L @ np.conj(g_D).T @ gamma_R @ g_D)


def get_LDOS_k_average(lead_left, lead_right, device, kdp, energy,
    small = 1E-6, is_edges = True):
    # Get the recursively calculated GF for a range of kdp

    cellno = len(device.cells)

    atno = len(device.cells[0].xyz)

    if is_edges:

        cell_rng = [0, cellno]

    else:

        pad = 5

        cell_rng = [pad, cellno - pad]

    GF_full_nn, GF_full_n1 = double_folding(lead_left, lead_right, device, kdp,
        energy, small)

    LDOS_data = np.array([[np.append(device.cells[j].xyz[i,:2],
        (-1 / np.pi) * GF_full_nn[j,i,i].imag) for i in range(atno)]\
        for j in range(cell_rng[0], cell_rng[1])]).reshape(
        ((cell_rng[1] - cell_rng[0]) * atno, 3))

    reps = 20

    LDOS_data = np.array(
        [LDOS_data + i * device.lat_vecs_sc[0] for i in range(reps)]).reshape(
        reps * cellno * atno, 3)

    return LDOS_data


def __main__():
    pass

if __name__ == '__main__':
    __main__()
