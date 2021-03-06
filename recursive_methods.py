import numpy as np

import matplotlib.pyplot as plt

from numpy.linalg import inv, norm

from graphene_supercell import *
from devices import device, device_finite, plot_xyz
from leads import make_lead
from potentials import potential


# NOTE :    May need to check indexing on R_to_L and L_to_R to make sure that
#           the hoppings between cells are correctly indexed. The code should
#           work fine at the minute since all cells have the same coupling
#           between them


def R_to_L_SINGLE_CELL(lead_left, lead_right, dev, kdp, energy, small = 1E-6):
    """ Used when the device consists of only a single cell """

    atno = len(dev.cells[0].xyz)

    # Calculate the Self Energy for the right lead
    SE_right_lead = np.conj(dev.cells[0].get_V()).T @ \
        lead_right.get_GF(kdp, energy, small) @ dev.cells[0].get_V()

    # Calculate the left lead SE
    SE_left_lead = dev.cells[0].get_V() @ \
        lead_left.get_GF(kdp, energy, small) @ \
        np.conj(dev.cells[0].get_V()).T

    # Make the Green's Function for the final cell including both SE's
    GF = inv( np.identity(atno) * (energy + small * 1j) -
        dev.cells[0].get_H(kdp) - SE_left_lead - SE_right_lead
        ).reshape((1, atno, atno))

    return GF


def R_to_L_RIGHT(lead_left, lead_right, dev, kdp, energy, small = 1E-6):
    """ Calculate the Greens function of the cell touching the right lead """

    atno = len(dev.cells[0].xyz)

    # Calculate the Self Energy for the right lead
    SE_right_lead = np.conj(dev.cells[-1].get_V()).T @ \
        lead_right.get_GF(kdp, energy, small) @ dev.cells[-1].get_V()

    # Caclulate the Greens Function for the furthest right device cell which
    # is attached to the lead
    GF_part_nn = inv( np.identity(atno) * (energy + small * 1j) -
        dev.cells[-1].get_H(kdp) - SE_right_lead
        ).reshape((1, atno, atno))

    return GF_part_nn


def R_to_L_MID(i, lead_left, lead_right, dev, kdp, energy, GF_part_nn,
    small = 1E-6):
    """ Calculate the Greens function of the cell touching the left lead """

    atno = len(dev.cells[0].xyz)

    # Calculate the self energy all blocks up to this point. Because we will be
    # prepending arrays to the array of arrays, the GF for the previous cell
    # will always be the first in the list
    SE = np.conj(dev.cells[-(1 + i)].get_V()).T @ \
        GF_part_nn[0] @ dev.cells[-(1 + i)].get_V()

    # Calculate the value of the Greens Function for the current cell given the
    # SE of the cumulative combined system to this point
    GF_tmp = inv( np.identity(atno) * (energy + small * 1j) -
        dev.cells[-(1 + i)].get_H(kdp) - SE).reshape((1, atno, atno))

    return GF_tmp


def R_to_L_LEFT(lead_left, lead_right, dev, kdp, energy, GF_part_nn,
    small = 1E-6):
    """ Calculate the Greens function of the cell touching the left lead """

    atno = len(dev.cells[0].xyz)

    # Calculate the left lead SE
    SE_left_lead = dev.cells[0].get_V() @ \
        lead_left.get_GF(kdp, energy, small) @ \
        np.conj(dev.cells[0].get_V()).T

    # Calculate the device SE
    SE_dev = np.conj(dev.cells[0].get_V()).T @ GF_part_nn[0] @ \
        dev.cells[0].get_V()

    # Make the Green's Function for the final cell including both SE's
    GF_tmp = inv( np.identity(atno) * (energy + small * 1j) -
        dev.cells[0].get_H(kdp) - SE_left_lead - SE_dev
        ).reshape((1, atno, atno))

    return GF_tmp


def R_to_L(lead_left, lead_right, dev, kdp, energy, small = 1E-6):

    if len(dev.cells) == 1:

        # If there is only one cell, we include the self energy of both the left
        # and right leads in the calculation

        GF_full = R_to_L_SINGLE_CELL(lead_left, lead_right, dev, kdp, energy,
            small)

        return GF_full

    else:

        # Caclulate the Greens Function for the furthest right device cell which
        # is attached to the lead
        GF_part_nn = R_to_L_RIGHT(lead_left, lead_right, dev, kdp, energy,
            small)

        if len(dev.cells) > 2:

            # Iterate over the device, appending each successive GF to the array
            for i in range(1, len(dev.cells) - 1):

                GF_tmp = R_to_L_MID(i, lead_left, lead_right, dev, kdp, energy,
                    GF_part_nn, small)

                # Prepend the new GF to the array of arrays so that when the
                # iterations are complete, it is ordered from left to right
                GF_part_nn = np.concatenate((GF_tmp, GF_part_nn), axis = 0)

        # The calculation for the final cell must include both the self energy
        # of everything to the right (calculated so far) and also the left lead

        # Make the Green's Function for the final cell including both SE's
        GF_tmp = R_to_L_LEFT(lead_left, lead_right, dev, kdp, energy,
            GF_part_nn, small)

        # Prepend the final GF to the array and return it
        return np.concatenate((GF_tmp, GF_part_nn), axis = 0)


def L_to_R(lead_left, lead_right, dev, GF_part_nn, kdp, energy, small):
    # Assign the first element of both fully connected GFs, since the final
    # element of the partially connected GF is connected both forwards and
    # backwards. Indexes are from (right index) and to (left index)
    atno = len(dev.cells[0].xyz)

    GF_full_nn = GF_part_nn[0].reshape((1, atno, atno))

    GF_full_n1 = GF_part_nn[0].reshape((1, atno, atno))

    # Iterate over the remaining cells to complete the fully connected GFs
    for i in range(1, len(dev.cells)):

        GF_full_n1 = np.concatenate(
            (GF_full_n1, (GF_part_nn[i] @ dev.cells[i].get_V() @ GF_full_n1[-1]
                ).reshape((1, atno, atno))), axis = 0)

        GF_full_nn = np.concatenate((GF_full_nn,
            (GF_part_nn[i] + GF_part_nn[i] @ dev.cells[i].get_V() @ \
            GF_full_nn[-1] @ np.conj(dev.cells[i].get_V()).T @ \
            GF_part_nn[i]).reshape((1, atno, atno))), axis = 0)

    # Return the fully connected GFs
    return GF_full_nn, GF_full_n1


def double_folding(lead_left, lead_right, dev, kdp, energy, small = 1E-6):
    # Get the partially connected Green's Function from right to left, which is
    # returned ordered so that the fully connected final cell is the first
    # element of the returned array along axis 0
    GF_part_nn = R_to_L(lead_left, lead_right, dev, kdp, energy, small)

    # Use the partially connected GF to calculate the fully connected GF
    GF_full_nn, GF_full_n1 = L_to_R(lead_left, lead_right, dev, GF_part_nn,
        kdp, energy, small)

    # Return both the fully connected 
    return GF_full_nn, GF_full_n1


def __main__():
    pass

if __name__ == '__main__':
    __main__()