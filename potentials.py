import numpy as np
from numpy.linalg import norm as nrm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utility import print_out


class potential:
    """
    Class containing a range of different potential profiles that can be applied
    to my lattices.
    """

    def __init__(self, pot_type_IN = None, int_loc_IN = [0,0,0], \
        int_norm_IN = [0,1,0], **kwargs):

        self.pot_type = pot_type_IN

        # A point on the plane that defines the interface
        self.int_loc = np.array(int_loc_IN)

        # The norm to the plane that defines the interface
        self.int_norm = np.array(int_norm_IN) / nrm(np.array(int_norm_IN))

        self.pot_params = kwargs

        # Check that all required keyword arguments are provided
        if self._kwargs_check():

            # Then generate potential function based on pot_type
            if self.pot_type is None:

                def pot(xyz, sublat):

                    return self.equal_onsite(xyz, sublat, onsite = 0)

            elif self.pot_type == 'step':

                def pot(xyz, sublat):

                        return self.pot_func_step(xyz, sublat,**self.pot_params)

            elif self.pot_type == 'tanh':

                def pot(xyz, sublat):

                    return self.pot_func_tanh(xyz, sublat, **self.pot_params)

            elif self.pot_type == 'well':

                def pot(xyz, sublat):

                    return self.pot_func_BLG_well(xyz, sublat,**self.pot_params)

            else:

                sys.exit('Invalid potential choice in potentials.py')

            self.pot_func = pot


    def pot_func_BLG_well(self, xyz, sublat, gap_val, offset, well_depth,\
        channel_depth, channel_width, channel_length, gap_relax, channel_relax,\
        is_const_channel):

        # Calculate 1 / cosh(x / L) where 'x' is in the direction perpendicular
        # to the interface

        sech_perp = np.reciprocal(np.cosh(np.dot(
            xyz - self.int_loc, self.int_norm) / channel_width))

        if is_const_channel:

            # U(x) as per Angelika's paper (const in x)
            u_xy = well_depth * sech_perp

        else:

            # U(x,y) with varying channel depth. 'y' denotes direction along the
            # channel

            # unit vector parallel to the channel along positive axis
            int_par = np.cross([0,0,1], self.int_norm)

            # Calculate the position of the left/right slopes
            yL = self.int_loc - 0.5 * channel_length * int_par
            yR = self.int_loc + 0.5 * channel_length * int_par            

            # Function for 'left' slope in the channel
            tanhL = np.tanh(
                (np.dot(xyz - self.int_loc - yL, int_par)) / channel_relax)

            # Function for 'right' slope in the channel
            tanhR = np.tanh(
                (np.dot(xyz - self.int_loc - yR, int_par)) / channel_relax)

            # U(x,y) in full
            u_xy = (channel_depth + 0.5 * (well_depth - channel_depth) * (
                tanhL - tanhR)) * sech_perp

        half_delta = 0.5 * gap_val * (1 - gap_relax * sech_perp)

        # Initialise an array of zeros to be filled
        energies = np.zeros_like(sublat, np.float64)

        # Fill lower layer
        energies[xyz[:,2] == 0] = (u_xy - half_delta)[xyz[:,2] == 0]

        # Fill upper layer
        energies[xyz[:,2] != 0] = (u_xy + half_delta)[xyz[:,2] != 0]

        return energies


    def pot_func_step(self, xyz, sublat, gap_val, offset):

        index = self._sublat_index(sublat)

        return offset + gap_val * index * (2 * np.heaviside(np.dot(
            xyz - self.int_loc, self.int_norm), 0.5) - 1)


    def pot_func_tanh(self, xyz, sublat, gap_val, offset):
        # Scaling of the range over which tanh varies greatly from its limit
        scaling = 2

        index = self._sublat_index(sublat)

        return offset + gap_val * index * np.tanh((np.dot(
            xyz - self.int_loc, self.int_norm)) / scaling)


    def equal_onsite(self, xyz, sublat, onsite = 0):

        return np.array([onsite] * len(xyz))


    ####                             UTILITY                                ####


    def _kwargs_check(self, is_get_required = False):
        """
        Checks that the arguments passed to potential are correct for the chosen
        potential type
        
        """

        required = []

        # Generate a list of the required inputs in each case
        if self.pot_type == 'step' or self.pot_type == 'tanh':

            required = ['gap_val', 'offset']

        elif self.pot_type == 'well':

            required = ['gap_val', 'offset', 'well_depth', 'gap_relax', \
                'channel_width', 'channel_depth', 'channel_length', \
                'channel_relax']

        # Check if all inputs are provided
        if all(item in self.pot_params for item in required):

            if is_get_required:

                return True, required

            else:

                return True

        else:

            err_str = str(self.__class__.__name__) + '(): Require more \
                information for potential type : ' + str(self.pot_type) + \
                '\nInputs required are : \n' + str(required)

            print_out(err_str)

            # If not, give a value error and print required inputs
            raise ValueError(err_str)


    def get_req_params(self):
        """ Return a list of params required to make the selcted potential """

        if self.pot_type is None:

            return {'pot_type'  :   'NONE'}

        else:

            req_kwargs = self._kwargs_check(is_get_required = True)[1]

            required = {'pot_type'  :   self.pot_type}

            required.update({key : self.pot_params[key] for key in req_kwargs})

            return required


    def _sublat_index(self, sublat):

        index = np.zeros(len(sublat))

        if np.isin(sublat, np.array([0,1])).all():

            index[sublat == 0] = -1
            index[sublat == 1] = +1

        else:

            sys.exit('Invalid sublattice input')

        return index



def __main__():

    pot_type = 'well'

    # Dictionary of paramters used to define the potential
    pot_kwargs = {
        'gap_val'           :   0.150,  # 100meV delta0
        'offset'            :   0,      # 0eV

        'well_depth'        :   -0.02,  # -20meV U0
        'gap_relax'         :   0.3,    # dimensionless beta
        'channel_width'     :   850,    # 850A L

        # Select if the well depth is modulated along the channel
        'is_const_channel'  :   True,

        'channel_depth'     :   -0.04,  # -40meV U0
        'channel_length'    :   4000,   # 1000A
        'channel_relax'     :   300     # 200A
        }

    pot = potential(pot_type, [0,0,0], [1,0,0], **pot_kwargs)

    y_list = np.linspace(-5000, 5000, 1000)
    xyz1 = np.array([[0,y,0] for y in y_list])
    xyz2 = np.array([[0,y,1] for y in y_list])
    sublat = np.array([1] * len(xyz1))
    en1 = pot.pot_func(xyz1, sublat)
    en2 = pot.pot_func(xyz2, sublat)

    plt.plot(y_list, en1)
    plt.plot(y_list, en2)
    plt.show()

    xyz0 = np.array([[x, y, 0] for x in range(-5000, 5000, 200) for y in range(-5000, 5000, 400)])
    xyz1 = np.array([[x, y, 1] for x in range(-5000, 5000, 200) for y in range(-5000, 5000, 400)])

    pots0 = pot.pot_func(xyz0, [0] * len(xyz0))
    pots1 = pot.pot_func(xyz1, [0] * len(xyz1))

    [X0,Y0,Z0] = list(zip(*xyz0))
    [X1,Y1,Z1] = list(zip(*xyz1))

    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    ax.plot_trisurf(X0,Y0,pots0)
    ax.plot_trisurf(X1,Y1,pots1)
    plt.show()

    #xyz = np.array([[1,0,0],[0,1,0]])
    #sublat = np.array([0,1])
    #en = pot.pot_func(xyz, sublat)
    #print(en)

if __name__ == '__main__':
    __main__()