import numpy as np
from numpy.linalg import norm as nrm
import matplotlib.pyplot as plt

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

                        return self.pot_func_step(xyz, sublat)

            elif self.pot_type == 'tanh':

                def pot(xyz, sublat):

                    return self.pot_func_tanh(xyz, sublat)

            elif self.pot_type == 'well':

                def pot(xyz, sublat):

                    return self.pot_func_BLG_well(xyz, sublat)

            else:

                sys.exit('Invalid potential choice in potentials.py')

            self.pot_func = pot


    def pot_func_BLG_well(self, xyz, sublat):

        # Calculate 1 / cosh(y / L) where y is in the direction perpendicular to
        # the interface

        inv_coshx = np.reciprocal( np.cosh(
            np.dot( xyz - self.int_loc, self.int_norm) \
            / self.pot_params['channel_width']) )

        # Calculate both U(x) and Delta(x) as per Angelika's paper
        u_x = self.pot_params['well_depth'] * inv_coshx

        half_delta_x = 0.5 * self.pot_params['gap_val'] * (
            1 - self.pot_params['gap_relax'] * inv_coshx)

        # Initialise an array of zeros to be filled
        energies = np.zeros_like(sublat, np.float64)

        # Fill lower layer
        energies[xyz[:,2] == 0] = (u_x - half_delta_x)[xyz[:,2] == 0]

        # Fill upper layer
        energies[xyz[:,2] != 0] = (u_x + half_delta_x)[xyz[:,2] != 0]

        return energies


    def pot_func_step(self, xyz, sublat):

        index = self._sublat_index(sublat)

        return self.pot_params['offset'] + self.pot_params['gap_val'] * \
            index * (2 * np.heaviside(np.dot(
                xyz - self.int_loc, self.int_norm), 0.5) - 1)


    def pot_func_tanh(self, xyz, sublat):
        # Scaling of the range over which tanh varies greatly from its limit
        scaling = 2

        index = self._sublat_index(sublat)

        return self.pot_params['offset'] + self.pot_params['gap_val'] * \
            index * np.tanh((np.dot(
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
                'channel_width']

        elif self.pot_type == 'well_3D':

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
    gap_val = 0.1 # 100meV delta0
    offset = 0 # 0eV 
    well_depth = -0.02 # -20meV U0
    gap_relax = 0.3 # dimensionless beta
    channel_width = 50 # 20A L

    kwarg_list = {
        'gap_val':gap_val,
        'offset':offset,
        'well_depth':well_depth,
        'gap_relax':gap_relax,
        'channel_width':channel_width}

    pot = potential(pot_type, [0,0,0], [0,1,0], **kwarg_list)

    y_list = np.linspace(-500, 500, 1000)
    xyz1 = np.array([[0,y,0] for y in y_list])
    xyz2 = np.array([[0,y,1] for y in y_list])
    sublat = np.array([1] * len(xyz1))
    en1 = pot.pot_func(xyz1, sublat)
    en2 = pot.pot_func(xyz2, sublat)

    plt.plot(y_list, en1)
    plt.plot(y_list, en2)
    plt.show()

    #xyz = np.array([[1,0,0],[0,1,0]])
    #sublat = np.array([0,1])
    #en = pot.pot_func(xyz, sublat)
    #print(en)

if __name__ == '__main__':
    __main__()