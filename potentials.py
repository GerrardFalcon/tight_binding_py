import numpy as np
from numpy.linalg import norm as nrm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utility import create_out_file, print_out


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


    def _get_y_func(self, xyz, channel_length, channel_relax, **kwargs):
        """ Supplies the y-dependence to the 'well' potential """

        # unit vector parallel to the channel along positive axis
        int_par = np.cross([0,0,1], self.int_norm)

        # Calculate the position of the left/right slopes
        yL = self.int_loc - 0.5 * channel_length * int_par
        yR = self.int_loc + 0.5 * channel_length * int_par            

        # Function for 'left' slope in the channel
        tanhL = np.tanh(
            np.dot(yL + (xyz - self.int_loc), int_par) / channel_relax)

        # Function for 'right' slope in the channel
        tanhR = np.tanh(
            np.dot(yR - (xyz - self.int_loc), int_par) / channel_relax)

        return .5 * (tanhL + tanhR)


    def pot_func_BLG_well(self, xyz, sublat, gap_val, offset, well_depth,
        channel_width, gap_relax, is_const_channel = True, cut_at = None,
        gap_min = 0.01, lead_offset = -0.05, **kwargs):

        # Calculate 1 / cosh(x / L) where 'x' is in the direction perpendicular
        # to the interface

        sech_perp = np.reciprocal(np.cosh(
            np.dot(xyz - self.int_loc, self.int_norm) / channel_width))

        u_xy = well_depth * sech_perp

        half_delta = 0.5 * gap_val * (1 - gap_relax * sech_perp)

        if is_const_channel and cut_at is not None:

            # unit vector parallel to the channel along positive axis
            int_par = np.cross([0,0,1], self.int_norm)

            # Find the scaling due to the y-dependence for a sepecific
            # distance along the channel
            y_func = self._get_y_func(int_par * cut_at, **kwargs)

            # Update the values of 'pot_param' so that when we return the
            # values used to create the cut we also have the values used at the
            # cut, rather than just the input values
            self.pot_params['cut_well_depth'] = y_func * well_depth
            self.pot_params['cut_gap_val'] = y_func * gap_val

        else:
            
            # Get the full y dependence
            y_func = self._get_y_func(xyz, **kwargs)

        # Initialise an array of zeros to be filled
        energies = np.zeros_like(sublat, np.float64)

        # Fill lower layer
        energies[xyz[:,2] == 0] = ( (u_xy - half_delta) * y_func +
            (lead_offset - gam_min) * (1 - y_func) + offset )[xyz[:,2] == 0]

        # Fill upper layer
        energies[xyz[:,2] != 0] = ( (u_xy + half_delta) * y_func +
            (lead_offset + gam_min) * (1 - y_func) + offset )[xyz[:,2] != 0]

        return energies


    def pot_func_step(self, xyz, sublat, gap_val, offset):

        index = self._sublat_index(sublat)

        return offset + gap_val * index * (2 * np.heaviside(
            np.dot( xyz - self.int_loc, self.int_norm ), 0.5) - 1)


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

            if self.pot_params['is_const_channel'] and \
                self.pot_params['cut_at'] is None:

                required = ['gap_val', 'offset', 'well_depth', 'gap_relax', \
                    'channel_width']

            else:

                required = ['gap_val', 'offset', 'well_depth', 'gap_relax', \
                    'channel_width', 'cut_at', 'gap_min', 'channel_length', \
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

            for key, val in self.pot_params.items():

                # Also add values that are saved from taking a cut of the pot.
                if 'cut_' in key:

                    required.update({key : val})

            return required


    def _sublat_index(self, sublat):

        index = np.zeros(len(sublat))

        if np.isin(sublat, np.array([0,1])).all():

            index[sublat == 0] = -1
            index[sublat == 1] = +1

        else:

            sys.exit('Invalid sublattice input')

        return index


    def plot_pot_3D(self, xyz_IN, sublat):

        z_list = list(set(xyz_IN[:,2]))

        xyz = np.array([xyz_IN[xyz_IN[:,2] == z] for z in z_list])
        sub = np.array([sublat[xyz_IN[:,2] == z] for z in z_list])

        pots = [self.pot_func(xyz[i], sub[i]) for i in range(len(z_list))]

        fig = plt.figure()
        ax = fig.gca(projection = '3d')

        for i in range(len(z_list)):

            [X, Y, Z] = list(zip(*xyz[i]))

            ax.plot_trisurf(X, Y, pots[i])

        plt.show()


def __main__():

    create_out_file('test_out.txt')

    pot_type = 'well'

    # Dictionary of paramters used to define the potential
    pot_kwargs = {
        'gap_val'           :   0.150,  # 100meV delta0
        'offset'            :   0,      # 0eV

        'well_depth'        :   -0.02,  # -20meV U0
        'gap_relax'         :   0.3,    # dimensionless beta
        'channel_width'     :   500,    # 850A / 500A

        # Select if the well depth is modulated along the channel
        'is_const_channel'  :   False,
        # If is_const_channel is False but we are working with a finite system,
        # we can supply a y value for which to take a cut of the potential
        'cut_at'          :   0,

        'gap_min'           :   0.01,   # -40meV U0
        'channel_length'    :   2000,   # 1000A
        'channel_relax'     :   100     # 300A
        }

    pot = potential(pot_type, [0,0,0], [1,0,0], **pot_kwargs)

    lim_y = 1300

    y_list = np.linspace(-lim_y, lim_y, 1000)
    xyz1 = np.array([[0,y,0] for y in y_list])
    xyz2 = np.array([[0,y,1] for y in y_list])
    sublat = np.array([1] * len(xyz1))
    en1 = pot.pot_func(xyz1, sublat)
    en2 = pot.pot_func(xyz2, sublat)

    plt.plot(y_list, en1)
    plt.plot(y_list, en2)
    plt.show()

    
    xyz = np.array([[x, y, z] for x in range(-3000, 3000, 100) 
            for y in range(-lim_y, lim_y, 200) for z in [0,1]])

    sublat = np.array([0] * len(xyz))

    pot.plot_pot_3D(xyz, sublat)


    cuts = np.linspace(-lim_y + 100, -lim_y + 500, 4)
    cuts = np.append(cuts, 0)
    print('cuts at : ', cuts)
    x_list = np.linspace(-5000, 5000, 1000)

    en = []

    import matplotlib.colors as c
    colours = list(c._colors_full_map.values())

    pot_kwargs['is_const_channel'] = True

    for cut in cuts:

        pot_kwargs['cut_at'] = cut

        pot = potential(pot_type, [0,0,0], [1,0,0], **pot_kwargs)

        xyz1 = np.array([[x,0,0] for x in x_list])
        xyz2 = np.array([[x,0,1] for x in x_list])
        sublat = np.array([1] * len(xyz1))
        en1 = pot.pot_func(xyz1, sublat)
        en2 = pot.pot_func(xyz2, sublat)

        en.append([en1,en2])

    # value of x_list closest to 0
    x_mid = np.median(x_list)

    tmp = [abs(x - x_mid) for x in x_list]

    mid_at = x_list[tmp.index(min(tmp))]

    print('Middle of transverse potential located at x = ', mid_at)

    mid_vals = []

    for i in range(len(cuts)):

        mid_vals.append(float(en[i][1][x_list == mid_at]))

        plt.plot(x_list, en[i][0], color = colours[i + 20])
        plt.plot(x_list, en[i][1], color = colours[i + 20])

    print('Conduction band energies at mid-points E = ', mid_vals )

    plt.show()


if __name__ == '__main__':
    __main__()