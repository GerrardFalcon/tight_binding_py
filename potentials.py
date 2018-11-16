import numpy as np
from numpy.linalg import norm as nrm

import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

                def pot(xyz, sublat = None):

                    return self.equal_onsite(xyz, onsite = 0)

            elif self.pot_type == 'step':

                def pot(xyz, sublat):

                        return self.pot_func_step(xyz, sublat,**self.pot_params)

            elif self.pot_type == 'tanh':

                def pot(xyz, sublat):

                    return self.pot_func_tanh(xyz, sublat, **self.pot_params)

            elif self.pot_type == 'well':

                def pot(xyz, sublat = None):

                    return self.pot_func_BLG_well(xyz, **self.pot_params)

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
        tanhL = np.tanh(-
            np.dot(yL - (xyz - self.int_loc), int_par) / channel_relax)

        # Function for 'right' slope in the channel
        tanhR = np.tanh(
            np.dot(yR - (xyz - self.int_loc), int_par) / channel_relax)

        return .5 * (tanhL + tanhR)


    def pot_func_BLG_well(self, xyz, gap_val, offset, well_depth,
        channel_width, gap_relax, is_const_channel = True, cut_at = .0,
        gap_min = .01, lead_offset = -.2, is_shift_channel_mid = False,
        **kwargs):

        # Calculate 1 / cosh(x / L) where 'x' is in the direction perpendicular
        # to the interface
        sech_perp = np.reciprocal(np.cosh(
            np.dot(xyz - self.int_loc, self.int_norm) / channel_width))

        u_xy = well_depth * sech_perp

        half_delta = 0.5 * gap_val * (1 - gap_relax * sech_perp)

        #print(well_depth + 0.5 * gap_val * (1 - gap_relax))

        if is_const_channel:

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
            self.pot_params['cut_gap_min'] = (1 - y_func) * gap_min
            self.pot_params['cut_lead_offset'] = (1 - y_func) * lead_offset

        else:
            
            # Get the full y dependence
            y_func = self._get_y_func(xyz, **kwargs)

        # Initialise an array of zeros to be filled
        energies = np.zeros_like(xyz[:,0], np.float64)

        # Fill lower layer
        energies[xyz[:,2] == 0] = ( (u_xy - half_delta) * y_func +
            (lead_offset - gap_min) * (1 - y_func) + offset )[xyz[:,2] == 0]

        # Fill upper layer
        energies[xyz[:,2] != 0] = ( (u_xy + half_delta) * y_func +
            (lead_offset + gap_min) * (1 - y_func) + offset )[xyz[:,2] != 0]

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


    def equal_onsite(self, xyz, onsite = 0):

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

            # Return required params for a channel constant along one axis which
            # uses the bare values
            required = ['gap_val', 'offset', 'well_depth', 'gap_relax', \
                'channel_width','gap_min', 'lead_offset', \
                'channel_length', 'channel_relax']

            if self.pot_params['is_const_channel']:

                # Add cut_at the the list of required parameters
                required.append('cut_at')

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


    def print_pot_smoothing_info(self, xyz_IN, sublat = [0],
        plot_density = 100):
        """
        Outputs to file information about the distance over which smoothing
        occurs when the potential profile used is the well type with a potential
        step

        """
        x_lims = [np.min(xyz_IN[:,0]), np.max(xyz_IN[:,0])]
        y_lims = [np.min(xyz_IN[:,1]), np.max(xyz_IN[:,1])]

        z_list = list(set(xyz_IN[:,2]))
        sublat_list = list(set(sublat))

        # Checking along the centre of the channel for where the bottom of
        # the channel reaches 10% and 90% height
        y_2 = np.linspace(y_lims[0], 0, plot_density)
        xyz_mid = np.array([[0, y_val, np.max(z_list)] for y_val in y_2])
        e_at_min = self.pot_func(xyz_mid)
        e_lims = [np.min(e_at_min), np.max(e_at_min)]
        e_diff = e_lims[1] - e_lims[0]
        e_lims = [e_lims[0] + .05 * e_diff, e_lims[0] + .95 * e_diff]

        y_in_rng = xyz_mid[np.logical_and(e_at_min > e_lims[0],
            e_at_min < e_lims[1])][::,1]

        print_out('channel centre height varies over ' + \
            str(y_in_rng[-1] - y_in_rng[0]) + ' Angstroms')

        print_out('Actual channel length is ' + str(2 * abs(y_in_rng[-1]))
            + 'Angstroms')

        chnl_height = self.pot_func(np.array([[0,0,1]]),[0])

        print_out('pot_3D : Max channel height is ' + str(chnl_height))


    def plot_pot_3D(self, xyz_IN, sublat = [0], plot_density = 100):

        x_lims = [np.min(xyz_IN[:,0]), np.max(xyz_IN[:,0])]
        y_lims = [np.min(xyz_IN[:,1]), np.max(xyz_IN[:,1])]

        x = np.linspace(*x_lims, plot_density)
        y = np.linspace(*y_lims, plot_density)

        z_list = list(set(xyz_IN[:,2]))
        sublat_list = list(set(sublat))

        X, Y = np.meshgrid(x, y)

        pots = np.concatenate([
            [[[self.pot_func(np.array([[X[i,j], Y[i,j], z]]), sublat)[0]
            for j in range(plot_density)] for i in range(plot_density)]]
            for z in z_list for sublat in sublat_list], axis = 0)

        potsX = np.concatenate([
            [[self.pot_func(np.array([[x[i], 0, z]]), sublat)[0]
            for i in range(plot_density)]]
            for z in z_list for sublat in sublat_list], axis = 0)

        potsY = np.concatenate([
            [[self.pot_func(np.array([[0, y[i], z]]), sublat)[0]
            for i in range(plot_density)]]
            for z in z_list for sublat in sublat_list], axis = 0)

        # -------------------------------------------------------------------- #

        fig = plt.figure(figsize = (11, 9))
        gs = gridspec.GridSpec(6, 4, height_ratios = [1,1,.2,1,1,1], wspace = 1., hspace = 0.8)
        
        ax = fig.add_subplot(gs[0:2,0:3], projection = '3d')
        cax = fig.add_subplot(gs[2,:3])
        ax2 = fig.add_subplot(gs[3:5,3])
        ax3 = fig.add_subplot(gs[5,:3])
        ax4 = fig.add_subplot(gs[3:5,:3])

        max_xy = np.max(np.abs([*x_lims, *y_lims]))
        max_lims = np.array([-max_xy, max_xy]) * 1.1
        pot_lims = 1.1 * np.array([np.min(pots), np.max(pots)])
        pad = 0#1500

        for i, pot in enumerate(pots):

            ax.plot_surface(X, Y, pot, cmap = 'plasma')
            ax.set_alpha(1.)

            cset = ax2.plot(potsX[i], x)

            cset = ax3.plot(y, potsY[i])

        xyMap = ax4.contourf(Y, X, pots[-1], cmap='plasma')

        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent
        #ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        #ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        #ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        # Set axes labels
        ax.set_xlabel(r'x ($\AA$)')
        ax.set_ylabel(r'y ($\AA$)')
        ax.set_zlabel(r'$\varepsilon$ (eV)')

        #ax.auto_scale_xyz(X = x_lims, Y = y_lims, Z = pot_lims)
        ax.auto_scale_xyz(X = x_lims, Y = y_lims, Z = pot_lims)

        ax.view_init(azim = 45, elev = 25)

        ax2.set_xlim(1.1 * np.min(potsX), 1.1 * np.max(potsX))
        ax2.set_ylim(1.1 * np.min(x), 1.1 * np.max(x))
        ax2.set_xlabel(r'$\varepsilon$ (eV)', fontsize = 16)
        ax2.set_ylabel(r'x ($\AA$)', fontsize = 16)
        ax2.set_xticks([-.05,.05])
        ax2.set_yticks([-1000,0,1000])
        # Hide the right and top spines
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)

        ax3.set_xlim(1.1 * np.min(Y), 1.1 * np.max(Y))
        ax3.set_ylim(1.1 * np.min(pots), 1.1 * np.max(pots))
        ax3.set_xlabel(r'y ($\AA$)', fontsize = 16)
        ax3.set_ylabel(r'$\varepsilon$ (eV)', fontsize = 16)
        # Hide the right and top spines
        ax3.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)

        ax4.set_xlim(1.1 * np.min(Y), 1.1 * np.max(Y))
        ax4.set_ylim(1.1 * np.min(X), 1.1 * np.max(X))
        ax4.set_xlabel(r'y ($\AA$)', fontsize = 16)
        ax4.set_ylabel(r'x ($\AA$)', fontsize = 16)
        ax4.set_yticks([-1000,0,1000])
        # Hide the right and top spines
        ax4.spines['right'].set_visible(False)
        ax4.spines['top'].set_visible(False)

        cbar = fig.colorbar(mappable = xyMap, cax = cax,
            orientation='horizontal')
        cbar.set_label(r'$\varepsilon$ (eV)', fontsize = 16)
        cbar.ax.tick_params(labelsize = 15) 

        axList = [ax, ax2, ax3, ax4]
        for a in axList:
            plt.setp(a.get_xticklabels(), rotation='horizontal', fontsize=16)
            plt.setp(a.get_yticklabels(), rotation='horizontal', fontsize=16)

        plt.show()


    def plot_side_profile(self, xyz_IN, axis = 1, sublat = [0],
        plot_density = 100):

        dir_lims = [np.min(xyz_IN[:, axis]), np.max(xyz_IN[:, axis])]
        dir_list = np.linspace(*dir_lims, plot_density)
        z_list = list(set(xyz_IN[:,2]))
        sublat_list = list(set(sublat))

        xyz = np.zeros((plot_density, 3))
        xyz[:, axis] = dir_list

        xyz_list = []
        for z in z_list:

            xyz_tmp = np.copy(xyz)
            xyz_tmp[:, 2] = [z] * len(xyz)
            xyz_list.append(xyz_tmp)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        pots = np.concatenate([[self.pot_func(xyz, sublat)]
            for xyz in xyz_list for sublat in sublat_list])

        for pot in pots:

            plt.plot(xyz[:, axis], pot)

        plt.show()


    def plot_energy_cuts(self, xyz_IN, axis = 0, sublat = [0], cuts_at = [0],
        plot_density = 100):

        dir_lims = [np.min(xyz_IN[:, axis]), np.max(xyz_IN[:, axis])]
        dir_list = np.linspace(*dir_lims, plot_density)
        z_list = list(set(xyz_IN[:,2]))
        sublat_list = list(set(sublat))

        xyz = np.zeros((plot_density, 3))
        xyz[:, axis] = dir_list

        xyz_list = []
        for z in z_list:

            xyz_tmp = np.copy(xyz)
            xyz_tmp[:, 2] = [z] * len(xyz)
            xyz_list.append(xyz_tmp)

        import matplotlib.colors as c
        colours = list(c._colors_full_map.values())

        # Temporarily make 'is_const_channel' True and store the original value
        is_const_channel_tmp = self.pot_params['is_const_channel']
        self.pot_params['is_const_channel'] = True

        en = []

        for cut in cuts_at:

            self.pot_params['cut_at'] = cut

            en_tmp = [self.pot_func(xyz, sublat) \
                for xyz in xyz_list for sublat in sublat_list]

            en.append(en_tmp)

        # value of x_list closest to 0
        dir_mid = np.median(dir_list)

        tmp = [abs(x - dir_mid) for x in dir_list]

        mid_at = dir_list[tmp.index(min(tmp))]

        print('Middle of transverse potential located at x = ', mid_at)

        mid_vals = []

        for i in range(len(cuts_at)):

            mid_vals.append(float(en[i][1][dir_list == mid_at]))

            plt.plot(dir_list, en[i][0], color = colours[i + 20])
            plt.plot(dir_list, en[i][1], color = colours[i + 20])

        print('Conduction band energies at mid-points E = ', mid_vals )

        plt.show()

        # Return the value of 'is_const_channel' to its original value
        self.pot_params['is_const_channel'] = is_const_channel_tmp


def param_profile(pot_type, pot_kwargs):

    lim = 2000
    stepNo = 200
    y_list = np.linspace(-lim, lim, stepNo)

    pars = []
    for y in y_list:

        pot_tmp = potential(pot_type, [0,0,0], [1,0,0],
            **{**pot_kwargs, **{'cut_at':y}})

        pot_tmp.pot_func(np.array([[0, y, 1]]))

        pars.append(pot_tmp.pot_params)

    wd = [[y, pars[i]['cut_well_depth']] for i, y in enumerate(y_list)]
    gv = [[y, pars[i]['cut_gap_val']] for i, y in enumerate(y_list)]
    gm = [[y, pars[i]['cut_gap_min']] for i, y in enumerate(y_list)]
    lo = [[y, pars[i]['cut_lead_offset']] for i, y in enumerate(y_list)]

    fig = plt.figure(figsize = (12, 5))
    ax = fig.add_subplot(111)

    for param in [wd, gv, gm, lo]:

        ax.plot(*list(zip(*param)))
    
    ax.legend(('well_depth', 'gap_val', 'gap_min', 'lead_offset'))
    plt.setp(ax.get_xticklabels(), rotation='horizontal', fontsize=16)
    plt.setp(ax.get_yticklabels(), rotation='horizontal', fontsize=16)
    ax.set_xlabel(r'y ($\AA$)', fontsize = 16)
    ax.set_ylabel(r'$\varepsilon$ ($eV$)', fontsize = 16)
    plt.tight_layout()

    plt.show()


def __main__():

    create_out_file('test_out.txt')

    pot_type = 'well'

    # Dictionary of paramters used to define the potential
    
    pot_kwargs = {
        'gap_val'           :   .150,       # 100meV delta0
        'offset'            :   0,          # 0eV

        'well_depth'        :   -.02,       # -20meV U0
        'gap_relax'         :   .3,         # dimensionless beta
        'channel_width'     :   500,        # 850A / 500A

        # Select if the well depth is modulated along the channel
        'is_const_channel'  :   False,
        # If is_const_channel is True, we can also supply a y-value for which to
        # take a cut of the potential
        'cut_at'            :   0,      # -(1200, 1060, 930, 800, 0) w/ d faults

        'gap_min'           :   .01,        # -40meV U0
        'lead_offset'       :   -.2,       # -0.1
        'channel_length'    :   2400,       # 2000A
        'channel_relax'     :   200         # 100A
        }

    test1 = False
    test2 = True

    if test1:
        pot = potential(pot_type, [0,0,0], [1,0,0], **pot_kwargs)

        lim_y = 1000
        
        xyz = np.array([[x, y, z] for x in range(-3000, 3000 + 1, 100) 
                for y in range(-lim_y, lim_y + 1, 200) for z in [0,1]])

        pot.plot_pot_3D(xyz)

        pot.plot_side_profile(xyz)

        cuts = np.linspace(-lim_y + 100, -lim_y + 500, 4)
        cuts = np.append(cuts, 0)
        print('cuts at : ', cuts)

        pot.plot_energy_cuts(xyz, cuts_at = cuts)

    if test2:
        pot_kwargs.update({'is_const_channel':True})

        param_profile(pot_type, pot_kwargs)



if __name__ == '__main__':
    __main__()