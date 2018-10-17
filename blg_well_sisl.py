import numpy as np
import matplotlib.pyplot as plt

import copy

from sisl import Atom, Geometry, SuperCell, Hamiltonian, BandStructure, plot


# -------------------------------- POTENTIAL --------------------------------- #


class pot_func_well:
    """ Class which sets up the potential function for the system """

    def __init__(self, int_loc_IN, int_norm_IN, **kwargs):

        self.int_loc = np.array(int_loc_IN)

        # The norm to the plane that defines the interface
        self.int_norm = np.array(int_norm_IN) / np.linalg.norm(
            np.array(int_norm_IN))

        self.pot_params = kwargs

        def pot(xyz):

            return self.pot_func_BLG_well(xyz, **self.pot_params)

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
        channel_width, gap_relax, is_const_channel = True, cut_at = 0,
        gap_min = 0.01, lead_offset = -0.05, **kwargs):

        # Calculate 1 / cosh(x / L) where 'x' is in the direction perpendicular
        # to the interface
        sech_perp = np.reciprocal(np.cosh(
            np.dot(xyz - self.int_loc, self.int_norm) / channel_width))

        u_xy = well_depth * sech_perp

        half_delta = 0.5 * gap_val * (1 - gap_relax * sech_perp)

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


# ---------------------------------- DEVICE ---------------------------------- #

def make_dev(ori = 'zz', cell_num = (1, 1), stripe_len = 1, scaling = 1,
    nsc = [1,1,1], is_finite = False):
    """
    Make the device by giving sisl.Geometry an initial orthogonal supercell and
    then tiling it

    """
    a = 2.46 * scaling ; a_z = 3.35 ; d = a / (2 * np.sqrt(3))

    #shift_to_cell = np.array([.5 * d, .75 * a, 0])

    carbon_a = Atom(6, R = a / np.sqrt(3) + 0.01, tag = 'A')
    carbon_b = Atom(6, R = a / np.sqrt(3) + 0.01, tag = 'B')
    atom_list = [carbon_a, carbon_b] * 4

    xyz = np.array([
        [0,         0,          0   ],
        [d,       - a / 2,      0   ],
        [3 * d,   - a / 2,      0   ],
        [4 * d,     0,          0   ],
        [d,       - a / 2,      a_z   ],
        [2 * d,     0,          a_z   ],
        [4 * d,     0,          a_z   ],
        [5 * d,   - a / 2,      a_z   ]]) + np.array([.5 * d, .75 * a, 0])

    lat_vecs = np.array([
        [6 * d,     0,      0       ],
        [0,         a,      0       ],
        [0,         0,      2 * a_z ]])

    if ori == 'ac':

        xyz = np.array([xyz[:,1], xyz[:,0], xyz[:,2]]).T

        lat_vecs = np.array(
            [lat_vecs[:,1], lat_vecs[:,0], lat_vecs[:,2]]).T

    print('a', a)

    # Repeat in the direction of the stripe
    xyz = np.concatenate([xyz + i * lat_vecs[0] for i in range(stripe_len)]
        ) - (stripe_len / 2) * lat_vecs[0]
    atom_list = np.concatenate([atom_list for i in range(stripe_len)])
    
    # Repeat in the direction of transport to the left and right of the origin
    xyz = np.concatenate([xyz + i * lat_vecs[1] for i in range(sum(cell_num))]
        ) - cell_num[0] * lat_vecs[1]
    atom_list = np.concatenate([atom_list for i in range(sum(cell_num))])

    # Generate the supercell lattice vectors
    lat_vecs_sc = np.copy(lat_vecs)
    lat_vecs_sc[0] *= stripe_len
    lat_vecs_sc[1] *= np.sum(cell_num)
    print(np.sum(lat_vecs_sc, axis = 1))

    # Create the geometry and tile it in the non-transport direction
    blg = Geometry(xyz, atom_list, sc = SuperCell(list(lat_vecs_sc), nsc = nsc))

    return blg


def make_cell_num(cell_num_L, cell_num_R, stripe_len, SF, is_scale_CN = True):
    
    if cell_num_R is None: cell_num_R = cell_num_L

    if is_scale_CN:

        cell_num = (
            np.sum(np.divmod(cell_num_L, SF)),
            np.sum(np.divmod(cell_num_R, SF)))

        stripe_len = np.sum(np.divmod(stripe_len, SF))

    else: cell_num = (cell_num_L, cell_num_R)

    return cell_num, stripe_len


# ----------------------------------- MAIN ----------------------------------- #

def potential_testing(lat, pot):
    """
    Contains plotting functions used to test that the potential is correctly
    implemented

    """
    lim_x = 2000
    lim_y = 1000
    x_list = np.linspace(-lim_x, lim_x, 1000)
    y_list = np.linspace(-lim_y, lim_y, 1000)

    x_cuts = np.append(np.linspace(-lim_y + 200, -lim_y + 400, 4), 0)
    print('x cuts @ y = ', np.around(x_cuts,0))

    y_cuts = [0]
    print('y cuts @ x = ', np.around(y_cuts,0))

    import matplotlib.colors as c
    colours = list(c._colors_full_map.values())

    energy_cut_x = []
    energy_cut_y = []

    # Make multiple cuts along the x-axis
    for cut in x_cuts:

        en_tmp = []

        for z in [0,1]:

            xyz = np.array([[x, cut, z] for x in x_list])

            en_tmp.append(pot.pot_func(xyz))

        energy_cut_x.append(en_tmp)

    # Make multiple cuts along the y-axis
    for cut in y_cuts:

        en_tmp = []

        for z in [0,1]:

            xyz = np.array([[cut, y, z] for y in y_list])

            en_tmp.append(pot.pot_func(xyz))

        energy_cut_y.append(en_tmp)

    fig = plt.figure()
    ax_x = fig.add_subplot(211)
    ax_y = fig.add_subplot(212)

    colour_base = 40

    for i, energies in enumerate(energy_cut_x):

        ax_x.plot(x_list, energies[0], color = colours[i + colour_base])
        ax_x.plot(x_list, energies[1], color = colours[i + colour_base])

    for i, energies in enumerate(energy_cut_y):

        ax_y.plot(y_list, energies[0], color = colours[i + colour_base])
        ax_y.plot(y_list, energies[1], color = colours[i + colour_base])

    ax_x.set_xlabel("x (Ang)") ; ax_x.set_ylabel("E (eV)")
    ax_y.set_xlabel("y (Ang)") ; ax_y.set_ylabel("E (eV)")

    plt.show()


def lat_plot(dev):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dev.xyz[:,0], dev.xyz[:,1])
    pad = 0.1
    z_lower = min(dev.xyz[:, 2])
    for ia in dev:
        if dev.xyz[ia, 2] == z_lower:
            ax.annotate(str(ia), (dev.xyz[ia, 0] + pad, dev.xyz[ia, 1] - pad))
        if dev.xyz[ia, 2] != z_lower:
            ax.annotate(str(ia), (dev.xyz[ia, 0] + pad, dev.xyz[ia, 1] + pad))
    ax.set_xlabel('$x$') ; ax.set_ylabel('$y$')
    pad = 0.4
    ax.set_xlim(min(dev.xyz[:,0]) - pad, max(dev.xyz[:,0]) + pad)
    ax.set_ylim(min(dev.xyz[:,1]) - pad, max(dev.xyz[:,1]) + pad)
    plt.show()


def finite_bands(dev, pot, SF):
    H = Hamiltonian(dev)

    R = np.array([1.4,   1.44,   3.33,   3.37]) * SF
    t = (0,     3.16,   0,      0.39)

    H.construct([R, t])

    energies = pot.pot_func(dev.xyz)

    for i in dev.iter():

        H[i,i] = energies[i]
    
    #band = BandStructure(H, [[-np.pi / 2.46, 0, 0], [np.pi / 2.46, 0, 0]],
    #    400, [r'$-\pi$',r'$\pi$'])

    band = BandStructure(H, [[0, 0, 0], [0, 0.5, 0],
                  [1/3, 2/3, 0], [0, 0, 0]],
              400, [r'$\Gamma$', r'$M$', r'$K$', r'$\Gamma$'])

    bnds = band.asarray().eigh()

    lk, kt, kl = band.lineark(True)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for bnd in bnds.T:

        ax.plot(lk, bnd)

    plt.xticks(kt, kl)
    plt.xlim(0, lk[-1])
    #plt.ylim([0, 0.055])
    plt.ylim([-3, 3])
    #ax.set_ylim(-.1,.1)

    plt.show()


def __main__():

    SF = 10 # Scale Factor by which to scale the system

    # Define the number of cells either side of whatever interface we are using
    cell_num_L = 1          # 300
    cell_num_R = None       # If None this is set to equal cell_num_L

    stripe_len = 800        # 1000 / 1500 (sum of cell_num usually)

    # Form the cell_num and stripe_len w/ scaling if requested (True by default)
    cell_num, stripe_len = make_cell_num(cell_num_L, cell_num_R, stripe_len, SF)

    # Number of supercells to include in the x and y directions respectively
    nsc_x = 3
    nsc_y = 3

    dev = make_dev(ori = 'zz', cell_num = cell_num, stripe_len = stripe_len,
        scaling = SF, nsc = [nsc_x, nsc_y, 1])

    print('\n\tDevice is:\n\n', dev)

    print(dev.cell)

    pot_kwargs = {
        'gap_val'           :   0.150,  # 100meV delta0
        'offset'            :   0,      # 0eV

        'well_depth'        :   -0.02,  # -20meV U0
        'gap_relax'         :   0.3,    # dimensionless beta
        'channel_width'     :   500,    # 850A / 500A

        # Select if the well depth is modulated along the channel
        'is_const_channel'  :   False,
        # If is_const_channel is True, we can also supply a y-value for which to
        # take a cut of the potential
        'cut_at'            :   0,  # -(1200, 1060, 930, 800, 0) w/ d faults

        'gap_min'           :   0.01,   # -40meV U0
        'lead_offset'       :   0.0,   # -0.1

        'channel_length'    :   1000,   # 1000A
        'channel_relax'     :   100     # 100A
        }

    int_norm = [1, 0, 0] ; int_loc = [0, 0, 0]

    # Create the potential
    pot = pot_func_well(int_loc, int_norm, **pot_kwargs)

    potential_testing(dev, pot)

    energies = pot.pot_func(dev.xyz)
    
    plt.plot(dev.xyz[:,0], energies, 'bo')
    plt.show()

    finite_bands(dev, pot, SF)


if __name__ == '__main__':

    __main__()