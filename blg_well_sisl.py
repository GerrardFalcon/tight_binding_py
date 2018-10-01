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

            return self.potential_func(xyz, **self.pot_params)

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
            (np.dot((xyz - self.int_loc) - yL, int_par)) / channel_relax)

        # Function for 'right' slope in the channel
        tanhR = np.tanh(-
            (np.dot((xyz - self.int_loc) - yR, int_par)) / channel_relax)

        return 0.5 * (tanhL + tanhR)


    def _BLG_well_xy(self, y_func, u_xy, half_delta, well_depth, gap_min,
        **kwargs):
        """ Modifies the potential to vary in the y-direction """

        # U(x,y) with varying channel depth. 'y' denotes direction along the
        # channel

        # U(x,y) in full with the y dependence included
        u_xy *= y_func

        # delta(x,y) in full - scale to allow for a constant gap, vary in y
        # and apply minimimum gap
        half_delta *= y_func / (1 + gap_min)
        half_delta += gap_min

        return u_xy, half_delta


    def potential_func(self, xyz, gap_val, offset, well_depth,
        channel_width, gap_relax, is_const_channel = True, cut_at = None,
        **kwargs):

        # Calculate 1 / cosh(x / L) where 'x' is in the direction perpendicular
        # to the interface

        sech_perp = np.reciprocal(np.cosh(np.dot(
            xyz - self.int_loc, self.int_norm) / channel_width))

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

            # Get the values of u_xy and half_delta after being modified
            # for the specific position along the channel defined above
            u_xy, half_delta = self._BLG_well_xy(y_func, u_xy, half_delta,
                well_depth, **kwargs)

        else:
            
            # Get the values of u_xy and half_delta after being modified with
            # the y-dependence
            u_xy, half_delta = self._BLG_well_xy(
                self._get_y_func(xyz, **kwargs),
                u_xy, half_delta, well_depth, **kwargs)

        # Initialise an array of zeros to be filled
        energies = np.zeros(len(xyz), np.float64)

        # Fill lower layer
        energies[xyz[:,2] == 0] = (u_xy - half_delta + offset)[xyz[:,2] == 0]

        # Fill upper layer
        energies[xyz[:,2] != 0] = (u_xy + half_delta + offset)[xyz[:,2] != 0]

        return energies


# ---------------------------------- DEVICE ---------------------------------- #

def rotate_ac(lat):
    """
    Rotate the system by 90 degrees counterclockwise and redefine the
    corresponding supercell lattice vectors

    """

    lat = lat.rotatec(90)

    unit_vecs = np.identity(3)

    lat_cell_tmp = lat.cell

    sc_tmp = []

    for u_vec in unit_vecs:

        for vec in lat_cell_tmp:

            dot_prod = np.dot(u_vec, vec)

            if np.abs(dot_prod) > 1E-3:

                sc_tmp.append(np.sign(dot_prod) * vec)

                if np.sign(dot_prod) < 0:

                    lat = lat.translate(- vec)

    # Fix the supercell lattice vectors again
    lat.set_supercell(sc_tmp)

    return lat


def do_tiling(lat, tiling):
    """
    Tiles the lattice given our 'tiling' input and returns the lattice object

    """
    for i_axis in range(len(tiling)):

        if sum(np.abs(tiling[i_axis])) > 0:

            # Absolute value of tiling since tile only takes positive
            # values
            lat = lat.tile(sum(np.abs(tiling[i_axis])), axis = i_axis)

        for tile_val in tiling[i_axis]:

            # If passed a negative value along one axis, translate the
            # system
            if tile_val < 0:

                lat = lat.move( tile_val * lat.cell[i_axis] )

    return lat


def make_dev(a, a_z, ori = 'zz', tiling = None):
    """
    Make the device by giving sisl.Geometry an initial orthogonal supercell and
    then tiling it

    """

    d = a / (2 * np.sqrt(3))

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
        [5 * d,   - a / 2,      a_z   ]])# + shift_to_cell

    lat_vecs_sc = np.array([
        [6 * d,     0,      0       ],
        [0,         a,      0       ],
        [0,         0,      2 * a_z ]])

    blg = Geometry(xyz, atom_list)#, sc = SuperCell())

    # Centre the atoms in the supercell lattice vectors
    blg = blg.move(blg.center(what = 'cell') - blg.center(what = 'xyz'))

    if ori == 'ac':

        blg = rotate_ac(blg)

    if tiling is not None:

        blg = do_tiling(blg, tiling)

    return blg


# ----------------------------------- MAIN ----------------------------------- #

def potential_testing(lat, pot):
    """
    Contains plotting functions used to test that the potential is correctly
    implemented

    """
    lim_x = 3000
    lim_y = 1300
    x_list = np.linspace(-lim_x, lim_x, 1000)
    y_list = np.linspace(-lim_y, lim_y, 1000)

    x_cuts = np.append(np.linspace(-lim_y + 200, -lim_y + 400, 4), 0)
    print('Cuts in x-direction at y = ', x_cuts)

    y_cuts = [0]
    print('Cuts in y-direction at y = ', y_cuts)

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


def __main__():

    a = 2.46 ; a_z = 3.35

    # Make the device
    # Tiling is an array which determines how many unit cells are on each side
    # of the interface. i.e [[-2, 2], [0, 1]] makes two cells either side of the
    # origin in the x direction and one on the right in the y direction

    dev = make_dev(a, a_z, ori = 'zz', tiling = [[-2, 2], [0, 1]])

    print(dev.sc)

    print(dev)

    print('Number of atoms : ', len(dev.xyz))

    pot_kwargs = {
        'gap_val'           :   0.150,  # 100meV delta0
        'offset'            :   0,      # 0eV

        'well_depth'        :   -0.02,  # -20meV U0
        'gap_relax'         :   0.3,    # dimensionless beta
        'channel_width'     :   500,    # 850A / 500A

        # Select if the well depth is modulated along the channel
        'is_const_channel'  :   True,
        # If is_const_channel is True, we can also supply a y-value for which to
        # take a cut of the potential
        'cut_at'            :   0,  # -(1200, 1060, 930, 800, 0) w/ defaults

        'gap_min'           :   0.01,   # -40meV U0
        'channel_length'    :   2000,   # 2000A
        'channel_relax'     :   100     # 100A
        }

    int_norm = [1, 0, 0] ; int_loc = [0, 0, 0]

    # Create the potential
    pot = pot_func_well(int_loc, int_norm, **pot_kwargs)

    potential_testing(dev, pot)

    lat_plot(dev)

    H = Hamiltonian(dev)

    

    R = (1.4,   1.44,   3.33,   3.37)
    t = (0,     3.16,   0,      0.39)

    H.construct([R, t])

    energies = pot.pot_func(dev.xyz)

    for i in dev.iter():

        H[i,i] = energies[i]
    
    band = BandStructure(H, [[0, -np.pi, 0], [0, np.pi, 0]],
        40, [r'$-\pi$',r'$\pi$'])

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
    plt.ylim([-3, 3])

    #ax.set_ylim(-.1,.1)

    plt.show()


if __name__ == '__main__':

    __main__()