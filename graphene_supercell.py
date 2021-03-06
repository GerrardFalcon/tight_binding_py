import numpy as np
import matplotlib.pyplot as plt

from matplotlib import path
from scipy.io import savemat

from scipy import linalg

from potentials import potential

import sys



class graphene_cell_min:
    #
    #  \ /  ^ y           a1
    #   b   |__ x        /
    #   :               /
    #   a   <-- origin  \
    #  / \               \
    #                     a2

    def __init__(self, scaling = 1, energies = [0,0]):
        half_lat = 0.5 * 2.46 * scaling

        self.scaling = scaling

        self.xyz = np.array([
            [0,     0,                  0],
            [0,     2 / np.sqrt(3),     0]], np.float64) * half_lat

        self.lat_vecs = np.array([
            [1,     + np.sqrt(3),   0               ],
            [1,     - np.sqrt(3),   0               ],
            [0,     0,          20 / half_lat   ]], np.float64) * half_lat

        self.energy = np.array(energies)

        self.sublat = np.array([0, 1])


class MLG_cell(graphene_cell_min):
    """
    Creates and manipulates a graphene supercell.

    Cell is defined by either a choice of zig-zag or armchair orientation, or
    via the input of a single vector (in terms of multiples of the principal
    lattice vectors) which defines a periodic structure

    orientation :   string input :  ac or zz for armchair or zig-zag edge along
                                    the x-axis (4-atom unit cell with orthogonal
                                    lattice vectors)

                    tuple input  :  (n, m) where the vector R = n a1 + b a2
                                    defines a periodic edge

                    None         :  Returns the minimum unit cell

    """

    def __init__(self, index = 0, orientation = None, scaling = 1, **kwargs):
        # Create the initial unit cell from the minimal cell and define the
        # supercell lattice vectors to be the same initially
        super().__init__(scaling = scaling)

        self.lat_vecs_sc = self.lat_vecs.copy()

        self.cell_idx = index

        self.orientation = orientation

        self.rot_angle = 0

        self.keywords = kwargs

        # If not defining a different supercell, return the minimum cell
        if self.orientation is not None:

            # Get an int index based on the desired orientation. (Neatens code)
            ori_index = self._the_orientation()

            # If either armchair or zig zag, create the 4-atom supercell
            if ori_index in [0, 1]:

                # Create the default zz unit cell
                self.xyz, self.lat_vecs_sc, self.energy, self.sublat = \
                    self._get_min_orthogonal_cell()

                self._align_to_axes()

                # If ac is desired, rotate the system and realign axes
                if ori_index == 1:

                    self.rotate_xy(-90, is_radians = False)

                self._align_ortho_lat_vecs()

            else:

                self.orientation = np.array(self.orientation)

                self.orientation, self.lat_vecs_sc[0:2], self.xyz, self.energy,\
                    self.sublat = self._get_periodic_cell()

                self.rot_angle = self._align_to_axes(return_angle = True)

            # Centre atoms in the cell and fix cell with its bottom-left corner
            # in the x-y plane centred at the position `blc'
            if np.abs(self.lat_vecs_sc[1,1]) > 1E-5:

                # Check that this is the one in the y-direction
                blc = self.cell_idx * self.lat_vecs_sc[1]

            self._set_cell_corner(blc)


    def _the_orientation(self):
        """
        Put here to clean up the __init__ function

        Returns an integer defining which type of structure we are wanting, or
        provides an error statement if the input is invalid

        0 - zig-zag
        1 - armcahir
        2 - periodic
        """
        if type(self.orientation) == str:

            if self.orientation == 'zz':

                return 0

            elif self.orientation == 'ac':

                return 1

            else:

                err_str = self.__class__.__name__ +'(): Orientation string may \
                    be either \'zz\' or \'ac\', not ' + str(self.orientation)

                raise ValueError(err_str)

        elif type(self.orientation) == list:

            if len(self.orientation) == 3 and all(
                isinstance(i, int) for i in self.orientation):

                return 2

            else:

                err_str =  str(self.__class__.__name__) + '(): List input must \
                    contain three integer elements'

                raise ValueError(err_str)

        else:

            err_str = str(self.__class__.__name__) + '(): Orientation must be \
            either a string or numpy array, not ' + str(type(self.orientation))

            raise ValueError(err_str)


    def _get_min_orthogonal_cell(self):
        """
        Generate and return the minimum orthogonal cell which has the default
        orientation:
         ___________________________
        |                           |           # # # In cell coupling
        |     \           /         |           
        |       A1 # # # B1         |  --> y    ----- Periodic coupling
        |     #           #         | |
        |    #             #        | x         : : : Out of cell coupling
        |  B2               A2 : : :|
        |    \             /        |
        |___________________________|

        But can be rotated using the other in-built functions

        """
        new_A_site = [self.xyz[0] + self.lat_vecs[0]]

        new_B_site = [self.xyz[1] + self.lat_vecs[1]]

        new_xyz = np.concatenate((new_B_site, self.xyz, new_A_site))

        # Take absolute value of orthogonal lattice vectors so that they point
        # correctly along the positive x and y axes
        new_lat_vecs_sc = np.array([self.lat_vecs[0] + self.lat_vecs[1],
            self.lat_vecs[0] - self.lat_vecs[1],self.lat_vecs[2]])

        new_energies = np.append(np.insert(self.energy, 0, self.energy[1]),
            self.energy[0])

        new_sublat = np.append(np.insert(self.sublat, 0, self.sublat[1]),
            self.sublat[0])

        return new_xyz, new_lat_vecs_sc, new_energies, new_sublat


    def _check_rectangular(self):
        """ Raise error if not rectangular, else pass. Neatens up code """
        # Check unit cell is rectangular. Assumes it is if we have defined our
        # cell to have a specific shape. ! Could extend to include output from
        # _check_lat_vecs_aligned()

        if self.orientation is None:

            err_str = str(self.__class__.__name__) + '(): Supercell is not \
                rectangular, cannot fit to axes'

            raise TypeError(err_str)

        else:

            pass


    def _check_lat_vecs_aligned(self):
        """
        Returns True if the lattice vectors are aligned to the coordinate
        axes and False otherwise

        """
        unit_vecs = np.identity(3)

        # Make array containing whether the combination of each superlattice
        # vector with each unit vector is orthogonal 
        is_ortho = np.array([[
            True if np.abs(np.dot(self.lat_vecs_sc[i], unit_vecs[j])) < 1E-8 \
            else False for i in range(3)] for j in range(3)])

        # Expect 2 True values along each axis, meaning
        # np.sum(is_ortho, axis = 1) = [2,2,2]. Check this is the case and
        # return True or False
        return np.all(np.sum(is_ortho, axis = 1) == 2 * np.ones(3))


    def _align_ortho_lat_vecs(self, is_check_axis_align = True):
        """
        Check that the supercell lattice vectors are pointing along the positive
        x and y directions
        """
        # if unsure, check that the supercell vectors are aligned to the axes
        if is_check_axis_align:

            if not self._check_lat_vecs_aligned():

                err_str = str(self.__class__.__name__) +'(): Supercell vecs \
                    not parallel to axes in \'_align_ortho_lat_vecs\''

                raise TypeError(err_str)

        # Generate the three unit vectors to compare to
        unit_vecs = np.identity(3)

        # Organise them into the order x, y z and multiply by the sign of the
        # dot product with the unit vector to make sure they are positive.
        # Generates an extra layer of brackets in the list comprehension
        # which we flatten out
        vec_list_tmp = np.array([[
            np.sign(np.dot(self.lat_vecs_sc[i], unit_vecs[j])) * \
            self.lat_vecs_sc[i] for i in range(3)
            if np.abs(np.dot(self.lat_vecs_sc[i], unit_vecs[j])) > 1E-3]
            for j in range(3)]).flatten()

        # Double check that there are 9 elements in the list in case of errors
        if vec_list_tmp.shape != (9,):

            err_str = str(self.__class__.__name__) +'(): Something strange \
                happening in \'_align_ortho_lat_vecs\''

            raise TypeError(err_str)

        # If all is fine, reshape and redefine the supercell lattice vectors
        self.lat_vecs_sc = vec_list_tmp.reshape(3,3)


    def _align_to_axes(self, return_angle = False):
        """
        Find the angle between the nominal first supercell lattice vector and
        rotate the system so that it lies along the x-axis.

        ###     Assumes z - axis is already aligned since none of the rest of 
        ###     the code rotates that axis

        """
        self._check_rectangular()

        # Find angle between the first vector (assumed in the xy plane) and the
        # x - axis
        off_angle = np.arccos(self.lat_vecs_sc[0,0] / np.linalg.norm(
            self.lat_vecs_sc[0,0:2]))

        self.rotate_xy(-off_angle)

        if np.abs(np.dot(np.array([1,0,0]), self.lat_vecs_sc[0])) < 1E-8:

            err_str = str(self.__class__.__name__) + '(): Something went wrong,\
             first supercell lattice vector is not aligned to the x-axis'

            raise ValueError(err_str)

        else:

            self._align_ortho_lat_vecs(is_check_axis_align = False)

            return off_angle


    def _set_cell_corner(self, blc = np.array([0,0,0]), is_set_z = False):
        """
        Performs two operations:

        1.  Aligns the bottom left corver of the rectangular unit (2D) cell to a
            given coordinate (default at the origin)

        2.  Centres the atoms within the unit cell

        ## ASSUMES THAT THE CELL IS ORTHOGONAL AND IS ALIGNED TO THE XY AXES ##

        blc     :   desired location for the bottom left corner of the supercell
                    in the xy plane

        """

        self._check_rectangular()

        # Find max distance between atoms along each axis indpendently
        cell_diff = np.ptp(self.xyz, axis = 0)

        # Find the dimensions of the rectangular unit cell along each axis
        cell_dims = np.linalg.norm(self.lat_vecs_sc, axis = 1)

        # Calculate the distance to shift the cell by
        shift = blc + 0.5 * (cell_dims- cell_diff) - np.amin(self.xyz, axis = 0)

        # set the `z' elements to be zero so that atoms stay in the z=0 plane
        # if is_set_z is False
        if not is_set_z:

            shift[-1] = 0

        # Fix the new shifted xyz coordinates

        self.xyz += shift


    def _do_rotation_xy(self, angle):
        """
        Applies rotation about the z-axis to a list of xyz coordinates and
        vectors

        Assumes we are transforming input with dimensions [N, 3], where N is the
        number of coordinates to be rotated for each data set

        """

        # Generate the rotation matrix around the z-axis for `angle'
        rot_mat = np.array([
            [np.cos(angle),   - np.sin(angle),      0],
            [np.sin(angle),     np.cos(angle),      0],
            [0,                 0,                  1]])

        # Rotate both the coordinates and the lattice vectors for the minimal
        # cell and supercell
        xyz_tmp = (rot_mat @ self.xyz.T).T

        lat_vecs_tmp = (rot_mat @ self.lat_vecs.T).T

        lat_vecs_sc_tmp = (rot_mat @ self.lat_vecs_sc.T).T

        return xyz_tmp, lat_vecs_tmp, lat_vecs_sc_tmp


    def rotate_xy(self, angle, is_radians = True, is_copy = False):
        """
        Rotates the system counterclockwise in the xy-plane and returns either
        a copy of the new data or updates the supercell object

        By default we assume the input angle to be in radians since the argument
        will more often be provided by another function

        """
        # If not already in radians, convert the angle so that it can be used
        if not is_radians:

            angle *= np.pi / 180.

        if is_copy:

            return self._do_rotation_xy(angle)

        else:

            self.xyz, self.lat_vecs, self.lat_vecs_sc = \
                self._do_rotation_xy(angle)


    def set_energy(self, pot = potential()):

        self.energy = pot.pot_func(self.xyz, self.sublat)


    def get_H(self, kdp, kdp_perp = 0, sys_data_list = None,
        is_wrap_finite = False):
        """
        Returns the Hamiltonian for this type of cell. By default assumes a
        single cell has been passed and takes the properties required from
        `self'.

        Can be passed the required parameters: 

        xyz, sublat, energy, lat_vecs_sc

        manually via kwargs if being called from the finite device class

        """
        # Set default 'is_periodic' behaviour
        is_periodic = True

        # If 'is_periodic' is provided in keywords, use that in the call.
        if 'is_periodic' in self.keywords.keys():

            is_periodic = self.keywords['is_periodic']

        # Check if being passed kwargs (i.e len(kwargs) > 0 if being passed),
        # if len(kwargs) == 0 assume input is from within the class
        if sys_data_list is None:

                return self._get_H_withParams(kdp, kdp_perp, self.xyz,
                    self.sublat, self.energy, self.lat_vecs_sc,
                    is_periodic = is_periodic, is_wrap_finite = is_wrap_finite)

        else:

            try:

                return self._get_H_withParams(kdp, kdp_perp,
                    is_periodic = is_periodic, is_wrap_finite = is_wrap_finite,
                    **sys_data_list)

            except:

                err_str = str(self.__class__.__name__) +'(): Problem in ' + \
                    '\'get_H_withParms()\''

                raise ValueError(err_str)


    def _get_H_withParams(self, kdp, kdp_perp, xyz, sublat, energy, lat_vecs_sc,
        is_periodic = True, is_wrap_finite = False, **kwargs):
        """
        The code to calculate the Hamiltonian for a monolayer graphene system
        with one periodic direction given xyz, sublat, energy and lat_vecs
        """

        # Number of atoms in the unit cell
        atno = len(xyz)

        # Form an n x n matrix of the atomic coordinates
        coords = np.repeat(xyz, atno, axis = 0).reshape(atno, atno, 3)

        # Array of distances between atoms within the same unit cell
        intra_cell = np.linalg.norm(
            coords - np.transpose(coords, (1,0,2)), axis = 2)

        # Array of distances between atoms, between this unit cell and the next
        # one in the positive periodic (x by default) direction
        inter_cell = np.linalg.norm(
            coords + lat_vecs_sc[0] - np.transpose(coords, (1,0,2)), axis = 2)

        # Define the n-n coupling stength and interatomic distance with scaling
        # factors included which retain the value of the dirac velocity
        t0 = 3.16 / self.scaling ; a_cc = 2.46 * self.scaling / np.sqrt(3)

        tol = 1E-3 # Tolerance for the inter-atom distance

        # Generate an array of zeros for us to populate
        ham = np.zeros((atno, atno), dtype = np.complex128)

        # -------------------------------------------------------------------- #

        # n-n coupling within the cell (intra_cell part)
        ham[np.abs(intra_cell - a_cc) < tol] += -t0

        # n-n coupling with the next (non-transport) cell if periodic:
        if is_periodic:

            self._get_H_gamma_0(a_cc, t0, ham, kdp, inter_cell, tol = tol)

        if is_wrap_finite:

            # intercell in 'transport' direction
            inter_cell_perp = np.linalg.norm(coords + lat_vecs_sc[1] -
                np.transpose(coords, (1,0,2)), axis = 2)

            # Calculate the periodic part in the 'transport' direction
            self._get_H_gamma_0(a_cc, t0, ham, kdp_perp, inter_cell_perp,
                tol = tol)

        # Fill on-site potentials
        for i in range(atno):

            ham[i,i] += energy[i]

        return ham


    def _get_H_gamma_0(self, a_cc, t0, ham, kdp, inter_cell, tol = 1E-3):
        """
        Adds periodic gamma_0 coupling to the Hamiltonian given a value for kdp
        (wavevector dotted with momentum) and an array giving the relevant
        interatomic distances within the cell. Adapted to allow for periodicity
        in both axes given a boolean choice

        """
        # Forwards
        ham[np.abs(inter_cell - a_cc) < tol] += \
            -t0 * np.exp(complex(0, + kdp))

        # Backwards
        ham[(np.abs(inter_cell - a_cc) < tol).T] += \
            -t0 * np.exp(complex(0, - kdp))


    def get_V(self):
        """
        Create the array which couples consecutive unit cells forwards

        """

        # Number of atoms in the unit cell
        atno = len(self.xyz)

        # Form an n x n matrix of the atomic coordinates
        coords = np.repeat(self.xyz, atno, axis = 0).reshape(atno, atno, 3)

        # Array of distances between atoms, between this unit cell and the next
        # one in the non-periodic direction
        inter_cell = np.linalg.norm( coords + self.lat_vecs_sc[1] - 
            np.transpose(coords, (1,0,2)), axis = 2)

        # Define the n-n coupling stength and interatomic distance
        t0 = 3.16 / self.scaling ; a_cc = 2.46 * self.scaling / np.sqrt(3)

        tol = 1E-3 # Tolerance for the inter-atom distance

        # Generate an array of zeros for us to populate
        v = np.zeros((atno, atno), dtype = np.float64)
        
        # n-n coupling forwards to the next cell
        v[np.abs(inter_cell - a_cc) < tol] += -t0

        return v


    ###                     Periodic edge sys functions                     ###


    def _get_periodic_cell(self):
        """
        Designed to operate on the periodic cell only (not ZZ or AC)

        """

        # Get the rectangular supercell lattice vectors
        ori, lv_sc = self._get_periodic_cell_vecs()

        # Generate ranges for each direction to iterate over
        dir1 = [ori[0,0], ori[1,0] + np.sign(ori[1,0]),\
            np.sign(ori[1,0] - ori[0,0])]

        dir2 = [ori[0,1], ori[1,1] + np.sign(ori[1,1]),\
            np.sign(ori[1,1]-ori[0,1])]

        # Generate a list of all minimal unit cell centres in the supercell
        centres = np.array([[i * self.lat_vecs[0] + j * self.lat_vecs[1] \
            for i in range(dir1[0], dir1[1], dir1[2])] \
                for j in range(dir2[0], dir2[1], dir2[2])])

        # Reshape into a list of xyz vectors to shift the unit cells by
        centres = centres.reshape(
            centres.shape[0] * centres.shape[1], centres.shape[2])

        # Fill the rest of the unit cell
        xyz = np.array([self.xyz + centre for centre in centres])

        # Repeat for all sublattice indexes
        sublat_idx = np.array([self.sublat for centre in centres])

        # Repeat for all energies
        energies = np.array([self.energy for centre in centres])

        # Reshape both
        xyz = xyz.reshape(xyz.shape[0] * xyz.shape[1], xyz.shape[2])

        sublat_idx = sublat_idx.reshape(sublat_idx.shape[0]*sublat_idx.shape[1])

        energies = energies.reshape(energies.shape[0] * energies.shape[1])

        # Use the path package in matplotlib to check if points are within our
        # unit cell
        path_vec_list = np.concatenate((lv_sc,-lv_sc), axis = 0)

        # start position + small amount to correct the numebr of points selected
        cell_start = np.zeros((3)) - 0.01

        cell_corners = np.array([cell_start + np.sum(path_vec_list[0:i],
            axis = 0) for i in range(len(path_vec_list))])

        p = path.Path(cell_corners[:,0:2])

        is_in = p.contains_points(xyz[:,0:2])

        return ori, lv_sc, xyz[is_in], energies[is_in], sublat_idx[is_in]


    def _get_periodic_cell_vecs(self):

        lv1 = self.orientation[0] * self.lat_vecs[0] + \
            self.orientation[1] * self.lat_vecs[1]

        # Make a list of numbers to keep a track of iterations
        # count[0] = Current count, count[1] = start count, count[2] = max count
        # count[0] should initially be the same as count[1]
        count = [1, 1, 10]

        while count[0] <= count[2]:

            if count[0] == count[1]:

                # Make an array of all initial combinations of vectors with the
                # corresponding indexes
                idx_list = np.linspace(-count[0], count[0], 2 * count[0] + 1)

                tmp = np.array([[np.append(
                    idx1 * self.lat_vecs[0] + idx2 * self.lat_vecs[1],
                    np.array([idx1,idx2,0])) for idx1 in idx_list] \
                    for idx2 in idx_list])

                tmp = tmp.reshape(tmp.shape[0] * tmp.shape[1], tmp.shape[2])

                tmp = tmp[np.logical_not((tmp[:,3:5] == [0,0]).all(axis = 1))]

            else:

                idx_list = np.linspace(
                    -(count[0]-1), (count[0]-1), 2*(count[0]-1) + 1)

                # Form an array which lists the ring of larger vectors not
                # previously considered, along with the corresponding indexes
                tmp = np.append(
                    np.array([np.append(
                        idx * self.lat_vecs[0] + count[0] * self.lat_vecs[1],
                        np.array([idx, count[0], 0])) for idx in idx_list]),
                    np.array([np.append(
                        count[0] * self.lat_vecs[0] + idx * self.lat_vecs[1],
                        np.array([count[0], idx, 0])) for idx in idx_list])
                        , axis = 0)

            # Check if the dot product of each vectors with the chosen edge is
            # zero. This would indicate that we have correctly found the
            # orthogonal supercell lattice vector
            if (np.abs(np.dot(tmp[:,0:3], lv1)) < 1E-5).any():

                # Pick out the correct superlattice vector
                lv2 = tmp[np.abs(np.dot(tmp[:,0:3], lv1)) < 1E-5][0,:3]

                idx = tmp[np.abs(np.dot(tmp[:,0:3], lv1)) < 1E-5][0,3:]

                # Generate a term that will ensure that the axes follow the
                # right hand rule
                sgn = np.sign(np.dot(np.cross(lv1, lv2), np.array([0,0,1])))

                indexes = np.array([self.orientation, (sgn * idx)]).astype(int)

                return indexes, np.array([lv1, sgn * lv2])

            count[0] += 1

        err_str = str(self.__class__.__name__) +'(): Reached max iteration \
            number in \'_get_periodic_cell_vecs\''

        raise ValueError(err_str)


    ################################## UTILITY #################################


    def make_ori_str(self):

        if type(self.orientation) == str:

            return self.orientation

        elif type(self.orientation) in [list, np.ndarray]:

            # slice first 3 values to save non zz/ac periodic orientation
            arr = np.array(self.orientation).flatten()[:3]

            return (''.join([str(arr[i]) + '_' for i in range(3)]))[:-1]

        else:

            err_str = str(self.__class__.__name__) + '(): Invalid orientation \
                input : ' + str(self.orientation)

            raise ValueError(err_str)


    def get_req_params(self):
        """ Return all required parameters for the generation of this cell """

        req_dict = {
            'latt_type'     :   'MLG',
            'orientation'   :   self.orientation,
            }

        # Check if 'is_periodic is provided'
        if 'is_periodic' in self.keywords.keys():

            ip = self.keywords['is_periodic']
            
            # Only save as a required parameter if it is fault. We assume it is
            # normally true
            if not ip:

                req_dict.update({'is_periodic'  :   ip})

        # If scaling is not one, make it a required parameter
        if self.scaling != 1:

            req_dict.update({'scaling'  :   self.scaling})

        return req_dict


class BLG_cell(MLG_cell):
    """
    Generates the unit cell for bilayer graphene. Will only be for zig-zag and
    armchair initially, but will be extended to larger systems
    """

    def __init__(self, index = 0, orientation = None, is_gamma_3 = True,
        scaling = 1, **kwargs):

        is_angelika = True # Reverse sublattice ordering

        self.scaling = scaling

        a = 2.46 * self.scaling ; d = a / (2 * np.sqrt(3)) ; a_z = 3.35

        layer_1_z = 0
        layer_2_z = a_z

        self.xyz = np.array([
            [0,         0,          layer_1_z   ],
            [d,       - a / 2,      layer_1_z   ],
            [3 * d,   - a / 2,      layer_1_z   ],
            [4 * d,     0,          layer_1_z   ],
            [d,       - a / 2,      layer_2_z   ],
            [2 * d,     0,          layer_2_z   ],
            [4 * d,     0,          layer_2_z   ],
            [5 * d,   - a / 2,      layer_2_z   ]
            ], np.float64)

        self.lat_vecs_sc = np.array([
            [6 * d,     0,      0       ],
            [0,         a,      0       ],
            [0,         0,      2 * a_z ]], np.float64)

        self.lat_vecs = np.array([
            [3 * d,   - a / 2,      0       ],
            [3 * d,   + a / 2,      0       ],
            [0,         0,          a_z     ]])

        self.energy = np.array([0] * 8)

        self.sublat = np.array([0, 1] * 4)

        if is_angelika:

            self.sublat = np.array([1, 0] * 4)

        self.cell_idx = index

        self.orientation = orientation

        self.rot_angle = 0

        self.keywords = kwargs

        self.is_gamma_3 = is_gamma_3

        if self.orientation is not None:
            self._align_to_axes()

        # Check the intended orientation
        if self.orientation is not None:

            # Get an int index based on the desired orientation. (Neatens code)
            ori_index = self._the_orientation()

            # If zig-zag, rotate by 90 degrees and check axis are aligned
            # correctly
            if ori_index == 0:

                self.rotate_xy(90, is_radians = False)

                self._align_ortho_lat_vecs()

            # If armchair, check axes are aligned correctly, but otherwise leave
            # alone
            elif ori_index == 1:

                pass

            # If periodic of ori_index takes any other value, exit and return
            # error statement
            else:

                err_str = str(self.__class__.__name__) +'(): Only \'zz\' or \
                    \'ac\' available for BLG_call, not ' + str(self.orientation)

                raise ValueError(err_str)

            # Centre atoms in the cell and fix cell with its bottom-left corner
            # in the x-y plane centred at the position `blc'

            # Check that the second supercell lattice vector is the one in the
            # y-direction
            if np.abs(self.lat_vecs_sc[1,1]) > 1E-5:

                # Calculate the location of the 'bottom left corner' of the cell
                blc = self.cell_idx * self.lat_vecs_sc[1]

            # Set the location of the cell and center atoms in the cell
            self._set_cell_corner(blc)


    def _get_H_withParams(self, kdp, kdp_perp, xyz, sublat, energy,
        lat_vecs_sc, is_periodic = True, is_wrap_finite = False, **kwargs):
        """
        The code to calculate the Hamiltonian for a monolayer graphene system
        with one periodic direction given xyz, sublat, energy and lat_vecs
        """

        ####     SET UP COORDINATE ARRAYS USED TO COUPLE RELEVANT ATOMS     ####

        # Number of atoms in the unit cell
        atno = len(xyz)

        # Form an n x n matrix of the atomic coordinates
        coords = np.repeat(xyz, atno, axis = 0).reshape(atno, atno, 3)

        # Array of vectors/distances between atoms within the same unit cell:
        # Coordinates
        intra_cell_xyz = coords - np.transpose(coords, (1,0,2))
        # Distances
        intra_cell = np.linalg.norm(intra_cell_xyz, axis = 2)

        # Array of vectors/distances between atoms, between this unit cell and
        # the next one in the positive periodic (x by default) direction:
        # Coordinates
        inter_cell_xyz = intra_cell_xyz + lat_vecs_sc[0]

        # Distances
        inter_cell = np.linalg.norm(inter_cell_xyz, axis = 2)

        if is_wrap_finite:
            
            inter_cell_perp = np.linalg.norm(intra_cell_xyz + lat_vecs_sc[1],
                axis = 2)

        # True if the coupled atoms are on the same layer
        is_same_layer = np.abs(intra_cell_xyz[:,:,2]) < 1E-2


        # ----------------- CREATE AND FILL THE HAMILTONIAN ------------------ #

        tol = 1E-3 # Tolerance for the inter-atom distance

        # Generate an array of zeros for us to populate
        ham = np.zeros((atno, atno), dtype = np.complex128)


        ####                            GAMMA 0                             ####

        t0 = 3.16 / self.scaling                   # Coupling strength
        a_cc = 2.46 * self.scaling / np.sqrt(3)    # Coupling distance

        # n-n coupling within the cell
        ham[np.logical_and(
            np.abs(intra_cell - a_cc) < tol, is_same_layer)] += -t0

        # Add coupling in the periodic directions if 'is_periodic'

        if is_periodic:

            self._get_H_gamma_0(a_cc, t0, ham, kdp, inter_cell, is_same_layer,
                tol)

        if is_wrap_finite:

            self._get_H_gamma_0(a_cc, t0, ham, kdp_perp, inter_cell_perp,
                is_same_layer, tol)

        ####                            GAMMA 1                             ####

        t1 = 0.39       # Coupling strength (not affected by scaling)
        a_z = 3.35      # Coupling distance

        # Couple sites in different layers that have the same x-y coordinate
        ham[np.logical_and(
            np.abs(intra_cell - a_z) < tol, is_same_layer == False)] += -t1

        ####                            GAMMA 3                             ####

        if self.is_gamma_3:

            t3 = 0.38 / self.scaling               # Coupling strength
            # Coupling distance w/ IMPLICIT SCALING from a_cc
            a_t3 = np.sqrt(a_cc ** 2 + a_z ** 2)

            # Sublat array repeated along one axis
            sublat_arr = np.repeat(
                sublat, ham.shape[0], axis = 0).reshape(ham.shape[:2])

            # Within the cell (intra_cell) - OPENS GAP AT K-POINT?
            ham[np.logical_and.reduce((
                np.abs(intra_cell - a_t3) < tol, # Check if correct distance
                sublat_arr != sublat_arr.T, # Check if on different sublattices
                is_same_layer == False # Check on different layers
                ))] += -t3

            if is_periodic:

                # Add periodic gamma_3 coupling in periodic direction
                self._get_H_gamma_3(a_t3, t3, ham, kdp, sublat_arr, inter_cell,
                    is_same_layer, tol)

            if is_wrap_finite:

                # Add periodic gamma_3 coupling in 'transport' direction
                self._get_H_gamma_3(a_t3, t3, ham, kdp, sublat_arr, 
                    inter_cell_perp, is_same_layer, tol, is_wrap_finite = True,
                    intra_cell_xyz = intra_cell_xyz, lat_vecs_sc = lat_vecs_sc,
                    kdp_perp = kdp_perp)

        ####                             ENERGY                             ####

        # Fill on-site potentials
        for i in range(atno): ham[i,i] += energy[i]

        return ham


    def _get_H_gamma_0(self, a_cc, t0, ham, kdp, inter_cell, is_same_layer,
        tol):
        """
        Couples in-plane nearest neighbour sites between cells

        """
        # n-n coupling with the next cell (IN THE PERIODIC DIRECTION):

        # FORWARDS
        ham[np.logical_and(
            np.abs(inter_cell - a_cc) < tol, is_same_layer)] \
            += -t0 * np.exp(complex(0, + kdp))

        # BACKWARDS - (works by using the transpose of the coord matrices since
        # the Hamiltonian is hermitian)
        ham[np.logical_and(
            np.abs(inter_cell - a_cc) < tol, is_same_layer).T] \
            += -t0 * np.exp(complex(0, - kdp))


    def _get_H_gamma_3_inner(self, a_t3, t3, ham, kdp, sublat_arr, inter_cell,
        is_same_layer, tol):
        """
        Fills in the elements of the hamiltonian which correspond to 
        the gamma_3 coupling in bilayer graphene

        """

        # gamma_3 coupling to the next cell in the PERIODIC direction
        # FORWARDS
        ham[np.logical_and.reduce((
            np.abs(inter_cell - a_t3) < tol, # Check if correct distance
            sublat_arr != sublat_arr.T, # Check if on different sublattices
            is_same_layer == False # Check on different layers
            ))] += -t3 * np.exp(complex(0, + kdp))


        # BACKWARDS
        ham[np.logical_and.reduce((
            np.abs(inter_cell - a_t3) < tol, # Check if correct distance
            sublat_arr != sublat_arr.T, # Check if on different sublattices
            is_same_layer == False # Check on different layers
            )).T] += -t3 * np.exp(complex(0, - kdp))


    def _get_H_gamma_3(self, a_t3, t3, ham, kdp, sublat_arr, inter_cell,
        is_same_layer, tol, is_wrap_finite = False, intra_cell_xyz = None,
        lat_vecs_sc = None, kdp_perp = 0):
        """
        Fills in the elements of the hamiltonian which correspond to 
        the gamma_3 coupling in bilayer graphene

        """

        self._get_H_gamma_3_inner(a_t3, t3, ham, kdp, sublat_arr,
            inter_cell, is_same_layer, tol)

        if is_wrap_finite:

            if intra_cell_xyz is None or lat_vecs_sc is None:

                err_str = str(self.__class__.__name__) +'(): Must provide ' + \
                '\'both intra_cell_xyz\' and \'lat_vecs_sc\' ' + \
                'when is_wrap_finite = True'

                raise ValueError(err_str)

            else:

                # Also iterate over the n-n diagonal cells since gamma_3 can
                # couple to these as well
                for i in [-1,1]:

                    for j in [-1,1]:

                        # Shift coordinates to correspond to a diagonal cell
                        inter_cell_tmp = np.linalg.norm(intra_cell_xyz +
                            i * lat_vecs_sc[0] + j * lat_vecs_sc[1], axis = 2)

                        # gamma_3 coupling to the next cell in the PERIODIC dir.
                        # FORWARDS
                        ham[np.logical_and.reduce((
                            np.abs(inter_cell_tmp - a_t3) < tol, # Check dist
                            sublat_arr != sublat_arr.T, # Check sublattices
                            is_same_layer == False # Check on different layers
                            ))] += \
                            -t3 * np.exp(complex(0, i * kdp + j * kdp_perp))


    def get_V(self, kdp = 0):
        """
        Create the array which couples consecutive unit cells forwards

        """
        ####     SET UP COORDINATE ARRAYS USED TO COUPLE RELEVANT ATOMS     ####

        # Number of atoms in the unit cell
        atno = len(self.xyz)

        # Form an n x n matrix of the atomic coordinates
        coords = np.repeat(self.xyz, atno, axis = 0).reshape(atno, atno, 3)

        # Array of vectors/distances between atoms, between this unit cell and
        # the next one in the positive periodic (x by default) direction:

        # Coordinates
        inter_cell_xyz = coords - np.transpose(coords, (1,0,2)) + \
            self.lat_vecs_sc[1]

        # Distances
        inter_cell = np.linalg.norm(inter_cell_xyz, axis = 2)

        # True if the coupled atoms are on the same layer
        is_same_layer = np.abs(inter_cell_xyz[:,:,2]) < 1E-2


        ####                CREATE AND FILL THE HAMILTONIAN                 ####

        tol = 1E-3 # Tolerance for the inter-atom distance

        # Generate an array of zeros for us to populate
        v = np.zeros((atno, atno), dtype = np.complex128)


        ####                            GAMMA 0                             ####

        t0 = 3.16 / self.scaling                       # Coupling strength
        a_cc = 2.46 * self.scaling / np.sqrt(3)        # Coupling distance

        # n-n coupling FORWARDS to the next cell ( NON - PERIODIC DIRECTION )
        self._get_V_gamma_0(a_cc, t0, v, inter_cell, is_same_layer, tol)

        ####                            GAMMA 3                             ####

        t3 = 0.38 / self.scaling                # Coupling strength
        a_z = 3.35                              # Vertical coupling distance
        a_t3 = np.sqrt(a_cc ** 2 + a_z ** 2)    # Coupling distance
        
        if self.is_gamma_3:

            # interlayer coupling FORWARDS to the next cell ( NON-PERIODIC DIR)
            self._get_V_gamma_3(a_t3, t3, v, kdp, inter_cell, inter_cell_xyz,
                is_same_layer, tol)

        return v


    def _get_V_gamma_0(self, a_cc, t0, v, inter_cell, is_same_layer, tol):
        """ Same-layer nearest neighbour coupling between atoms """

        v[np.logical_and(np.abs(inter_cell - a_cc) < tol, is_same_layer)] += -t0


    def _get_V_gamma_3(self, a_t3, t3, v, kdp, inter_cell, inter_cell_xyz,
        is_same_layer, tol):
        """
        Fills all elements which couple sites via gamma_3 int he non-periodic
        direction

        """
        atno = len(self.xyz)

        # Sublat array repeated along one axis
        sublat_arr = np.repeat(self.sublat, atno, axis = 0).reshape(atno, atno)

        # gamma_3 coupling FORWARDS to cell + 1 ( NON - PERIODIC DIRECTION )
        v[np.logical_and.reduce((
            np.abs(inter_cell - a_t3) < tol, # Check if correct distance
            sublat_arr != sublat_arr.T, # Check if on different sublattices
            is_same_layer == False # Check on different layers
            ))] += -t3

        # ITERATE OVER NEIGHBOURING CELLS TO CHECK FOR DIAGONAL COUPLING
        # BETWEEN CELLS

        for i in [-1,1]:

            # gamma_3 coupling FORWARDS to next cell
            # ( NON - PERIODIC DIRECTION ) including phase picked up by
            # moving in the periodic direction

            # Shift coordinates to correspond to a diagonal cell
            inter_cell = np.linalg.norm(
                inter_cell_xyz + i * self.lat_vecs_sc[0], axis = 2)

            # If there is a matching atom, assign a coupling with a phase
            v[np.logical_and.reduce((
                np.abs(inter_cell - a_t3) < tol, # Check if correct distance
                sublat_arr != sublat_arr.T, # Check if on different sublats
                is_same_layer == False # Check on different layers
                ))] += -t3 * np.exp(complex(0, i * kdp))


    ################################## UTILITY #################################


    def get_req_params(self):
        """ Return all required parameters for the generation of this cell """

        req_dict = super().get_req_params()

        req_dict['latt_type'] = 'BLG'

        req_dict_extra = {'is_gamma_3'  :   self.is_gamma_3}

        return {**req_dict, **req_dict_extra}


# ----------------------------- MAKE THE CELLS ------------------------------- #

def stripe(idx, latt_type, orientation, stripe_len, **kwargs):
    """
    Returns the minimal orthogonal cell which has been repeated a set number of
    times in the non-transport direction

    """

    # Repeat the lattice locations up to the number of cells in the stripe,
    # shifting by one lattice vector length each time
    cell = latt_type(idx, orientation, **kwargs)

    cell.stripe_len = stripe_len

    cell.xyz = np.concatenate([cell.xyz + cell.lat_vecs_sc[0] * i
        for i in range(stripe_len)])

    # Update the relevant superlattice vector
    cell.lat_vecs_sc[0] *= stripe_len

    # Center the stripe around the origin
    cell.xyz -= cell.lat_vecs_sc[0] / 2

    # increase the length of self.sublat to correspond to the new size of the
    # cell
    cell.sublat = np.concatenate([cell.sublat for i in range(stripe_len)])

    return cell


def min_ortho_cell(idx, latt_type, orientation, **kwargs):
    """ Returns the minimal orthogonal cell of the requested latt_type """

    return latt_type(idx, orientation, **kwargs)


# ---------------------------------------------------------------------------- #


def __main__():
    dir_ext = 'saved_files/'

    cell_num = 1

    # Dictionary of paramters used to define the dev (no potential)
    dev_kwargs = {
        'is_gamma_3'    :   True,           # On/off gamma 3 coupling in BLG
        'latt_type'     :   BLG_cell,       # Pick a lattice type (MLG_cell,
                                            # BLG_cell) from grpahene_supercell
        'cell_func'     :   min_ortho_cell, # min_ortho_cell vs stripe
        'cell_num'      :   cell_num,       # Pick the number of cells in the
                                            # transport direction
        'stripe_len'    :   20,             # num of cells to repeat in stripe
        'is_periodic'   :   True,           # Periodic in non-trnsprt direction?
        'is_wrap_finite':   True,          # Whether to wrap the finite system
                                            # into a torus
        'orientation'   : 'zz'              # orientation of the cells
        }

    blgc = BLG_cell(0, **dev_kwargs)

    # Code to plot the cells of the bilayer to look at how to make hoppings

    xyz2 = np.concatenate(
        [blgc.xyz + i * blgc.lat_vecs_sc[0] + j * blgc.lat_vecs_sc[1]
        for i in range(2) for j in range(2)])
    dict_tmp = {'xyz':xyz2,'vecs':blgc.lat_vecs_sc[:,:2]}

    savemat('xyz_dat', dict_tmp)


    fig = plt.figure()

    ax1 = fig.add_subplot(311)
    ax1.scatter(
        blgc.xyz[np.logical_and(blgc.xyz[:,2] == 0, blgc.sublat == 0)][:,0],
        blgc.xyz[np.logical_and(blgc.xyz[:,2] == 0, blgc.sublat == 0)][:,1],
        c = 'C7', label = 'A1')
    ax1.scatter(
        blgc.xyz[np.logical_and(blgc.xyz[:,2] == 0, blgc.sublat == 1)][:,0],
        blgc.xyz[np.logical_and(blgc.xyz[:,2] == 0, blgc.sublat == 1)][:,1],
        c = 'C8', label = 'B1')
    ax1.legend(fancybox = True, shadow = True, bbox_to_anchor = (1, 1))

    ax2 = fig.add_subplot(312)
    ax2.scatter(
        blgc.xyz[np.logical_and(blgc.xyz[:,2] != 0, blgc.sublat == 0)][:,0],
        blgc.xyz[np.logical_and(blgc.xyz[:,2] != 0, blgc.sublat == 0)][:,1],
        c = 'C1', label = 'A2')
    ax2.scatter(
        blgc.xyz[np.logical_and(blgc.xyz[:,2] != 0, blgc.sublat == 1)][:,0],
        blgc.xyz[np.logical_and(blgc.xyz[:,2] != 0, blgc.sublat == 1)][:,1],
        c = 'C5', label = 'B2')
    ax2.legend(fancybox = True, shadow = True, bbox_to_anchor = (1, 1))

    ax3 = fig.add_subplot(313)
    ax3.scatter(
        blgc.xyz[np.logical_and(blgc.xyz[:,2] == 0, blgc.sublat == 0)][:,0],
        blgc.xyz[np.logical_and(blgc.xyz[:,2] == 0, blgc.sublat == 0)][:,1],
        c = 'C7')
    ax3.scatter(
        blgc.xyz[np.logical_and(blgc.xyz[:,2] == 0, blgc.sublat == 1)][:,0],
        blgc.xyz[np.logical_and(blgc.xyz[:,2] == 0, blgc.sublat == 1)][:,1],
        c = 'C8')
    ax3.scatter(
        blgc.xyz[np.logical_and(blgc.xyz[:,2] != 0, blgc.sublat == 0)][:,0],
        blgc.xyz[np.logical_and(blgc.xyz[:,2] != 0, blgc.sublat == 0)][:,1],
        c = 'C1')
    ax3.scatter(
        blgc.xyz[np.logical_and(blgc.xyz[:,2] != 0, blgc.sublat == 1)][:,0],
        blgc.xyz[np.logical_and(blgc.xyz[:,2] != 0, blgc.sublat == 1)][:,1],
        c = 'C5')

    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax3.set_aspect('equal')


    if dev_kwargs['orientation'] == 'ac':
        x_rng = [0, 1.42 * 3]
        y_rng = [0, 2.46]
    else:
        x_rng = [0, 2.46]
        y_rng = [0, 1.42 * 3]

    ax1.set_xlim(x_rng)
    ax1.set_ylim(y_rng)
    ax2.set_xlim(x_rng)
    ax2.set_ylim(y_rng)
    ax3.set_xlim(x_rng)
    ax3.set_ylim(y_rng)

    plt.show()

    # ------------------------------------------------------------------------ #

    tmp = []

    for ori in ['zz', 'ac']:

        dev_kwargs['orientation'] = ori
        dev_kwargs['is_gamma_3'] = True
        dev_kwargs['cell_num'] = 1

        iwf = True

        blgc = BLG_cell(0, **dev_kwargs)

        kx_rng = np.linspace(-np.pi, np.pi, 100)

        kp = - (4 * np.pi / (np.sqrt(3) * 2.46)) / 1.42

        print(kp)

        np.append(kx_rng, kp)

        if ori == 'zz':

            tmp.append(blgc.get_H(kp, 0, is_wrap_finite = iwf))

        else:

            tmp.append(blgc.get_H(0, -kp, is_wrap_finite = iwf))

        kx_rng = np.sort(kx_rng)

        tab1 = [[kx, linalg.eigvalsh(
            blgc.get_H(kx, 0, is_wrap_finite = iwf))] for kx in kx_rng]
        tab2 = [[kx, linalg.eigvalsh(
            blgc.get_H(0, kx, is_wrap_finite = iwf))] for kx in kx_rng]

        l = len(tab1[0][1])

        tab1 = [[[kx, vals[i]] for kx, vals in tab1] for i in range(l)]
        tab2 = [[[kx, vals[i]] for kx, vals in tab2] for i in range(l)]

        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        for i in range(l):
            x1, y1 = zip(*tab1[i])
            x2, y2 = zip(*tab2[i])

            ax1.plot(x1,y1)
            ax2.plot(x2,y2)

        ax1.set_ylim([-10,10])
        ax2.set_ylim([-10,10])

    plt.show()

    print(tmp[0][0,7])
    print(tmp[0][7,0])

    print(tmp[1][0,7])
    print(tmp[1][7,0])

    print(abs(tmp[0][7,0]))
    print(abs(tmp[1][7,0]))

    tol = 1E-2

    print(abs(tmp[0] - tmp[1]) < tol)

    print('All elements equal? : ', (np.abs(tmp[0] - tmp[1]) < tol).all())


if __name__ == "__main__":
    __main__()