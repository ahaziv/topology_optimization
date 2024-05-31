import numpy as np
from typing import Optional, List, Dict
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import os


# source - https://core.ac.uk/reader/39324130
class TopologyOptimizer:
    def __init__(self, nel_x: int, nel_y: int, tau, boundaries: List[Dict], forces: List[Dict]):
        self.young_modulus_mat = 1
        self.young_modulus_void = 1e-4
        self.poisson_rat = 0.3
        self.volume_iter_num = 100
        self.dt = 0.1  # step size for fictitious time t
        self.d = -0.02  # augmented lagrangian parameter
        self.p = 4  # augmented lagrangian parameter
        self.nel_x = nel_x
        self.nel_y = nel_y
        self.tau = tau
        self.init_forces_and_bcs()
        self.boundary_indices = set()
        self.ext_force_indices = set()
        self.boundaries = boundaries
        self.forces = forces

    @staticmethod
    def load_data(var_name):
        data = np.genfromtxt(os.path.join(os.getcwd(), var_name), delimiter=',')
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        return data

    def init_forces_and_bcs(self):
        self.ext_force = lil_matrix((2 * (self.nel_y + 1) * (self.nel_x + 1), 1), dtype=np.float64)
        # ext_force is the force vector in the x-y dimension, where the even members represent forces in the x-axis
        # and the odd represent forces in the y-axis
        self.U = np.zeros((2 * (self.nel_y + 1) * (self.nel_x + 1)), dtype=np.float64)
        self.freedofs = np.arange(0, 2 * (self.nel_y + 1) * (self.nel_x + 1))
        self.freedofsphi = np.arange(0, (self.nel_y + 1) * (self.nel_x + 1))

    @staticmethod
    def check_bc_and_force_params_validity(face: str, start_loc: float, end_loc: float, direction: str = 'down'):
        valid_choices = ('up', 'down', 'left', 'right')
        if face not in valid_choices:
            raise ValueError(f'Invalid boundary on face {face}, please choose from: {valid_choices}')
        if direction not in valid_choices:
            raise ValueError(f'Invalid force direction {direction}, please choose from: {valid_choices}')
        if start_loc < 0. or end_loc > 1. or start_loc >= end_loc:
            raise ValueError(f'Invalid combination for start and end, please make sure 0 <= start < end <= 1')

    def find_boundary_indices(self, face: str, start_loc: float, end_loc: float, direction: str):
        """
        element indexes:
                                the i * 2(nel_y + 1) - 2 indices (i > 0)
                            ————————————————————————
                            |                       |
        first nel_y indices |                       | last nel_y indices
                            |                       |
                            ————————————————————————
                               the i * 2(nel_y + 1) indices (i >= 0)

        where even indices represent force/displacement along the x-axis
        and odd indices represent force/displacement along the y-axis
        """
        dir_idx = 0 if direction in ('right', 'left') else 1
        if face == 'left':
            start_location_idx = round(2 * (self.nel_x + 1) * start_loc)
            end_location_idx = round(2 * (self.nel_x + 1) * end_loc)
            indices = np.arange(start_location_idx + dir_idx, end_location_idx + dir_idx, 2)

        elif face == 'right':
            start_location_idx = 2 * (self.nel_y + 1) * self.nel_x + round(2 * self.nel_y * start_loc)
            end_location_idx = 2 * (self.nel_y + 1) * self.nel_x + round(2 * self.nel_y * end_loc)
            indices = np.arange(start_location_idx + dir_idx, end_location_idx + dir_idx + 1, 2)

        elif face == 'down':
            start_location_idx = 2 * (self.nel_y + 1) * round(self.nel_x * start_loc)
            end_location_idx = 2 * (self.nel_y + 1) * round(self.nel_x * end_loc) + 1
            indices = np.arange(start_location_idx + dir_idx, end_location_idx + dir_idx, 2 * (self.nel_y + 1))

        else:   # 'up'
            start_location_idx = 2 * (self.nel_y + 1) * (1 + round(self.nel_x * start_loc)) - 2
            end_location_idx = 2 * (self.nel_y + 1) * (1 + round(self.nel_x * end_loc)) - 1
            indices = np.arange(start_location_idx + dir_idx, end_location_idx + dir_idx, 2 * (self.nel_y + 1))

        return indices

    def add_boundary_condition(self, face: str, start_loc: float, end_loc: float, direction: str):
        self.check_bc_and_force_params_validity(face, start_loc, end_loc)
        fixeddofs = self.find_boundary_indices(face, start_loc, end_loc, direction)
        fixeddofs_set = set(fixeddofs)
        if self.ext_force_indices.intersection(fixeddofs_set):
            raise UserWarning('An overlapping external force and BC were found, check for redundancy.')
        self.boundary_indices.update(fixeddofs_set)
        self.freedofs = np.setdiff1d(self.freedofs, fixeddofs)

    def add_external_force(self, face: str, start_loc: float, end_loc: float, direction: str,
                           magnitude: Optional[float] = 1.):
        self.check_bc_and_force_params_validity(face, start_loc, end_loc, direction)
        indices = self.find_boundary_indices(face, start_loc, end_loc, direction)
        force_sign = magnitude if direction in ('right', 'up') else -magnitude
        # indices_set = set(indices)
        # if self.boundary_indices.intersection(indices_set):
        #     raise UserWarning('An overlapping external force and BC were found, check for redundancy.')
        self.ext_force_indices.update(set(indices))
        self.ext_force[indices, 0] = force_sign

    def displacement_field(self):
        # displacement_field
        A11 = np.array([[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]])
        A12 = np.array([[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]])
        B11 = np.array([[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]])
        B12 = np.array([[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]])

        self.KE = 1 / (1 - self.poisson_rat ** 2) / 24 \
                  * (np.block([[A11, A12], [np.transpose(A12), A11]]) + self.poisson_rat
                     * np.block([[B11, B12], [np.transpose(B12), B11]]))
        # topological derivative
        a1 = 3 * (1 - self.poisson_rat) / (2 * (1 + self.poisson_rat) * (7 - 5 * self.poisson_rat)) \
             * (-(1 - 14 * self.poisson_rat + 15 * self.poisson_rat ** 2) * self.young_modulus_mat) / \
             (1 - 2 * self.poisson_rat) ** 2
        a2 = 3 * (1 - self.poisson_rat) / (
                    2 * (1 + self.poisson_rat) * (7 - 5 * self.poisson_rat)) * 5 * self.young_modulus_mat
        self.A = (a1 + 2 * a2) / 24 * (np.block([[A11, A12], [np.transpose(A12), A11]]) + (a1 / (a1 + 2 * a2))
                                  * np.block([[B11, B12], [np.transpose(B12), B11]]))

    def plot_beam(self, material_dist):
        rgb_dist = np.stack((-material_dist + 1,)*3, axis=-1)
        for i, index in enumerate(self.boundary_indices):
            x, y = int(index / (2 * self.nel_y + 2) - 1), int((index - 1) / 2) % (self.nel_y + 1)
            if 0 <= x < rgb_dist.shape[1] and 0 <= y < rgb_dist.shape[0]:
                rgb_dist[y, x] = [0, 0, 1]
            if i == int(len(self.boundary_indices) / 2):
                mid_boundary_index = (x, y)
        for i, index in enumerate(self.ext_force_indices):
            x, y = int(index / (2 * self.nel_y + 2) - 1), int((index - 1) / 2) % (self.nel_y + 1)
            if 0 <= x < rgb_dist.shape[1] and 0 <= y < rgb_dist.shape[0]:
                rgb_dist[y, x] = [1, 0, 0]
            if i == int(len(self.ext_force_indices) / 2):
                mid_force_index = (x, y)

        # TODO - remove after testing
        # for x in range(50):
        #     rgb_dist[0, x] = [0, 1, 0]

        # Plot the array
        plt.imshow(rgb_dist, aspect='auto', origin='lower')
        plt.axis('equal')
        plt.axis('tight')
        plt.xlim((-20, material_dist.shape[1] + 20))
        plt.ylim((-20, material_dist.shape[0] + 20))
        plt.title(f'nel_x: {self.nel_x}, nel_y: {self.nel_y}')
        plt.arrow(mid_force_index[0], mid_force_index[1], 10, 10,
                  fc='red', ec='red', length_includes_head=True, head_width=2, head_length=5)
        plt.text(mid_force_index[0] + 10, mid_force_index[1] + 10, 'F', color='red')
        plt.text(mid_boundary_index[0], mid_boundary_index[1], 'Fixed Boundary', color='blue')
        plt.show(block=False)
        plt.pause(0.1)

    def level_set_optimize(self, volume_max):
        """
        nelx, nely - number of elements in x/y direction
        volume_max - maximum allowable volume
        tau - regularization parameter
        """
        self.phi = np.ones((self.nel_y + 1) * (self.nel_x + 1))      # level set function
        material_dist = np.ones((self.nel_y, self.nel_x))       # material distribution

        vol_init = np.sum(material_dist) / (self.nel_x * self.nel_y)

        # finite element analysis preparation
        self.displacement_field()
        nodenrs = np.reshape(np.arange(0, (1 + self.nel_x) * (1 + self.nel_y)), (1 + self.nel_y, 1 + self.nel_x), order='F')
        edofVec = np.reshape(2 * nodenrs[0:-1, 0:-1] + 2, (self.nel_x * self.nel_y, 1), order='F')          # TODO - is the +2 necessary?
        edofMat = np.tile(np.array([0, 1, 2 * self.nel_y + 2, 2 * self.nel_y + 3, 2 * self.nel_y, 2 * self.nel_y + 1, -2, -1]), (self.nel_x * self.nel_y, 1)) + np.tile(edofVec, (1, 8))
        iK = np.reshape(np.kron(edofMat, np.ones((8, 1))).T, 64 * self.nel_x * self.nel_y, order='F')
        jK = np.reshape(np.kron(edofMat, np.ones((1, 8))).T, 64 * self.nel_x * self.nel_y, order='F')

        # for reaction diffusion function
        NNdife = (1 / 6) * np.array([[4, -1, -2, -1],
                                     [-1, 4, -1, -2],
                                     [-2, -1, 4, -1],
                                     [-1, -2, -1, 4]])

        NNe = (1 / 36) * np.array([[4, 2, 1, 2],
                                   [2, 4, 2, 1],
                                   [1, 2, 4, 2],
                                   [2, 1, 2, 4]])

        edofVec2 = np.reshape(nodenrs[0:-1, 0:-1] + 1, (self.nel_x * self.nel_y, 1), order='F')
        edofMat2 = np.tile(edofVec2, (1, 4)) + np.tile([0, self.nel_y + 1, self.nel_y, -1], (self.nel_x * self.nel_y, 1))

        iN = np.reshape(np.kron(edofMat2, np.ones((4, 1))).T, 16 * self.nel_x * self.nel_y, order='F')
        jN = np.reshape(np.kron(edofMat2, np.ones((1, 4))).T, 16 * self.nel_x * self.nel_y, order='F')

        sNN = np.reshape(np.tile(NNe.flatten(), (1, self.nel_y * self.nel_x)), 16 * self.nel_x * self.nel_y, order='F')
        NN = csr_matrix((sNN, (iN, jN)), shape=(int(max(iN)) + 1, int(max(jN)) + 1))

        sNNdif = np.reshape(np.outer(NNdife.flatten(), np.ones((1, self.nel_y * self.nel_x))), 16 * self.nel_x * self.nel_y, order='F')
        NNdif = csr_matrix((sNNdif, (iN, jN)), shape=(int(max(iN)) + 1, int(max(jN)) + 1))

        for boundary in self.boundaries:
            self.add_boundary_condition(face=boundary['face'], start_loc=boundary['start_loc'],
                                        end_loc=boundary['end_loc'], direction=boundary['direction'])
        for force in self.forces:
            self.add_external_force(face=force['face'], start_loc=force['start_loc'],
                                    end_loc=force['end_loc'], direction=force['direction'])

        fixeddofsphi = np.sort(np.concatenate((
            np.arange(0, self.nel_y + 1),
            np.arange(self.nel_y + 1, (self.nel_y + 1) * self.nel_x, self.nel_y + 1),
            np.arange(2 * (self.nel_y + 1) - 1, (self.nel_y + 1) * self.nel_x, self.nel_y + 1),
            np.arange((self.nel_y + 1) * self.nel_x, (self.nel_y + 1) * (self.nel_x + 1))
        )))
        self.freedofsphi = np.setdiff1d(self.freedofsphi, fixeddofsphi)
        self.phi[fixeddofsphi] = 0

        stiffness_mat = NN.multiply(1 / self.dt) + NNdif.multiply(self.tau * self.nel_y * self.nel_x)

        self.ext_force = self.ext_force.tocsr()
        norm_factor = 1e-4
        # main loop
        for iterNum in range(1, 201):
            sK_calc_var = np.outer(self.KE, self.young_modulus_void + np.ravel(material_dist, order='F')*(self.young_modulus_mat - self.young_modulus_void))
            sK = np.reshape(sK_calc_var, 64 * self.nel_x * self.nel_y, order='F')
            K = csr_matrix((sK, (iK, jK)), shape=(int(max(iK)) + 1, int(max(jK)) + 1))      # K - stiffness matrix
            K = (K + K.T) / 2

            self.U[self.freedofs] = spsolve(K[self.freedofs, :][:, self.freedofs], self.ext_force[self.freedofs])

            SED = (self.young_modulus_void + material_dist * (self.young_modulus_mat - self.young_modulus_void)) * np.reshape(np.sum((self.U[edofMat] @ self.KE) * self.U[edofMat], axis=1), (self.nel_y, self.nel_x), order='F')
            TD = (norm_factor + material_dist * (1 - norm_factor)) * np.reshape(np.sum((self.U[edofMat] @ self.A) * self.U[edofMat], axis=1), (self.nel_y, self.nel_x), order='F')
            td2 = np.block([[TD[0, 0], TD[0, :], TD[0, -1]], [TD[:, 0].reshape(-1, 1), TD, TD[:, -1].reshape(-1, 1)], [TD[-1, 0], TD[-1, :], TD[-1, -1]]])
            TDN = 0.25 * (td2[0:-1, 0:-1] + td2[1:, 0:-1] + td2[0:-1, 1:] + td2[1:, 1:])

            objective = np.sum(SED)
            vol = np.sum(material_dist) / (self.nel_x * self.nel_y)

            print(f'It.: {iterNum} Compl.: {objective / (self.nel_x * self.nel_y):.4e} Vol.: {vol:.2f}')

            self.plot_beam(material_dist)

            if iterNum > self.volume_iter_num and (abs(vol - volume_max) < 0.005) and all(
                    abs(objective - objective[-6:-1]) < 0.01 * abs(objective[-1])):
                break

            # set augmented lagrangian parameters
            ex = volume_max + (vol_init - volume_max) * max(0., 1 - iterNum / self.volume_iter_num)
            lambda_val = np.sum(np.sum(TDN)) / ((self.nel_y + 1) * (self.nel_x + 1)) * np.exp(self.p * ((vol - ex) / ex + self.d))
            C = 1 / np.sum(np.abs(TDN)) * (self.nel_y * self.nel_x)
            g2 = np.reshape(TDN, (self.nel_y + 1) * (self.nel_x + 1), order='F')

            # update level set function
            Y = NN @ (C * (g2 - lambda_val * np.ones_like(g2)) + self.phi / self.dt)
            self.phi[self.freedofsphi] = spsolve(stiffness_mat[self.freedofsphi, :][:, self.freedofsphi], Y[self.freedofsphi])
            self.phi = np.minimum(1, np.maximum(-1, self.phi))

            phin = np.reshape(self.phi, (self.nel_y + 1, self.nel_x + 1), order='F')
            phie = 0.25 * (phin[0:-1, 0:-1] + phin[1:, 0:-1] + phin[0:-1, 1:] + phin[1:, 1:])
            material_dist[:, :] = (phie[:, :] > 0)


# -------------------------------------------- main ------------------------------------------------------------------ #
