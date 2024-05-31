from topology_optimizer import TopologyOptimizer
from use_case_generators import *

voxel_num = 7560
outer_nelx, outer_nely = get_random_board(voxel_num)
outer_Vmax = 0.5
outer_tau = 6e-4
(boundaries, forces) = get_random_bcs_and_forces(outer_nelx, outer_nely)

# boundaries = [{'face': 'left', 'start_loc': 0., 'end_loc': 0.5, 'direction': 'up'},
#               {'face': 'left', 'start_loc': 0., 'end_loc': 0.5, 'direction': 'right'},]
# forces = [{'face': 'right', 'start_loc': 0.5, 'end_loc': 1., 'direction': 'up', 'magnitude': -0.6606237819101702},
#           {'face': 'right', 'start_loc': 0.5, 'end_loc': 1., 'direction': 'right', 'magnitude': -0.750717136326795}]

print(f'boundaries: {boundaries}\nforces: {forces}')
optimizer = TopologyOptimizer(outer_nelx, outer_nely, outer_tau, boundaries, forces)
optimizer.level_set_optimize(outer_Vmax)

temp = 1