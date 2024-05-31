from typing import Tuple, List, Dict, Optional
import numpy as np
import copy

def get_random_board(total_voxels: int) -> Tuple[int, int]:
    divisor_dict = {27720: [(105, 264), (110, 252), (120, 231), (126, 220), (132, 210), (140, 198), (154, 180), (165, 168)],
                    20160: [(90, 240), (90, 224), (96, 210), (105, 192), (112, 180), (120, 168), (126, 160), (140, 144)],
                    15120: [(72, 210), (80, 189), (84, 180), (90, 168), (105, 144), (108, 140), (112, 135), (120, 126)],
                    10080: [(60, 168), (63, 160), (70, 144), (72, 140), (80, 126), (84, 120), (90, 112), (96, 105)],
                    7560: [(54, 140), (56, 135), (60, 126), (63, 120), (70, 108), (72, 105), (84, 90)],
                    5040: [(42, 120), (45, 112), (48, 105), (56, 90), (60, 84), (63, 80), (70, 72)]}
    divisors = divisor_dict[total_voxels]
    (num_1, num_2) = divisors[np.random.randint(len(divisors))]
    nel_x, nel_y = (num_1, num_2) if np.random.choice([True, False]) else (num_2, num_1)
    return nel_x, nel_y


def random_section(total_elem_num: int, max_length: int, pivot: int) -> (int, int):
    min_length = 5
    min_idx = (pivot - np.random.randint(min_length, max_length / 2)) % total_elem_num
    max_idx = (pivot + np.random.randint(min_length, max_length / 2)) % total_elem_num
    return min_idx, max_idx


def reverse_directions(components: List[Dict]):
    for component in components:
        if component['face'] in ('right', 'down'):
            start_loc, end_loc = component['start_loc'], component['end_loc']
            component['start_loc'] = 1. - end_loc
            component['end_loc'] = 1. - start_loc


def translate_elements(nel_x: int, nel_y: int, min_idx: int, max_idx: int,
                       force: Optional[Tuple[float, float]] = None) -> List[Dict]:
    """ component := {face, start_loc, end_loc, direction} """
    total_elem_num = 2 * (nel_x - 1) + 2 * (nel_y - 1)
    components = [{}]
    faces = [(nel_y, 'left'), (nel_x - 1, 'up'), (nel_y - 1, 'right'), (nel_x - 2, 'down'), (nel_y, 'left')]
    el_sum = 0
    for i, (face_el_num, face) in enumerate(faces):
        el_sum += face_el_num
        if min_idx > el_sum:
            continue
        components[0].update({'face': face,
                              'start_loc': (min_idx + face_el_num - el_sum) / face_el_num})

        if min_idx > max_idx:
            max_idx += total_elem_num
        if max_idx <= el_sum:
            components[0]['end_loc'] = (max_idx + face_el_num - el_sum) / face_el_num
        else:
            components[0]['end_loc'] = 1.
            face_el_num, face = faces[i + 1] if i < 3 else faces[0]
            el_sum += face_el_num
            components.append({'face': face,
                               'start_loc': 0.,
                               'end_loc': (max_idx + face_el_num - el_sum) / face_el_num})
        break

    reverse_directions(components)

    directional_components = []
    for component in components:
        directional_components += [copy.deepcopy(component), copy.deepcopy(component)]
        directional_components[-1]['direction'] = 'right'
        directional_components[-2]['direction'] = 'up'
        if force:
            directional_components[-1]['magnitude'] = force[0]
            directional_components[-2]['magnitude'] = force[1]

    return directional_components


def get_random_bcs_and_forces(nel_x: int, nel_y: int) -> Tuple[List[Dict], List[Dict]]:
    max_length = min(nel_x, nel_y)
    total_elem_num = 2 * (nel_x - 1) + 2 * (nel_y - 1)
    pivot_candidates = list(range(nel_y)) + list(range(nel_y + nel_x - 1, 2 * nel_y + nel_x - 2)) if nel_y < nel_x \
        else list(range(nel_y, nel_y + nel_x - 1)) + list(range(2 * nel_y + nel_x - 2, 2 * nel_y + 2 * nel_x - 4))
    bc_pivot = pivot_candidates[np.random.randint(0, len(pivot_candidates) - 1)]
    bc_min_idx, bc_max_idx = random_section(total_elem_num, max_length, bc_pivot)
    bcs = translate_elements(nel_x, nel_y, bc_min_idx, bc_max_idx)
    f_pivot = bc_pivot + int(total_elem_num / 2) % total_elem_num
    f_min_idx, f_max_idx = random_section(total_elem_num, max_length, f_pivot)
    theta = np.random.random() * 2 * np.pi
    force_vec = (np.cos(theta), np.sin(theta))
    forces = translate_elements(nel_x, nel_y, f_min_idx, f_max_idx, force_vec)
    return bcs, forces
