import importlib
import time
import os

import torch
import numpy as np

def load_checkpoint(checkpoint_path, device):
    _, ext = os.path.splitext(os.path.basename(checkpoint_path))
    assert ext in (".pth", ".tar"), "Only support ext and tar extensions of model checkpoint."
    model_checkpoint = torch.load(checkpoint_path, map_location=device)

    if ext == ".pth":
        print(f"Loading {checkpoint_path}.")
        return model_checkpoint
    else:  # tar
        print(f"Loading {checkpoint_path}, epoch = {model_checkpoint['epoch']}.")
        return model_checkpoint["model"]


def prepare_empty_dir(dirs, resume=False):
    """
    if resume experiment, assert the dirs exist,
    if not resume experiment, make dirs.

    Args:
        dirs (list): directors list
        resume (bool): whether to resume experiment, default is False
    """
    for dir_path in dirs:
        if resume:
            assert dir_path.exists()
        else:
            dir_path.mkdir(parents=True, exist_ok=True)


class ExecutionTime:
    """
    Usage:
        timer = ExecutionTime()
        <Something...>
        print(f'Finished in {timer.duration()} seconds.')
    """

    def __init__(self):
        self.start_time = time.time()

    def duration(self):
        return int(time.time() - self.start_time)


def initialize_config(module_cfg, pass_args=True):
    """According to config items, load specific module dynamically with params.
    e.g., Config items as follow：
        module_cfg = {
            "module": "model.model",
            "main": "Model",
            "args": {...}
        }
    1. Load the module corresponding to the "module" param.
    2. Call function (or instantiate class) corresponding to the "main" param.
    3. Send the param (in "args") into the function (or class) when calling ( or instantiating)
    """
    module = importlib.import_module(module_cfg["module"])

    if pass_args:
        return getattr(module, module_cfg["main"])(**module_cfg["args"])
    else:
        return getattr(module, module_cfg["main"])


def print_tensor_info(tensor, flag="Tensor"):
    floor_tensor = lambda float_tensor: int(float(float_tensor) * 1000) / 1000
    print(flag)
    print(
        f"\tmax: {floor_tensor(torch.max(tensor))}, min: {float(torch.min(tensor))}, mean: {floor_tensor(torch.mean(tensor))}, std: {floor_tensor(torch.std(tensor))}")


##################### DCASE Util functions #####################

# TODO: move this to a separate file (utils_seld.py)

def distance_between_cartesian_coordinates(x1, y1, z1, x2, y2, z2):
    """
    Angular distance between two cartesian coordinates
    MORE: https://en.wikipedia.org/wiki/Great-circle_distance
    Check 'From chord length' section

    :return: angular distance in degrees
    """
    # Normalize the Cartesian vectors
    N1 = np.sqrt(x1**2 + y1**2 + z1**2 + 1e-10)
    N2 = np.sqrt(x2**2 + y2**2 + z2**2 + 1e-10)
    x1, y1, z1, x2, y2, z2 = x1/N1, y1/N1, z1/N1, x2/N2, y2/N2, z2/N2

    #Compute the distance
    dist = x1*x2 + y1*y2 + z1*z2
    dist = np.clip(dist, -1, 1)
    dist = np.arccos(dist) * 180 / np.pi
    return dist

def determine_similar_location(sed_pred0, sed_pred1, doa_pred0, doa_pred1, class_cnt, thresh_unify, nb_classes):
    if (sed_pred0 == 1) and (sed_pred1 == 1):
        if distance_between_cartesian_coordinates(doa_pred0[class_cnt], doa_pred0[class_cnt+1*nb_classes], doa_pred0[class_cnt+2*nb_classes],
                                                  doa_pred1[class_cnt], doa_pred1[class_cnt+1*nb_classes], doa_pred1[class_cnt+2*nb_classes]) < thresh_unify:
            return 1
        else:
            return 0
    else:
        return 0

def reshape_3Dto2D(A):
    return A.reshape(A.shape[0] * A.shape[1], A.shape[2])

def write_output_format_file(_output_format_file, _output_format_dict):
    """
    Writes DCASE output format csv file, given output format dictionary

    :param _output_format_file:
    :param _output_format_dict:
    :return:
    """
    _fid = open(_output_format_file, 'w')
    # _fid.write('{},{},{},{}\n'.format('frame number with 20ms hop (int)', 'class index (int)', 'azimuth angle (int)', 'elevation angle (int)'))
    for _frame_ind in _output_format_dict.keys():
        for _value in _output_format_dict[_frame_ind]:
            # Write Cartesian format output. Since baseline does not estimate track count and distance we use fixed values.
            _fid.write('{},{},{},{},{},{},{}\n'.format(int(_frame_ind), int(_value[0]), 0, float(_value[1]), float(_value[2]), float(_value[3]), 0))
    _fid.close()

def get_multi_accdoa_labels(accdoa_in, nb_classes):
    """
    Args:
        accdoa_in:  [batch_size, frames, num_track*num_axis*num_class=3*3*12]
        nb_classes: scalar
    Return:
        sedX:       [batch_size, frames, num_class=12]
        doaX:       [batch_size, frames, num_axis*num_class=3*12]
    """
    x0, y0, z0 = accdoa_in[:, :, :1*nb_classes], accdoa_in[:, :, 1*nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:3*nb_classes]
    sed0 = np.sqrt(x0**2 + y0**2 + z0**2) > 0.5
    doa0 = accdoa_in[:, :, :3*nb_classes]

    x1, y1, z1 = accdoa_in[:, :, 3*nb_classes:4*nb_classes], accdoa_in[:, :, 4*nb_classes:5*nb_classes], accdoa_in[:, :, 5*nb_classes:6*nb_classes]
    sed1 = np.sqrt(x1**2 + y1**2 + z1**2) > 0.5
    doa1 = accdoa_in[:, :, 3*nb_classes: 6*nb_classes]

    x2, y2, z2 = accdoa_in[:, :, 6*nb_classes:7*nb_classes], accdoa_in[:, :, 7*nb_classes:8*nb_classes], accdoa_in[:, :, 8*nb_classes:]
    sed2 = np.sqrt(x2**2 + y2**2 + z2**2) > 0.5
    doa2 = accdoa_in[:, :, 6*nb_classes:]

    return sed0, doa0, sed1, doa1, sed2, doa2


def load_output_format_file(_output_format_file):
    """
    Loads DCASE output format csv file and returns it in dictionary format

    :param _output_format_file: DCASE output format CSV
    :return: _output_dict: dictionary
    """
    _output_dict = {}
    _fid = open(_output_format_file, 'r')
    # next(_fid)
    for _line in _fid:
        _words = _line.strip().split(',')
        _frame_ind = int(_words[0])
        if _frame_ind not in _output_dict:
            _output_dict[_frame_ind] = []
        if len(_words) == 5:  # frame, class idx, source_id, polar coordinates(2) # no distance data, for example in synthetic data fold 1 and 2
            _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4])])
        if len(_words) == 6: # frame, class idx, source_id, polar coordinates(2), distance 
            _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4])])
        elif len(_words) == 7: # frame, class idx, source_id, cartesian coordinates(3), distance
            _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4]), float(_words[5])])
    _fid.close()
    return _output_dict


def convert_output_format_polar_to_cartesian(in_dict):
    out_dict = {}
    for frame_cnt in in_dict.keys():
        if frame_cnt not in out_dict:
            out_dict[frame_cnt] = []
            for tmp_val in in_dict[frame_cnt]:

                ele_rad = tmp_val[3]*np.pi/180.
                azi_rad = tmp_val[2]*np.pi/180

                tmp_label = np.cos(ele_rad)
                x = np.cos(azi_rad) * tmp_label
                y = np.sin(azi_rad) * tmp_label
                z = np.sin(ele_rad)
                out_dict[frame_cnt].append([tmp_val[0], tmp_val[1], x, y, z])
    return out_dict


def get_adpit_labels_for_file(_desc_file, total_label_frames, _nb_unique_classes=13):
        """
        Reads description file and returns classification based SED labels and regression based DOA labels
        for multi-ACCDOA with Auxiliary Duplicating Permutation Invariant Training (ADPIT)

        :param _desc_file: metadata description file
        :return: label_mat: of dimension [nb_frames, 6, 4(=act+XYZ), max_classes]
        """

        se_label = np.zeros((total_label_frames, 6, _nb_unique_classes))  # [nb_frames, 6, max_classes]
        x_label = np.zeros((total_label_frames, 6, _nb_unique_classes))
        y_label = np.zeros((total_label_frames, 6, _nb_unique_classes))
        z_label = np.zeros((total_label_frames, 6, _nb_unique_classes))

        for frame_ind, active_event_list in _desc_file.items():
            if frame_ind < total_label_frames:
                active_event_list.sort(key=lambda x: x[0])  # sort for ov from the same class
                active_event_list_per_class = []
                for i, active_event in enumerate(active_event_list):
                    active_event_list_per_class.append(active_event)
                    if i == len(active_event_list) - 1:  # if the last
                        if len(active_event_list_per_class) == 1:  # if no ov from the same class
                            # a0----
                            active_event_a0 = active_event_list_per_class[0]
                            se_label[frame_ind, 0, active_event_a0[0]] = 1
                            x_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[2]
                            y_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[3]
                            z_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[4]
                        elif len(active_event_list_per_class) == 2:  # if ov with 2 sources from the same class
                            # --b0--
                            active_event_b0 = active_event_list_per_class[0]
                            se_label[frame_ind, 1, active_event_b0[0]] = 1
                            x_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[2]
                            y_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[3]
                            z_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[4]
                            # --b1--
                            active_event_b1 = active_event_list_per_class[1]
                            se_label[frame_ind, 2, active_event_b1[0]] = 1
                            x_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[2]
                            y_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[3]
                            z_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[4]
                        else:  # if ov with more than 2 sources from the same class
                            # ----c0
                            active_event_c0 = active_event_list_per_class[0]
                            se_label[frame_ind, 3, active_event_c0[0]] = 1
                            x_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[2]
                            y_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[3]
                            z_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[4]
                            # ----c1
                            active_event_c1 = active_event_list_per_class[1]
                            se_label[frame_ind, 4, active_event_c1[0]] = 1
                            x_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[2]
                            y_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[3]
                            z_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[4]
                            # ----c2
                            active_event_c2 = active_event_list_per_class[2]
                            se_label[frame_ind, 5, active_event_c2[0]] = 1
                            x_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[2]
                            y_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[3]
                            z_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[4]

                    elif active_event[0] != active_event_list[i + 1][0]:  # if the next is not the same class
                        if len(active_event_list_per_class) == 1:  # if no ov from the same class
                            # a0----
                            active_event_a0 = active_event_list_per_class[0]
                            se_label[frame_ind, 0, active_event_a0[0]] = 1
                            x_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[2]
                            y_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[3]
                            z_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[4]
                        elif len(active_event_list_per_class) == 2:  # if ov with 2 sources from the same class
                            # --b0--
                            active_event_b0 = active_event_list_per_class[0]
                            se_label[frame_ind, 1, active_event_b0[0]] = 1
                            x_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[2]
                            y_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[3]
                            z_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[4]
                            # --b1--
                            active_event_b1 = active_event_list_per_class[1]
                            se_label[frame_ind, 2, active_event_b1[0]] = 1
                            x_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[2]
                            y_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[3]
                            z_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[4]
                        else:  # if ov with more than 2 sources from the same class
                            # ----c0
                            active_event_c0 = active_event_list_per_class[0]
                            se_label[frame_ind, 3, active_event_c0[0]] = 1
                            x_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[2]
                            y_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[3]
                            z_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[4]
                            # ----c1
                            active_event_c1 = active_event_list_per_class[1]
                            se_label[frame_ind, 4, active_event_c1[0]] = 1
                            x_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[2]
                            y_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[3]
                            z_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[4]
                            # ----c2
                            active_event_c2 = active_event_list_per_class[2]
                            se_label[frame_ind, 5, active_event_c2[0]] = 1
                            x_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[2]
                            y_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[3]
                            z_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[4]
                        active_event_list_per_class = []

        label_mat = np.stack((se_label, x_label, y_label, z_label), axis=2)  # [nb_frames, 6, 4(=act+XYZ), max_classes]
        return label_mat

