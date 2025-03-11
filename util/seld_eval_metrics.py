# Implements the localization and detection metrics proposed in [1] with extensions to support multi-instance of the same class from [2].
#
# [1] Joint Measurement of Localization and Detection of Sound Events
# Annamaria Mesaros, Sharath Adavanne, Archontis Politis, Toni Heittola, Tuomas Virtanen
# WASPAA 2019
#
# [2] Overview and Evaluation of Sound Event Localization and Detection in DCASE 2019
# Politis, Archontis, Annamaria Mesaros, Sharath Adavanne, Toni Heittola, and Tuomas Virtanen.
# IEEE/ACM Transactions on Audio, Speech, and Language Processing (2020).
#
# This script has MIT license
#

# Metrics borrowed from https://github.com/sharathadavanne/seld-dcase2023

import os
import numpy as np

eps = np.finfo(np.float).eps
from scipy.optimize import linear_sum_assignment
from IPython import embed

from scipy import stats

from util.utils import load_output_format_file, convert_output_format_cartesian_to_polar, segment_labels


class SELDMetrics(object):
    def __init__(self, doa_threshold=20, nb_classes=11, average='macro'):
        '''
            This class implements both the class-sensitive localization and location-sensitive detection metrics.
            Additionally, based on the user input, the corresponding averaging is performed within the segment.

        :param nb_classes: Number of sound classes. In the paper, nb_classes = 11
        :param doa_thresh: DOA threshold for location sensitive detection.
        '''
        self._nb_classes = nb_classes

        # Variables for Location-senstive detection performance
        self._TP = np.zeros(self._nb_classes)
        self._FP = np.zeros(self._nb_classes)
        self._FP_spatial = np.zeros(self._nb_classes)
        self._FN = np.zeros(self._nb_classes)

        self._Nref = np.zeros(self._nb_classes)

        self._spatial_T = doa_threshold

        self._S = 0
        self._D = 0
        self._I = 0

        # Variables for Class-sensitive localization performance
        self._total_DE = np.zeros(self._nb_classes)

        self._DE_TP = np.zeros(self._nb_classes)
        self._DE_FP = np.zeros(self._nb_classes)
        self._DE_FN = np.zeros(self._nb_classes)

        self._average = average

    def early_stopping_metric(self, _er, _f, _le, _lr):
        """
        Compute early stopping metric from sed and doa errors.

        :param sed_error: [error rate (0 to 1 range), f score (0 to 1 range)]
        :param doa_error: [doa error (in degrees), frame recall (0 to 1 range)]
        :return: early stopping metric result
        """
        seld_metric = np.mean([
            _er,
            1 - _f,
            _le / 180,
            1 - _lr
        ], 0)
        return seld_metric

    def compute_seld_scores(self):
        '''
        Collect the final SELD scores

        :return: returns both location-sensitive detection scores and class-sensitive localization scores
        '''
        ER = (self._S + self._D + self._I) / (self._Nref.sum() + eps)
        classwise_results = []
        if self._average == 'micro':
            # Location-sensitive detection performance
            F = self._TP.sum() / (eps + self._TP.sum() + self._FP_spatial.sum() + 0.5 * (self._FP.sum() + self._FN.sum()))

            # Class-sensitive localization performance
            LE = self._total_DE.sum() / float(self._DE_TP.sum() + eps) if self._DE_TP.sum() else 180
            LR = self._DE_TP.sum() / (eps + self._DE_TP.sum() + self._DE_FN.sum())

            SELD_scr = self.early_stopping_metric(ER, F, LE, LR)

        elif self._average == 'macro':
            # Location-sensitive detection performance
            F = self._TP / (eps + self._TP + self._FP_spatial + 0.5 * (self._FP + self._FN))

            # Class-sensitive localization performance
            LE = self._total_DE / (self._DE_TP + eps)
            LE[self._DE_TP==0] = 180.0
            LR = self._DE_TP / (eps + self._DE_TP + self._DE_FN)

            SELD_scr = self.early_stopping_metric(np.repeat(ER, self._nb_classes), F, LE, LR)
            classwise_results = np.array([np.repeat(ER, self._nb_classes), F, LE, LR, SELD_scr])
            F, LE, LR, SELD_scr = F.mean(), LE.mean(), LR.mean(), SELD_scr.mean()
        return ER, F, LE, LR, SELD_scr, classwise_results

    def update_seld_scores(self, pred, gt):
        '''
        Implements the spatial error averaging according to equation 5 in the paper [1] (see papers in the title of the code).
        Adds the multitrack extensions proposed in paper [2]

        The input pred/gt can either both be Cartesian or Degrees

        :param pred: dictionary containing class-wise prediction results for each N-seconds segment block
        :param gt: dictionary containing class-wise groundtruth for each N-seconds segment block
        '''
        for block_cnt in range(len(gt.keys())):
            loc_FN, loc_FP = 0, 0
            for class_cnt in range(self._nb_classes):
                # Counting the number of referece tracks for each class in the segment
                nb_gt_doas = max([len(val) for val in gt[block_cnt][class_cnt][0][1]]) if class_cnt in gt[block_cnt] else None
                nb_pred_doas = max([len(val) for val in pred[block_cnt][class_cnt][0][1]]) if class_cnt in pred[block_cnt] else None
                if nb_gt_doas is not None:
                    self._Nref[class_cnt] += nb_gt_doas
                if class_cnt in gt[block_cnt] and class_cnt in pred[block_cnt]:
                    # True positives or False positive case

                    # NOTE: For multiple tracks per class, associate the predicted DOAs to corresponding reference
                    # DOA-tracks using hungarian algorithm and then compute the average spatial distance between
                    # the associated reference-predicted tracks.

                    # Reference and predicted track matching
                    matched_track_dist = {}
                    matched_track_cnt = {}
                    gt_ind_list = gt[block_cnt][class_cnt][0][0]
                    pred_ind_list = pred[block_cnt][class_cnt][0][0]
                    for gt_ind, gt_val in enumerate(gt_ind_list):
                        if gt_val in pred_ind_list:
                            gt_arr = np.array(gt[block_cnt][class_cnt][0][1][gt_ind])
                            gt_ids = np.arange(len(gt_arr[:, -1])) #TODO if the reference has track IDS use here - gt_arr[:, -1]
                            gt_doas = gt_arr[:, 1:]

                            pred_ind = pred_ind_list.index(gt_val)
                            pred_arr = np.array(pred[block_cnt][class_cnt][0][1][pred_ind])
                            pred_doas = pred_arr[:, 1:]

                            if gt_doas.shape[-1] == 2: # convert DOAs to radians, if the input is in degrees
                                gt_doas = gt_doas * np.pi / 180.
                                pred_doas = pred_doas * np.pi / 180.

                            dist_list, row_inds, col_inds = least_distance_between_gt_pred(gt_doas, pred_doas)

                            # Collect the frame-wise distance between matched ref-pred DOA pairs
                            for dist_cnt, dist_val in enumerate(dist_list):
                                matched_gt_track = gt_ids[row_inds[dist_cnt]]
                                if matched_gt_track not in matched_track_dist:
                                    matched_track_dist[matched_gt_track], matched_track_cnt[matched_gt_track] = [], []
                                matched_track_dist[matched_gt_track].append(dist_val)
                                matched_track_cnt[matched_gt_track].append(pred_ind)

                    # Update evaluation metrics based on the distance between ref-pred tracks
                    if len(matched_track_dist) == 0:
                        # if no tracks are found. This occurs when the predicted DOAs are not aligned frame-wise to the reference DOAs
                        loc_FN += nb_pred_doas
                        self._FN[class_cnt] += nb_pred_doas
                        self._DE_FN[class_cnt] += nb_pred_doas
                    else:
                        # for the associated ref-pred tracks compute the metrics
                        for track_id in matched_track_dist:
                            total_spatial_dist = sum(matched_track_dist[track_id])
                            total_framewise_matching_doa = len(matched_track_cnt[track_id])
                            avg_spatial_dist = total_spatial_dist / total_framewise_matching_doa

                            # Class-sensitive localization performance
                            self._total_DE[class_cnt] += avg_spatial_dist
                            self._DE_TP[class_cnt] += 1

                            # Location-sensitive detection performance
                            if avg_spatial_dist <= self._spatial_T:
                                self._TP[class_cnt] += 1
                            else:
                                loc_FP += 1
                                self._FP_spatial[class_cnt] += 1
                        # in the multi-instance of same class scenario, if the number of predicted tracks are greater
                        # than reference tracks count as FP, if it less than reference count as FN
                        if nb_pred_doas > nb_gt_doas:
                            # False positive
                            loc_FP += (nb_pred_doas-nb_gt_doas)
                            self._FP[class_cnt] += (nb_pred_doas-nb_gt_doas)
                            self._DE_FP[class_cnt] += (nb_pred_doas-nb_gt_doas)
                        elif nb_pred_doas < nb_gt_doas:
                            # False negative
                            loc_FN += (nb_gt_doas-nb_pred_doas)
                            self._FN[class_cnt] += (nb_gt_doas-nb_pred_doas)
                            self._DE_FN[class_cnt] += (nb_gt_doas-nb_pred_doas)
                elif class_cnt in gt[block_cnt] and class_cnt not in pred[block_cnt]:
                    # False negative
                    loc_FN += nb_gt_doas
                    self._FN[class_cnt] += nb_gt_doas
                    self._DE_FN[class_cnt] += nb_gt_doas
                elif class_cnt not in gt[block_cnt] and class_cnt in pred[block_cnt]:
                    # False positive
                    loc_FP += nb_pred_doas
                    self._FP[class_cnt] += nb_pred_doas
                    self._DE_FP[class_cnt] += nb_pred_doas

            self._S += np.minimum(loc_FP, loc_FN)
            self._D += np.maximum(0, loc_FN - loc_FP)
            self._I += np.maximum(0, loc_FP - loc_FN)
        return


def distance_between_spherical_coordinates_rad(az1, ele1, az2, ele2):
    """
    Angular distance between two spherical coordinates
    MORE: https://en.wikipedia.org/wiki/Great-circle_distance

    :return: angular distance in degrees
    """
    dist = np.sin(ele1) * np.sin(ele2) + np.cos(ele1) * np.cos(ele2) * np.cos(np.abs(az1 - az2))
    # Making sure the dist values are in -1 to 1 range, else np.arccos kills the job
    dist = np.clip(dist, -1, 1)
    dist = np.arccos(dist) * 180 / np.pi
    return dist


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


def least_distance_between_gt_pred(gt_list, pred_list):
    """
        Shortest distance between two sets of DOA coordinates. Given a set of groundtruth coordinates,
        and its respective predicted coordinates, we calculate the distance between each of the
        coordinate pairs resulting in a matrix of distances, where one axis represents the number of groundtruth
        coordinates and the other the predicted coordinates. The number of estimated peaks need not be the same as in
        groundtruth, thus the distance matrix is not always a square matrix. We use the hungarian algorithm to find the
        least cost in this distance matrix.
        :param gt_list_xyz: list of ground-truth Cartesian or Polar coordinates in Radians
        :param pred_list_xyz: list of predicted Carteisan or Polar coordinates in Radians
        :return: cost - distance
        :return: less - number of DOA's missed
        :return: extra - number of DOA's over-estimated
    """

    gt_len, pred_len = gt_list.shape[0], pred_list.shape[0]
    ind_pairs = np.array([[x, y] for y in range(pred_len) for x in range(gt_len)])
    cost_mat = np.zeros((gt_len, pred_len))

    if gt_len and pred_len:
        if len(gt_list[0]) == 3: #Cartesian
            x1, y1, z1, x2, y2, z2 = gt_list[ind_pairs[:, 0], 0], gt_list[ind_pairs[:, 0], 1], gt_list[ind_pairs[:, 0], 2], pred_list[ind_pairs[:, 1], 0], pred_list[ind_pairs[:, 1], 1], pred_list[ind_pairs[:, 1], 2]
            cost_mat[ind_pairs[:, 0], ind_pairs[:, 1]] = distance_between_cartesian_coordinates(x1, y1, z1, x2, y2, z2)
        else:
            az1, ele1, az2, ele2 = gt_list[ind_pairs[:, 0], 0], gt_list[ind_pairs[:, 0], 1], pred_list[ind_pairs[:, 1], 0], pred_list[ind_pairs[:, 1], 1]
            cost_mat[ind_pairs[:, 0], ind_pairs[:, 1]] = distance_between_spherical_coordinates_rad(az1, ele1, az2, ele2)

    row_ind, col_ind = linear_sum_assignment(cost_mat)
    cost = cost_mat[row_ind, col_ind]
    return cost, row_ind, col_ind



def jackknife_estimation(global_value, partial_estimates, significance_level=0.05):
    """
    Compute jackknife statistics from a global value and partial estimates.
    Original function by Nicolas Turpault

    :param global_value: Value calculated using all (N) examples
    :param partial_estimates: Partial estimates using N-1 examples at a time
    :param significance_level: Significance value used for t-test

    :return:
    estimate: estimated value using partial estimates
    bias: Bias computed between global value and the partial estimates
    std_err: Standard deviation of partial estimates
    conf_interval: Confidence interval obtained after t-test
    """

    mean_jack_stat = np.mean(partial_estimates)
    n = len(partial_estimates)
    bias = (n - 1) * (mean_jack_stat - global_value)

    std_err = np.sqrt(
        (n - 1) * np.mean((partial_estimates - mean_jack_stat) * (partial_estimates - mean_jack_stat), axis=0)
    )

    # bias-corrected "jackknifed estimate"
    estimate = global_value - bias

    # jackknife confidence interval
    if not (0 < significance_level < 1):
        raise ValueError("confidence level must be in (0, 1).")

    t_value = stats.t.ppf(1 - significance_level / 2, n - 1)

    # t-test
    conf_interval = estimate + t_value * np.array((-std_err, std_err))

    return estimate, bias, std_err, conf_interval

class ComputeSELDResults(object):
    def __init__(
            self, params, ref_files_folder=None, use_polar_format=True
    ):
        self._use_polar_format = use_polar_format
        self._desc_dir = ref_files_folder if ref_files_folder is not None else params["train_dataset"]["args"]["labels_path"]
        self._doa_thresh = params['seld_metrics']['lad_doa_thresh']

        self.num_classes = params['model']['args']['feat_config']['unique_classes']
        
        # collect reference files
        self._ref_labels = {}
        for split in os.listdir(self._desc_dir):      
            for ref_file in os.listdir(os.path.join(self._desc_dir, split)):
                # Load reference description file
                gt_dict = load_output_format_file(os.path.join(self._desc_dir, split, ref_file))
                if not self._use_polar_format:
                    gt_dict = convert_output_format_polar_to_cartesian(gt_dict)
                nb_ref_frames = max(list(gt_dict.keys()))
                self._ref_labels[ref_file] = [segment_labels(gt_dict, nb_ref_frames), nb_ref_frames]

        self._nb_ref_files = len(self._ref_labels)
        self._average = params['seld_metrics']['average']

    @staticmethod
    def get_nb_files(file_list, tag='all'):
        '''
        Given the file_list, this function returns a subset of files corresponding to the tag.

        Tags supported
        'all' -
        'ir'

        :param file_list: complete list of predicted files
        :param tag: Supports two tags 'all', 'ir'
        :return: Subset of files according to chosen tag
        '''
        _group_ind = {'room': 10}
        _cnt_dict = {}
        for _filename in file_list:

            if tag == 'all':
                _ind = 0
            else:
                _ind = int(_filename[_group_ind[tag]])

            if _ind not in _cnt_dict:
                _cnt_dict[_ind] = []
            _cnt_dict[_ind].append(_filename)

        return _cnt_dict

    def get_SELD_Results(self, pred_files_path, is_jackknife=False):
        # collect predicted files info
        pred_files = os.listdir(pred_files_path)
        pred_labels_dict = {}
        eval = SELDMetrics(nb_classes=self.num_classes, doa_threshold=self._doa_thresh, average=self._average)
        for pred_cnt, pred_file in enumerate(pred_files):
            # Load predicted output format file
            pred_dict = load_output_format_file(os.path.join(pred_files_path, pred_file))
            if self._use_polar_format:
                pred_dict = convert_output_format_cartesian_to_polar(pred_dict)
            pred_labels = segment_labels(pred_dict, self._ref_labels[pred_file][1])
            # Calculated scores
            eval.update_seld_scores(pred_labels, self._ref_labels[pred_file][0])
            if is_jackknife:
                pred_labels_dict[pred_file] = pred_labels
        # Overall SED and DOA scores
        ER, F, LE, LR, seld_scr, classwise_results = eval.compute_seld_scores()

        if is_jackknife:
            global_values = [ER, F, LE, LR, seld_scr]
            if len(classwise_results):
                global_values.extend(classwise_results.reshape(-1).tolist())
            partial_estimates = []
            # Calculate partial estimates by leave-one-out method
            for leave_file in pred_files:
                leave_one_out_list = pred_files[:]
                leave_one_out_list.remove(leave_file)
                eval = SELDMetrics(nb_classes=self.num_classes, doa_threshold=self._doa_thresh, average=self._average)
                for pred_cnt, pred_file in enumerate(leave_one_out_list):
                    # Calculated scores
                    eval.update_seld_scores(pred_labels_dict[pred_file], self._ref_labels[pred_file][0])
                ER, F, LE, LR, seld_scr, classwise_results = eval.compute_seld_scores()
                leave_one_out_est = [ER, F, LE, LR, seld_scr]
                if len(classwise_results):
                    leave_one_out_est.extend(classwise_results.reshape(-1).tolist())

                # Overall SED and DOA scores
                partial_estimates.append(leave_one_out_est)
            partial_estimates = np.array(partial_estimates)
                    
            estimate, bias, std_err, conf_interval = [-1]*len(global_values), [-1]*len(global_values), [-1]*len(global_values), [-1]*len(global_values)
            for i in range(len(global_values)):
                estimate[i], bias[i], std_err[i], conf_interval[i] = jackknife_estimation(
                           global_value=global_values[i],
                           partial_estimates=partial_estimates[:, i],
                           significance_level=0.05
                           )
            return [ER, conf_interval[0]], [F, conf_interval[1]], [LE, conf_interval[2]], [LR, conf_interval[3]], [seld_scr, conf_interval[4]], [classwise_results, np.array(conf_interval)[5:].reshape(5,13,2) if len(classwise_results) else []]
      
        else:      
            return ER, F, LE, LR, seld_scr, classwise_results

    def get_consolidated_SELD_results(self, pred_files_path, score_type_list=['all', 'room']):
        '''
            Get all categories of results.
            ;score_type_list: Supported
                'all' - all the predicted files
                'room' - for individual rooms

        '''

        # collect predicted files info
        pred_files = os.listdir(pred_files_path)
        nb_pred_files = len(pred_files)

        # Calculate scores for different splits, overlapping sound events, and impulse responses (reverberant scenes)

        print('Number of predicted files: {}\nNumber of reference files: {}'.format(nb_pred_files, self._nb_ref_files))
        print('\nCalculating {} scores for {}'.format(score_type_list, os.path.basename(pred_output_format_files)))

        for score_type in score_type_list:
            print('\n\n---------------------------------------------------------------------------------------------------')
            print('------------------------------------  {}   ---------------------------------------------'.format('Total score' if score_type=='all' else 'score per {}'.format(score_type)))
            print('---------------------------------------------------------------------------------------------------')

            split_cnt_dict = self.get_nb_files(pred_files, tag=score_type) # collect files corresponding to score_type
            # Calculate scores across files for a given score_type
            for split_key in np.sort(list(split_cnt_dict)):
                # Load evaluation metric class
                eval = SELDMetrics(nb_classes=self.num_classes, doa_threshold=self._doa_thresh, average=self._average)
                for pred_cnt, pred_file in enumerate(split_cnt_dict[split_key]):
                    # Load predicted output format file
                    pred_dict = load_output_format_file(os.path.join(pred_output_format_files, pred_file))
                    if self._use_polar_format:
                        pred_dict = convert_output_format_cartesian_to_polar(pred_dict)
                    pred_labels = segment_labels(pred_dict, self._ref_labels[pred_file][1])

                    # Calculated scores
                    eval.update_seld_scores(pred_labels, self._ref_labels[pred_file][0])

                # Overall SED and DOA scores
                ER, F, LE, LR, seld_scr, classwise_results = eval.compute_seld_scores()

                print('\nAverage score for {} {} data using {} coordinates'.format(score_type, 'fold' if score_type=='all' else split_key, 'Polar' if self._use_polar_format else 'Cartesian' ))
                print('SELD score (early stopping metric): {:0.2f}'.format(seld_scr))
                print('SED metrics: Error rate: {:0.2f}, F-score:{:0.1f}'.format(ER, 100*F))
                print('DOA metrics: Localization error: {:0.1f}, Localization Recall: {:0.1f}'.format(LE, 100*LR))

def reshape_3Dto2D(A):
    return A.reshape(A.shape[0] * A.shape[1], A.shape[2])


if __name__ == "__main__":
    pred_output_format_files = 'results/3_11553814_dev_split0_multiaccdoa_foa_20220429142557_test' # Path of the DCASEoutput format files
    params = parameters.get_params()
    # Compute just the DCASE final results 
    score_obj = ComputeSELDResults(params)
    use_jackknife=False
    ER, F, LE, LR, seld_scr, classwise_test_scr = score_obj.get_SELD_Results(pred_output_format_files,is_jackknife=use_jackknife )
   
    print('SELD score (early stopping metric): {:0.2f} {}'.format(seld_scr[0] if use_jackknife else seld_scr, '[{:0.2f}, {:0.2f}]'.format(seld_scr[1][0], seld_scr[1][1]) if use_jackknife else ''))
    print('SED metrics: Error rate: {:0.2f} {}, F-score: {:0.1f} {}'.format(ER[0]  if use_jackknife else ER, '[{:0.2f},  {:0.2f}]'.format(ER[1][0], ER[1][1]) if use_jackknife else '', 100*F[0]  if use_jackknife else 100*F, '[{:0.2f}, {:0.2f}]'.format(100*F[1][0], 100*F[1][1]) if use_jackknife else ''))
    print('DOA metrics: Localization error: {:0.1f} {}, Localization Recall: {:0.1f} {}'.format(LE[0] if use_jackknife else LE, '[{:0.2f}, {:0.2f}]'.format(LE[1][0], LE[1][1]) if use_jackknife else '', 100*LR[0]  if use_jackknife else 100*LR,'[{:0.2f}, {:0.2f}]'.format(100*LR[1][0], 100*LR[1][1]) if use_jackknife else ''))
    if params['average']=='macro':
        print('Classwise results on unseen test data')
        print('Class\tER\tF\tLE\tLR\tSELD_score')
        for cls_cnt in range(params['unique_classes']):
            print('{}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}'.format(
cls_cnt, 
classwise_test_scr[0][0][cls_cnt] if use_jackknife else classwise_test_scr[0][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][0][cls_cnt][0], classwise_test_scr[1][0][cls_cnt][1]) if use_jackknife else '', 
classwise_test_scr[0][1][cls_cnt] if use_jackknife else classwise_test_scr[1][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][1][cls_cnt][0], classwise_test_scr[1][1][cls_cnt][1]) if use_jackknife else '', 
classwise_test_scr[0][2][cls_cnt] if use_jackknife else classwise_test_scr[2][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][2][cls_cnt][0], classwise_test_scr[1][2][cls_cnt][1]) if use_jackknife else '', 
classwise_test_scr[0][3][cls_cnt] if use_jackknife else classwise_test_scr[3][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][3][cls_cnt][0], classwise_test_scr[1][3][cls_cnt][1]) if use_jackknife else '', 
classwise_test_scr[0][4][cls_cnt] if use_jackknife else classwise_test_scr[4][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][4][cls_cnt][0], classwise_test_scr[1][4][cls_cnt][1]) if use_jackknife else ''))

