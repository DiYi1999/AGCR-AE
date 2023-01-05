# util functions about data

from scipy.stats import rankdata, iqr, trim_mean
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, mean_squared_error
import numpy as np
from numpy import percentile

"""
Thanks to this excellent work, we adapted its experiment metrics calculation:
@inproceedings{deng2021graph,
  title={Graph neural network-based anomaly detection in multivariate time series},
  author={Deng, Ailin and Hooi, Bryan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={5},
  pages={4027--4035},
  year={2021}
}
https://github.com/d-ailin/GDN
"""

def get_attack_interval(attack):
    heads = []
    tails = []
    for i in range(len(attack)):
        if attack[i] == 1:
            if attack[i - 1] == 0:
                heads.append(i)

            if i < len(attack) - 1 and attack[i + 1] == 0:
                tails.append(i)
            elif i == len(attack) - 1:
                tails.append(i)
    res = []
    for i in range(len(heads)):
        res.append((heads[i], tails[i]))
    return res


# calculate F1 scores
def eval_rate_scores(scores, true_scores, th_steps, return_thresold=False, focus_on='F1'):
    padding_list = [0] * (len(true_scores) - len(scores))

    if len(padding_list) > 0:
        scores = padding_list + scores

    scores_sorted = rankdata(scores, method='ordinal')
    th_steps = th_steps
    th_vals = np.array(range(th_steps+1)) * 1.0 / th_steps
    fmeas = [None] * (th_steps+1)
    thresholds = [None] * (th_steps+1)
    for i in range(th_steps+1):
        cur_pred = scores_sorted > th_vals[i] * len(scores)

        if focus_on=='F1':
            fmeas[i] = f1_score(true_scores, cur_pred, zero_division=1)
        else:
            fmeas[i] = f1_score(true_scores, cur_pred) + precision_score(true_scores, cur_pred)

        if int(th_vals[i] * len(scores))==0:
            score_index = scores_sorted.tolist().index(1)
            thresholds[i] = scores[score_index]-1
        else:
            score_index = scores_sorted.tolist().index(int(th_vals[i] * len(scores)))
            thresholds[i] = scores[score_index]

    if return_thresold:
        return fmeas, thresholds
    return fmeas

# calculate F1 scores
def eval_number_scores(scores, true_scores, th_steps, return_thresold=False, focus_on='F1'):
    scores_sorted = rankdata(scores, method='ordinal')
    th_steps = th_steps
    abnormal_num = int(np.sum(true_scores))
    if abnormal_num-th_steps < 0:
        start_num = 0
    else:
        start_num = abnormal_num-th_steps
    if abnormal_num+th_steps > len(true_scores):
        end_num = len(true_scores)
    else:
        end_num = abnormal_num+th_steps
    fmeas = [None] * (end_num - start_num + 1)
    thresholds = [None] * (end_num - start_num + 1)
    for pred_abnormal_num in range(start_num, end_num+1):
        if len(true_scores)-pred_abnormal_num==0:
            score_index = scores_sorted.tolist().index(1)
            thresholds[pred_abnormal_num - start_num] = scores[score_index]-1
        else:
            score_index = scores_sorted.tolist().index(len(true_scores)-pred_abnormal_num)
            thresholds[pred_abnormal_num-start_num] = scores[score_index]
        cur_pred = scores > thresholds[pred_abnormal_num - start_num]

        if focus_on=='F1':
            fmeas[pred_abnormal_num-start_num] = f1_score(true_scores, cur_pred, zero_division=1)
        else:
            fmeas[pred_abnormal_num-start_num] = f1_score(true_scores, cur_pred, zero_division=1)\
                                                + precision_score(true_scores, cur_pred, zero_division=1)

    if return_thresold:
        return fmeas, thresholds
    return fmeas

def eval_mseloss(predicted, ground_truth):
    ground_truth_list = np.array(ground_truth)
    predicted_list = np.array(predicted)
    loss = mean_squared_error(predicted_list, ground_truth_list)

    return loss


def get_err_median_and_iqr(predicted, groundtruth):
    """
    计算给定俩数列的 偏差 的中位数和四分位数

    :param predicted: 预测数列
    :param groundtruth: 真实数列
    :return: abs(预测数列 - 真实数列)的中位数就四分位数
    """

    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_median = np.median(np_arr)
    err_iqr = iqr(np_arr)

    return err_median, err_iqr


def get_err_median_and_quantile(predicted, groundtruth, percentage):
    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_median = np.median(np_arr)
    err_delta = percentile(np_arr, int(percentage * 100)) - percentile(np_arr, int((1 - percentage) * 100))

    return err_median, err_delta


def get_err_mean_and_quantile(predicted, groundtruth, percentage):
    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_median = trim_mean(np_arr, percentage)
    err_delta = percentile(np_arr, int(percentage * 100)) - percentile(np_arr, int((1 - percentage) * 100))

    return err_median, err_delta


def get_err_mean_and_std(predicted, groundtruth):
    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_mean = np.mean(np_arr)
    err_std = np.std(np_arr)

    return err_mean, err_std


def get_f1_score(scores, gt, contamination):
    padding_list = [0] * (len(gt) - len(scores))

    threshold = percentile(scores, 100 * (1 - contamination))

    if len(padding_list) > 0:
        scores = padding_list + scores

    pred_labels = (scores > threshold).astype('int').ravel()

    return f1_score(gt, pred_labels)