import numpy as np
from scipy.stats import rankdata, iqr, trim_mean

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
"""

def get_final_err_scores(pred_data, true_data):
    all_scores = get_full_err_scores(pred_data, true_data)
    final_scores = np.max(all_scores, axis=1).tolist()

    return final_scores


def get_full_err_scores(pred_data, true_data):
    pred_data = np.array(pred_data)
    true_data = np.array(true_data)

    all_scores = None
    feature_num = pred_data.shape[-1]

    for i in range(feature_num):
        scores = get_err_scores(pred_data[:, i], true_data[:, i])

        if all_scores is None:
            all_scores = scores
        else:
            all_scores = np.vstack((
                all_scores,
                scores
            ))

    return all_scores


def get_err_scores(pred_data_one, true_data_one):

    n_err_mid, n_err_iqr = get_err_median_and_iqr(pred_data_one, true_data_one)

    test_delta = np.abs(np.subtract(
        np.array(pred_data_one).astype(np.float64),
        np.array(true_data_one).astype(np.float64)
    ))
    epsilon = 1e-2

    err_scores = test_delta

    smoothed_err_scores = np.zeros(err_scores.shape)
    before_num = 3
    for i in range(before_num, len(err_scores)):
        smoothed_err_scores[i] = np.mean(err_scores[i - before_num:i + 1])

    return smoothed_err_scores


def get_err_median_and_iqr(predicted, groundtruth):

    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))
    err_median = np.median(np_arr)
    err_iqr = iqr(np_arr)

    return err_median, err_iqr
