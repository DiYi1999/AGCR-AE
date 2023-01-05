from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from utils.data import *
import numpy as np

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

def get_best_performance_data(total_err_scores, gt_labels, topk=1, focus_on='F1'):

    total_features = total_err_scores.shape[0]

    topk_indices = np.argpartition(total_err_scores, range(total_features - topk - 1, total_features), axis=0)[-topk:]

    total_topk_err_scores = []
    topk_err_score_map = []

    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)

    final_topk_fmeas, thresolds = eval_rate_scores(total_topk_err_scores, gt_labels, 400, return_thresold=True, focus_on=focus_on)

    th_i = final_topk_fmeas.index(max(final_topk_fmeas))
    thresold = thresolds[th_i]

    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > thresold] = 1

    acc = accuracy_score(gt_labels, pred_labels)
    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)

    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)

    return max(final_topk_fmeas), acc, pre, rec, auc_score, thresold


def get_best_performance_alllabel(total_err_scores, all_test_labels, topk=1, focus_on='F1'):

    assert total_err_scores.shape == all_test_labels.shape, '形状不一致，请转置或切短前后'

    f1_score_list = []
    acc_list = []
    pre_list = []
    rec_list = []
    auc_score_list = []
    thresold_list = []

    for i in range(total_err_scores.shape[0]):
        final_topk_fmeas, thresolds = eval_rate_scores(total_err_scores[i], all_test_labels[i], 400, return_thresold=True, focus_on=focus_on)

        th_i = final_topk_fmeas.index(max(final_topk_fmeas))
        thresold = thresolds[th_i]

        pred_labels = np.zeros(len(total_err_scores[i]))
        pred_labels[total_err_scores[i] > thresold] = 1
        print(f'预测异常点个数：{np.sum(pred_labels)}')
        print(f'实际异常点个数：{np.sum(all_test_labels[i])}')

        f1_score = max(final_topk_fmeas)
        acc = accuracy_score(all_test_labels[i], pred_labels)
        pre = precision_score(all_test_labels[i], pred_labels, zero_division=1)
        rec = recall_score(all_test_labels[i], pred_labels, zero_division=1)
        print(f'f1,acc,pre,rec:{f1_score,acc,pre,rec}')
        try:
            auc_score = roc_auc_score(all_test_labels[i], total_err_scores[i])
            auc_score_list.append(auc_score)
        except:
            print(0)
        f1_score_list.append(f1_score)
        acc_list.append(acc)
        pre_list.append(pre)
        rec_list.append(rec)
        thresold_list.append(thresold)

    f1_score = np.mean(f1_score_list)
    acc = np.mean(acc_list)
    pre = np.mean(pre_list)
    rec = np.mean(rec_list)
    auc_score = np.mean(auc_score_list)

    return f1_score, acc, pre, rec, auc_score, thresold_list


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe