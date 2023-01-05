import os
import pickle
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import warnings

warnings.filterwarnings('ignore')

"""
Thanks for all these excellent works, where our data comes from:

  title={Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding},
  author={Hundman, Kyle and Constantinou, Valentino and Laporte, Christopher and Colwell, Ian and Soderstrom, Tom},
  journal={arXiv preprint arXiv:1802.04431},
  year={2018}
}
https://github.com/khundman/telemanom

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

@article{tuli2022tranad,
  title={{TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data}},
  author={Tuli, Shreshth and Casale, Giuliano and Jennings, Nicholas R},
  journal={Proceedings of VLDB},
  volume={15},
  number={6},
  pages={1201-1214},
  year={2022}
}
https://github.com/imperial-qore/TranAD
"""

class NASA_Anomaly(Dataset):
    def __init__(self, root_path, flag='train', lag=None,
                 features='M', data_path='SMAP', data_name='MSL',
                 missing_rate=0.2, missvalue=np.nan, target=0, scale=False):
        # size [seq_len, label_len pred_len]
        # info
        self.lag = lag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.missing_rate = missing_rate
        self.missvalue =missvalue

        self.root_path = root_path
        self.data_path = data_path
        self.data_name = data_name
        self.__read_data__()

    def get_data_dim(self, dataset):
        if dataset == 'SMAP':
            return 25
        elif dataset == 'MSL':
            return 27
        elif str(dataset).startswith('machine'):
            return 38
        else:
            raise ValueError('unknown dataset ' + str(dataset))

    def __read_data__(self):
        """
        get data from files

        return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
        """

        x_dim = self.get_data_dim(self.data_name)
        if self.flag in ['val', 'train']:
            data = pd.read_csv(os.path.join(self.root_path, self.data_path, '{}_train.csv'.format(self.data_name)), sep=',', index_col=False)
            data = data.values.reshape((-1, x_dim))
        elif self.flag == 'test':
            try:
                data = pd.read_csv(os.path.join(self.root_path, self.data_path, '{}_test.csv'.format(self.data_name)), sep=',', index_col=False)
                data = data.values.reshape((-1, x_dim))
            except (KeyError, FileNotFoundError):
                data = None
            try:
                label = pd.read_csv(os.path.join(self.root_path, self.data_path, '{}_test_label.csv'.format(self.data_name)), sep=',', index_col=False)
                label = label.values.reshape((-1))
            except (KeyError, FileNotFoundError):
                label = None
            try:
                all_label = pd.read_csv(os.path.join(self.root_path, self.data_path, '{}_test_all_label.csv'.format(self.data_name)), sep=',', index_col=False)
                all_label = all_label.values
            except (KeyError, FileNotFoundError):
                all_label = None
            assert len(data) == len(label), "length of test data should be the same as label"
            self.label = label
            self.all_label = all_label
        if self.scale:
            data = self.preprocess(data)

        # df_stamp是时间标签信息
        df_stamp = pd.DataFrame(columns=['date'])
        date = pd.date_range(start='1/1/2015', periods=len(data), freq='4s')
        df_stamp['date'] = date
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
        # df_stamp['minute'] = df_stamp.minute.map(lambda x:x//10)
        df_stamp['second'] = df_stamp.date.apply(lambda row: row.second, 1)
        data_stamp = df_stamp.drop(['date'], 1).values
        self.data_stamp = data_stamp

        # make missing
        full_data = data
        data_1d = full_data.reshape(-1)
        data_len = len(data_1d)
        mask_1d = np.ones(data_len)
        corrupt_ids = random.sample(range(data_len), int(self.missing_rate * data_len))
        for i in corrupt_ids:
            data_1d[i] = self.missvalue
            mask_1d[i] = 0
        missed_data = data_1d.reshape(data.shape)
        miss_mask = mask_1d.reshape(data.shape)

        if self.flag == 'train':
            if self.features == 'M':
                self.missed_data = missed_data
                self.full_data = full_data
                self.mask = miss_mask
            elif self.features == 'S':
                df_missed_data =missed_data[:, [self.target]]
                df_full_data = data[:, [self.target]]
                df_miss_mask = miss_mask[:, [self.target]]
                self.missed_data = df_missed_data
                self.full_data = df_full_data
                self.mask = df_miss_mask

        else:
            border1s = [0, 0, 0]
            border2s = [None, len(data) // 4, len(data)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            if self.features == 'M':
                self.missed_data = missed_data[border1:border2]
                self.full_data = full_data[border1:border2]
                self.mask = miss_mask[border1:border2]
                # self.label = label[border1:border2]
                self.data_stamp = self.data_stamp[border1:border2]
            elif self.features == 'S':
                df_missed_data =missed_data[:, [self.target]]
                df_full_data = data[:, [self.target]]
                df_miss_mask = miss_mask[:, [self.target]]
                self.missed_data = df_missed_data[border1:border2]
                self.full_data = df_full_data[border1:border2]
                self.mask = df_miss_mask[border1:border2]
                # self.label = label[border1:border2]
                self.data_stamp = self.data_stamp[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.lag

        missed_batch = self.missed_data[s_begin:s_end]
        mask_batch = self.mask[s_begin:s_end]
        full_batch = self.full_data[s_begin:s_end]
        datetime_batch = self.data_stamp[s_begin:s_end]

        if self.flag in ['train', 'val']:
            return missed_batch, mask_batch, datetime_batch, full_batch
        else:
            anomaly_label = self.label
            anomaly_all_label = self.all_label
            return missed_batch, mask_batch, datetime_batch, full_batch, anomaly_label, anomaly_all_label

    def __len__(self):
        return len(self.missed_data) - self.lag + 1

    def preprocess(self, df):
        """
        returns normalized and standardized data.
        """

        df = np.asarray(df, dtype=np.float32)

        if len(df.shape) == 1:
            raise ValueError('Data must be a 2-D array')

        if np.any(np.isnan(df).sum() != 0):
            print('Data contains null values. Will be replaced with 0')
            df = np.nan_to_num(df)

        # normalize data
        df = MinMaxScaler().fit_transform(df)
        print('Data normalized')

        return df

