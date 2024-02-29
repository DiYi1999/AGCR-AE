import argparse
import torch
import pandas as pd


def preIDW(x_batch):
    x_data = x_batch.numpy()
    for i in range(x_data.shape[0]):
        i_batch = pd.DataFrame(x_data[i])
        i_batch = i_batch.fillna(method="ffill", axis=0)
        i_batch = i_batch.fillna(method="backfill", axis=0)
        x_data[i] = i_batch.values
    return torch.FloatTensor(x_data)















