import os
from pathlib import Path

import pandas as pd
import torch
import random
import time
import matplotlib.font_manager
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# matplotlib.font_manager._rebuild()
from torch import nn
import numpy as np
from argparse import ArgumentParser
# from pytorch_lightning import Trainer
from models.AGCR_AE import GCR_AE
from models.AGCR_AE import AGCR_AE_critic
from utils.preprocess import preIDW
from utils.evaluate import metric, MAE
from utils.scores import get_final_err_scores, get_full_err_scores
from utils.evaluate import get_best_performance_data, get_best_performance_alllabel
from utils.gradient import hard_gradient_penalty
from data.lightingdata import LigDataloader
# import pytorch_lightning as pl
# import pytorch_lightning.callbacks as plc
# from pytorch_lightning.loggers import TensorBoardLogger
# from utils import load_model_path_by_args


def main(args):
    ### 实验复现随机种子定义
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)

    ### 缺失数据迭代器实例化
    myLigDataloader = LigDataloader(args)
    Train_Dataloader = myLigDataloader.train_dataloader()
    Test_Dataloader = myLigDataloader.test_dataloader()

    ### 模型实例化及优化器定义
    # 生成器/自编码器
    myGCRAN_Autoencoder = GCR_AE(args).to(args.device)
    # 判别器
    myGCRAN_critic = AGCR_AE_critic(in_feats=args.number_of_nodes, h_feats=args.critic_h_feats, dropout=args.dropout).to(args.device)
    # 优化器
    if not args.fine_tune:
        optim_auto = torch.optim.Adam(
            myGCRAN_Autoencoder.parameters(),
            lr=args.auto_lr,
            betas=args.betas,
            weight_decay=args.weight_decay,
        )
        if args.adversarial:
            optim_crit = torch.optim.Adam(
                myGCRAN_critic.parameters(),
                lr=args.crit_lr,
                betas=args.betas,
                weight_decay=args.weight_decay,
            )

    ### 开始训练
    myGCRAN_Autoencoder.train()
    myGCRAN_critic.train()
    dur = []
    a_losses = []
    c_losses = []
    d_losses = []
    w_losses = []
    t0 = time.time()
    b_loss = 999
    patience_cnt = 0
    for epoch in range(args.epochs):
        h1 = None
        h2 = None
        for i, batch in enumerate(Train_Dataloader):
            if args.adversarial:
                missed_batch, mask_batch, datetime_batch, full_batch = batch
                promiss_batch = preIDW(missed_batch)
                promiss_batch_tensor = torch.FloatTensor(promiss_batch).to(args.device)
                mask_batch_tensor = torch.BoolTensor(mask_batch.to(torch.bool)).to(args.device)
                # full_batch_tensor = torch.FloatTensor(full_batch).to(args.device)
                # 为了防止信息泄露，上面这个其实用不到，写出来也没进行使用
                # 训练判别器
                faked_batch_tensor, h1, h2 = myGCRAN_Autoencoder(promiss_batch_tensor, h1, h2)
                x_real = torch.mul(promiss_batch_tensor, mask_batch_tensor)
                x_fake = torch.mul(faked_batch_tensor, mask_batch_tensor)
                c_real = myGCRAN_critic(x_real)
                c_fake = myGCRAN_critic(x_fake)
                gp = hard_gradient_penalty(myGCRAN_critic, x_real, x_fake, args.device)
                loss_w = c_fake.mean() - c_real.mean()
                loss_C = loss_w + gp
                torch.autograd.set_detect_anomaly(True)
                loss_C.backward()
                optim_crit.step()
                if i % args.n_critic == 0:
                    faked_batch_tensor, h1, h2 = myGCRAN_Autoencoder(promiss_batch_tensor, h1, h2)
                    x_real = torch.mul(promiss_batch_tensor, mask_batch_tensor)
                    x_fake = torch.mul(faked_batch_tensor, mask_batch_tensor)
                    loss_A = args.lossfunction(promiss_batch_tensor[mask_batch_tensor],
                                               faked_batch_tensor[mask_batch_tensor])
                    if args.glob_attr:
                        loss_A_with_glob = loss_A + args.global_rate * args.lossfunction(
                            torch.FloatTensor(np.mean(x_real.cpu().detach().numpy(), axis=0)).to(args.device),
                            torch.FloatTensor(np.mean(x_fake.cpu().detach().numpy(), axis=0)).to(args.device)
                        )
                    c_fake = myGCRAN_critic(x_fake)
                    if args.glob_attr:
                        loss_D = loss_A_with_glob - c_fake.mean()
                        optim_auto.zero_grad()
                        loss_D.backward()
                        optim_auto.step()
                    else:
                        loss_D = loss_A - c_fake.mean()
                        optim_auto.zero_grad()
                        loss_D.backward()
                        optim_auto.step()
            else:
                missed_batch, mask_batch, datetime_batch, full_batch = batch
                promiss_batch = preIDW(missed_batch)
                promiss_batch_tensor = torch.FloatTensor(promiss_batch).to(args.device)
                mask_batch_tensor = torch.BoolTensor(mask_batch.to(torch.bool)).to(args.device)
                # full_batch_tensor = torch.FloatTensor(full_batch).to(args.device)
                # 为了防止信息泄露，上面这个其实用不到，写出来也没进行使用用
                reco_batch_tensor, h1, h2 = myGCRAN_Autoencoder(promiss_batch_tensor, h1, h2)
                x_real = torch.mul(promiss_batch_tensor, mask_batch_tensor)
                x_reco = torch.mul(reco_batch_tensor, mask_batch_tensor)
                loss_A = args.lossfunction(promiss_batch_tensor[mask_batch_tensor],
                                           reco_batch_tensor[mask_batch_tensor])
                if args.glob_attr:
                    loss_A_with_glob = loss_A + args.global_rate * args.lossfunction(
                            torch.FloatTensor(np.mean(x_real.cpu().detach().numpy(), axis=0)).to(self.device),
                            torch.FloatTensor(np.mean(x_reco.cpu().detach().numpy(), axis=0)).to(self.device)
                        )
                    optim_auto.zero_grad()
                    loss_A_with_glob.backward()
                    optim_auto.step()
                else:
                    optim_auto.zero_grad()
                    loss_A.backward()
                    optim_auto.step()
        if epoch % 1 == 0:
            dur.append(time.time() - t0)

            a_losses.append(loss_A.detach().item())
            if args.adversarial:
                c_losses.append(loss_C.detach().item())
                d_losses.append(loss_D.detach().item())
                w_losses.append(loss_w.detach().item())
                if args.lossprint:
                    print(
                        "EPOCH: %05d," % epoch,
                        "A_LOSS: %f," % loss_A.detach().item(),
                        "C_LOSS: %f," % loss_C.detach().item(),
                        "D_LOSS: %f," % loss_D.detach().item(),
                        "W_LOSS: %f " % loss_w.detach().item(),
                        "= (%f" % c_fake.mean().detach().item(),
                        "-(%f))" % c_real.mean().detach().item(),
                        "GP: %f" % gp.detach().item(),
                    )
            else:
                if args.lossprint:
                    print(
                        "EPOCH: %05d," % epoch,
                        "A_LOSS: %f" % loss_A.detach().item(),
                    )
        if epoch % 1 == 0:
            patience_cnt += 1
            if loss_A < b_loss:
                patience_cnt = 0
                b_loss = loss_A
                if args.adversarial:
                    torch.save(
                        {
                            "auto_state_dict": myGCRAN_Autoencoder.state_dict(),
                            "optim_auto_state_dict": optim_auto.state_dict(),
                            "crit_state_dict": myGCRAN_critic.state_dict(),
                            "optim_crit_state_dict": optim_crit.state_dict(),
                        },
                        "GCRAN.pth",
                    )
                else:
                    torch.save(
                        {
                            "auto_state_dict": myGCRAN_Autoencoder.state_dict(),
                            "optim_auto_state_dict": optim_auto.state_dict(),
                        },
                        "GCRAN.pth",
                    )
        if patience_cnt > args.patience:
            break

    ### 测试
    checkpoint = torch.load("GCRAN.pth")
    myGCRAN_Autoencoder.load_state_dict(checkpoint["auto_state_dict"])
    optim_auto.load_state_dict(checkpoint["optim_auto_state_dict"])
    myGCRAN_Autoencoder.eval()
    with torch.no_grad():
        for i, batch in enumerate(Test_Dataloader):
            missed_batch, mask_batch, datetime_batch, full_batch, anomaly_label, anomaly_all_label = batch
            promiss_batch = preIDW(missed_batch)
            promiss_batch_tensor = torch.FloatTensor(promiss_batch).to(args.device)
            mask_batch_tensor = torch.BoolTensor(mask_batch.to(torch.bool)).to(args.device)
            full_batch_tensor = torch.FloatTensor(full_batch).to(args.device)
            reco_batch_tensor, h1, h2 = myGCRAN_Autoencoder(promiss_batch_tensor, h1, h2)
            x_real = torch.mul(promiss_batch_tensor, mask_batch_tensor)
            x_reco = torch.mul(reco_batch_tensor, mask_batch_tensor)
            if i == 0:
                real_test_data = x_real[0,:,:]
                reco_test_data = x_reco[0,:,:]
                real_full_test_data = full_batch_tensor[0,:,:]
                reco_full_test_data = reco_batch_tensor[0,:,:]
                for i in range(1,args.batch_size):
                    real_test_data = torch.cat(
                        (real_test_data[:real_test_data.shape[0] - x_real.shape[1] + 1, :], x_real[i, :, :]), dim=0)
                    reco_test_data = torch.cat(
                        (reco_test_data[:reco_test_data.shape[0] - x_reco.shape[1] + 1, :], x_reco[i, :, :]), dim=0)
                    real_full_test_data = torch.cat(
                        (real_full_test_data[:real_full_test_data.shape[0] - full_batch_tensor.shape[1] + 1, :], full_batch_tensor[i, :, :]), dim=0)
                    reco_full_test_data = torch.cat(
                        (reco_full_test_data[:reco_full_test_data.shape[0] - reco_batch_tensor.shape[1] + 1, :], reco_batch_tensor[i, :, :]), dim=0)
            else:
                for i in range(args.batch_size):
                    real_test_data = torch.cat(
                        (real_test_data[:real_test_data.shape[0] - x_real.shape[1] + 1, :], x_real[i, :, :]), dim=0)
                    reco_test_data = torch.cat(
                        (reco_test_data[:reco_test_data.shape[0] - x_reco.shape[1] + 1, :], x_reco[i, :, :]), dim=0)
                    real_full_test_data = torch.cat(
                        (real_full_test_data[:real_full_test_data.shape[0] - full_batch_tensor.shape[1] + 1, :], full_batch_tensor[i, :, :]), dim=0)
                    reco_full_test_data = torch.cat(
                        (reco_full_test_data[:reco_full_test_data.shape[0] - reco_batch_tensor.shape[1] + 1, :], reco_batch_tensor[i, :, :]), dim=0)

    ### 计算异常得分
    reco_test_data = reco_test_data.tolist()
    real_test_data = real_test_data.tolist()
    reco_full_test_data = reco_full_test_data.tolist()
    real_full_test_data = real_full_test_data.tolist()
    test_labels = anomaly_label.tolist()[0][:len(reco_test_data)]
    test_all_labels = anomaly_all_label[0].T.cpu().numpy()[:,:len(reco_test_data)]
    final_test_scores = get_final_err_scores(reco_test_data, real_test_data)
    all_test_scores = get_full_err_scores(reco_test_data, real_test_data)

    ### 计算各评价指标
    if args.how_precision:
        top1_bestF1_result = get_best_performance_alllabel(all_test_scores, test_all_labels, topk=1, focus_on=args.focus_on)
    else:
        top1_bestF1_result = get_best_performance_data(all_test_scores, test_labels, topk=1, focus_on=args.focus_on)
    if args.focus_on=='F1':
        print(f'F1 score: {top1_bestF1_result[0]}')
    else:
        print(f'F1 score: {top1_bestF1_result[0]-top1_bestF1_result[2]}')
    print(f'accuracy: {top1_bestF1_result[1]}')
    print(f'precision: {top1_bestF1_result[2]}')
    print(f'recall: {top1_bestF1_result[3]}')
    print(f'AUC(ROC下面积): {top1_bestF1_result[4]}')
    print(f'阶梯遍历计算出最大F1时采用的阈值是: {top1_bestF1_result[5]}')
    mae, mse, rmse, mape, mspe = metric(np.array(reco_test_data), np.array(real_test_data))
    normal_mse = MAE(np.array(reco_test_data)[:290,:], np.array(real_test_data)[:290,:]) + MAE(np.array(reco_test_data)[400:550, :], np.array(real_test_data)[400:550, :])
    print('mse:{}, mae:{}'.format(mse, mae))

    ### 将相关参数及调试结果写入文件
    current_path = os.path.abspath(__file__)
    a = current_path.split('/')[:-1]
    last_path = '/'.join(a)
    os.chdir(last_path)
    result_save_path = f'./results/结果保存.csv'
    dirname = os.path.dirname(result_save_path)
    Path(dirname).mkdir(parents=True, exist_ok=True)
    # 没有这个文件就创建，接下来开始写入
    df = pd.DataFrame({'missing_rate': [args.missing_rate],
                       'MSE': [mse],
                       'normal_mse': [normal_mse],
                       'precision': [top1_bestF1_result[2]],
                       'recall': [top1_bestF1_result[3]],
                       'F1_score':[[top1_bestF1_result[0]],[top1_bestF1_result[0]-top1_bestF1_result[2]]][not args.focus_on=='F1'],
                       'AUC': [top1_bestF1_result[4]],
                       'accuracy': [top1_bestF1_result[1]],
                       'lag': [args.lag],
                       'batch_size': [args.batch_size],
                       'gru1_outlen': [args.gru1_outlen],
                       'gru2_outlen': [args.gru2_outlen],
                       'critic_h_feats': [args.critic_h_feats],
                       'GCN_embedding_dimensions': [args.GCN_embedding_dimensions],
                       'K': [args.K],
                       'dropout': [args.dropout],
                       'AE_if_activation': args.AE_if_activation,
                       'n_critic': [args.n_critic],
                       'auto_lr': [args.auto_lr],
                       'crit_lr': [args.crit_lr],
                       'betas': str(args.betas),
                       'focus_on':str(args.focus_on),
                       'EwEb': args.EwEb,
                       'skip': args.skip,
                       'skip_rate': [args.skip_rate],
                       'glob_attr': args.glob_attr,
                       'global_attribute': str(args.global_attribute),
                       'global_rate': [args.global_rate],
                       'adversarial': args.adversarial,
                       'patience': [args.patience],
                       'epochs': [args.epochs]
                       })
    df.to_csv(path_or_buf=result_save_path, index=False, mode='a', header=False)

if __name__ == '__main__':exuemosh
    parser = ArgumentParser()

    ### data, model
    parser.add_argument('--data_name', type=str, default='MSL', help='data')
    parser.add_argument('--root_path', type=str, default='/media/xjtu/新加卷/DY/GitHub/AGCR-AE/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='dataset/msl/', help='location of the data file')
    parser.add_argument('--lag', type=int, help='lag/slide_win/x_channels', default=200)###
    parser.add_argument('--missing_rate', type=float, default=0.2, help='missing_rate')###
    parser.add_argument('--missvalue', default=np.nan, help='np.nan, 0, etc.')
    parser.add_argument('--scale', type=bool, default=True, help='normalization or not')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--features', type=str, default='M', help='features [S, M]')
    parser.add_argument('--target', type=str, default='OT', help='target feature')

    ### Trainin
    parser.add_argument('--device', type=str, default='cuda:0', help='use gpu or cpu')
    parser.add_argument('--epochs', type=int, default=500, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='input data batch size')#
    parser.add_argument('--random_seed', help='random seed', type=int, default=0)
    parser.add_argument('--patience', type=int, default=10, help='stop patience')
    parser.add_argument('--lossprint', type=bool, default=True, help='print or not')
    parser.add_argument('--how_precision', type=bool, default=False, help='Flase from GDN')
    parser.add_argument('--focus_on', type=str, default='F1+Pre', help='F1 or F1+Pre from GDN')

    ### optimizer
    parser.add_argument('--auto_lr', default=0.001, type=float, help='Generator 0.001')
    parser.add_argument('--crit_lr', default=0.001, type=float, help='discriminator 0.001')
    parser.add_argument('--weight_decay', default=0, type=float, help='weight_decay')
    parser.add_argument('--lossfunction', default=nn.MSELoss(), help='nn.MSELoss()')
    parser.add_argument('--fine_tune', default=False, type=bool, help='if True, the previous state is used for optimizer instantiation')
    parser.add_argument('--betas', default=(0, 0.9), help='Adam (0.5,0.9), (0.9,0.99), (0.9,0.999)')

    ### AGCR-AE
    parser.add_argument('--EwEb', help='use EwEb strategy or not', type=bool, default=True)
    parser.add_argument('--skip', help='use skip connection strategy or not', type=bool, default=False)
    parser.add_argument('--skip_rate', help='skip connection rate', type=float, default=0.5)
    parser.add_argument('--glob_attr', help='use global attribute strategy or not', type=bool, default=False)
    parser.add_argument('--global_attribute', help='which global attribute', type=str, default='mean')
    parser.add_argument('--global_rate', help='global attribute strategy rate', type=float, default=1)
    parser.add_argument('--number_of_nodes', help='number_of_nodes', type=int, default=27)
    parser.add_argument('--gru1_outlen', help='gru1_outlen', type=int, default=500)###
    parser.add_argument('--gru2_outlen', help='gru2_outlen', type=int, default=400)###
    parser.add_argument('--K', help='Graph convolution K', type=int, default=2)
    parser.add_argument('--GCN_embedding_dimensions', help='embedding E', type=int, default=25)
    parser.add_argument('--AE_if_activation', type=bool, default=False, help="Whether the last layer of the "
                                                                             "autoencoder uses an activation "
                                                                             "function")### ###
    parser.add_argument('--AE_activation', default=torch.sigmoid, help="which activation function")

    ### adversarial strategy
    parser.add_argument('--adversarial', type=bool, default=True, help='use adversarial strategy or not')
    parser.add_argument('--n_critic', type=int, default=5, help='5')

    # discriminator
    parser.add_argument('--dropout', type=float, default=0.25, help='dropout')
    parser.add_argument('--critic_h_feats', type=int, default=500, help='critic_h_feats')###

    # parser.set_defaults(max_epochs=100)
    args = parser.parse_args()


    main(args=args)
