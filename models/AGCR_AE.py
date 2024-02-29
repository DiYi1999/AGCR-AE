import torch
from torch import nn
from models.GCRC import GCRC


class GCR_AE(nn.Module):
    def __init__(self, args):
        super(GCR_AE, self).__init__()
        self.args = args

        self.e = nn.Parameter(torch.randn(args.number_of_nodes, args.GCN_embedding_dimensions)
                              , requires_grad=True).to(args.device)
        print('e has been init')
        self.GRU_GCN_ENCODER = GCRC(args=args,
                                    number_of_nodes=args.number_of_nodes,
                                    in_channels=args.lag,
                                    out_channels=args.gru1_outlen,
                                    K=args.K,
                                    embedding_dimensions=args.GCN_embedding_dimensions)
        self.GRU_GCN_DECODER = GCRC(args=args,
                                    number_of_nodes=args.number_of_nodes,
                                    in_channels=args.gru1_outlen,
                                    out_channels=args.gru2_outlen,
                                    K=args.K,
                                    embedding_dimensions=args.GCN_embedding_dimensions)
        self.GRU_GCN_DECODER_skip = GCRC(args=args,
                                         number_of_nodes=args.number_of_nodes,
                                         in_channels=args.lag,
                                         out_channels=args.gru2_outlen,
                                         K=args.K,
                                         embedding_dimensions=args.GCN_embedding_dimensions)
        self.linear = torch.nn.Linear(args.gru2_outlen, args.lag)

    def forward(self, x, h1, h2):
        if self.args.skip:
            h1 = self.GRU_GCN_ENCODER(x, self.e, h1)
            h21 = self.GRU_GCN_DECODER(h1, self.e, h2)
            h22 = self.GRU_GCN_DECODER_skip(x, self.e, h2)
            h2 = (1 - self.args.skip_rate) * h21 + self.args.skip_rate * h22
            x_new = self.linear(h2.permute(0, 2, 1)).permute(0, 2, 1)
            if self.args.AE_if_activation:
                x_new = self.args.AE_activation(x_new)
        else:
            h1 = self.GRU_GCN_ENCODER(x, self.e, h1)
            h2 = self.GRU_GCN_DECODER(h1, self.e, h2)
            x_new = self.linear(h2.permute(0, 2, 1)).permute(0, 2, 1)
            if self.args.AE_if_activation:
                x_new = self.args.AE_activation(x_new)
        h1_new = h1.data
        h2_new = h2.data
        return x_new, h1_new, h2_new


class AGCR_AE_critic(nn.Module):
    def __init__(self, in_feats, h_feats, dropout):
        super(AGCR_AE_critic, self).__init__()

        self.linear1 = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, in_feats)
        self.linear3 = nn.Linear(in_feats, 1)
        self.relu = nn.ReLU()
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.0

    def forward(self, features):
        h = self.relu(self.linear1(features))
        if self.dropout:
            h = self.dropout(h)
        h = self.relu(self.linear2(h))
        if self.dropout:
            h = self.dropout(h)
        h = self.linear3(h)
        return h