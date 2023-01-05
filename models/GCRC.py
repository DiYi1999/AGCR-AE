import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot, zeros


class GCN_e(nn.Module):
    r"""
    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Filter size :math:`K`.
        embedding_dimensions (int): Number of node embedding dimensions.
    """

    def __init__(
        self, args, in_channels: int, out_channels: int, K: int, embedding_dimensions: int
    ):

        super(GCN_e, self).__init__()

        self.args = args
        self.K = K
        if args.EwEb:
            self.weights_pool = torch.nn.Parameter(
                torch.Tensor(embedding_dimensions, K, in_channels, out_channels)
            )
            self.bias_pool = torch.nn.Parameter(
                torch.Tensor(embedding_dimensions, out_channels)
            )
        else:
            self.weights_pool = torch.nn.Parameter(
                torch.Tensor(args.number_of_nodes, K, in_channels, out_channels)
            )
            self.bias_pool = torch.nn.Parameter(
                torch.Tensor(args.number_of_nodes, out_channels)
            )
        glorot(self.weights_pool)
        zeros(self.bias_pool)

    def forward(self, X: torch.FloatTensor, E: torch.FloatTensor) -> torch.FloatTensor:
        r"""Making a forward pass.
        Arg types:
            * **X** (PyTorch Float Tensor) - Node features.
            * **E** (PyTorch Float Tensor) - Node embeddings.
        Return types:
            * **X_G** (PyTorch Float Tensor) - Hidden state matrix for all nodes.
        """

        number_of_nodes = E.shape[0]
        supports = F.softmax(F.relu(torch.mm(E, E.transpose(0, 1))), dim=1)
        support_set = [torch.eye(number_of_nodes).to(supports.device), supports]
        for _ in range(2, self.K):
            support = torch.matmul(2 * supports, support_set[-1]) - support_set[-2]
            support_set.append(support)
        supports = torch.stack(support_set, dim=0)
        if self.args.EwEb:
            W = torch.einsum("nd,dkio->nkio", E, self.weights_pool)
            bias = torch.matmul(E, self.bias_pool)
        else:
            W = self.weights_pool
            bias = self.bias_pool
        X_G = torch.einsum("knm,bmc->bknc", supports, X)
        X_G = X_G.permute(0, 2, 1, 3)
        X_G = torch.einsum("bnki,nkio->bno", X_G, W) + bias
        return X_G


class GCRC(nn.Module):
    r"""
    Args:
        number_of_nodes (int): Number of vertices.
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Filter size :math:`K`.
        embedding_dimensions (int): Number of node embedding dimensions.
    """

    def __init__(
        self,
        args,
        number_of_nodes: int,
        in_channels: int,
        out_channels: int,
        K: int,
        embedding_dimensions: int,
    ):

        super(GCRC, self).__init__()

        self.args = args
        self.number_of_nodes = number_of_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.embedding_dimensions = embedding_dimensions
        self._setup_layers()

    def _setup_layers(self):
        self._gate = GCN_e(
            args=self.args,
            in_channels=self.in_channels + self.out_channels,
            out_channels=2 * self.out_channels,
            K=self.K,
            embedding_dimensions=self.embedding_dimensions,
        )

        self._update = GCN_e(
            args=self.args,
            in_channels=self.in_channels + self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            embedding_dimensions=self.embedding_dimensions,
        )


    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels, X.shape[1]).to(X.device)
        return H

    def forward(
        self, X: torch.FloatTensor, E: torch.FloatTensor, H: torch.FloatTensor = None
    ) -> torch.FloatTensor:
        r"""Making a forward pass.
        Arg types:
            * **X** (PyTorch Float Tensor) - Node feature matrix.
            * **E** (PyTorch Float Tensor) - Node embedding matrix.
            * **H** (PyTorch Float Tensor) - Node hidden state matrix. Default is None.
        Return types:
            * **H** (PyTorch Float Tensor) - Hidden state matrix for all nodes.
        """
        X_T = X.permute(0, 2, 1)
        H = self._set_hidden_state(X_T, H)
        H_T = H.permute(0, 2, 1)
        X_H = torch.cat((X_T, H_T), dim=-1)
        Z_R = torch.sigmoid(self._gate(X_H, E))
        Z, R = torch.split(Z_R, self.out_channels, dim=-1)
        C = torch.cat((X_T, Z * H_T), dim=-1)
        HC = torch.tanh(self._update(C, E))
        H_T = R * H_T + (1 - R) * HC
        H_TT = H_T.permute(0, 2, 1)
        return H_TT




