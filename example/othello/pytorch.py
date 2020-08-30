"""
A pytorch implementation of the neural network for the Othello game.
Based on https://github.com/suragnair/alpha-zero-general/tree/master/othello/pytorch
"""


import os

import numpy as np
from alpha_zero_general import DotDict
from alpha_zero_general import NeuralNet
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

args = DotDict(
    {
        "lr": 0.001,
        "dropout": 0.3,
        "epochs": 1,
        "batch_size": 512,
        "cuda": torch.cuda.is_available(),
        "num_channels": 512,
    }
)


class AverageMeter:
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f"{self.avg:.2e}"

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class PyTorchNetWrapper(NeuralNet):
    def __init__(self, game):
        self.board_x, self.board_y = game.get_board_size()
        self.model = self.get_model(
            game.get_board_size(), game.get_action_size(), args,
        )
        if args.cuda:
            self.model.cuda()

    @staticmethod
    def get_model(board_size, action_size, args):
        """
        Return compiled tensorflow.keras.models.Model.
        """
        raise NotImplementedError

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.model.parameters())

        for epoch in range(args.epochs):
            self.model.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / args.batch_size)

            t = tqdm(
                range(batch_count),
                desc=f"Training Net - Epoch {epoch + 1}",
                disable=True,
            )
            for _ in t:
                sample_ids = np.random.randint(
                    len(examples), size=args.batch_size
                )
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if args.cuda:
                    boards = boards.contiguous().cuda()
                    target_pis = target_pis.contiguous().cuda()
                    target_vs = target_vs.contiguous().cuda()

                # compute output
                out_pi, out_v = self.model(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

        return pi_losses.avg, v_losses.avg

    def predict(self, board):
        """
        board: np array with board
        """
        # preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        if args.cuda:
            board = board.contiguous().cuda()
        board = board.view(1, self.board_x, self.board_y)
        self.model.eval()
        with torch.no_grad():
            pi, v = self.model(board)

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(
        self, folder="checkpoint", filename="checkpoint.pth.tar"
    ):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(
                "Checkpoint Directory does not exist! Making directory {}".format(
                    folder
                )
            )
            os.mkdir(folder)
        torch.save({"state_dict": self.model.state_dict()}, filepath)

    def load_checkpoint(
        self, folder="checkpoint", filename="checkpoint.pth.tar"
    ):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ValueError("No model in path {}".format(filepath))
        map_location = None if args.cuda else "cpu"
        checkpoint = torch.load(filepath, map_location=map_location)
        self.model.load_state_dict(checkpoint["state_dict"])

    def get_weights(self):
        return {
            key: value.cpu() for key, value in self.model.state_dict().items()
        }

    def set_weights(self, weights):
        self.model.load_state_dict(weights)

    def request_gpu(self):
        return self.model.args.cuda


class OthelloNNet(PyTorchNetWrapper):
    @staticmethod
    def get_model(board_size, action_size, args):
        return OthelloNNetModel(board_size, action_size, args)


class OthelloNNetModel(nn.Module):
    def __init__(self, board_size, action_size, args):
        # game params
        self.board_x, self.board_y = board_size
        self.action_size = action_size
        self.args = args

        super().__init__()
        self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            args.num_channels, args.num_channels, 3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            args.num_channels, args.num_channels, 3, stride=1
        )
        self.conv4 = nn.Conv2d(
            args.num_channels, args.num_channels, 3, stride=1
        )

        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)

        self.fc1 = nn.Linear(
            args.num_channels * (self.board_x - 4) * (self.board_y - 4), 1024
        )
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        # s: batch_size x board_x x board_y

        # batch_size x 1 x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y)
        # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))
        # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn3(self.conv3(s)))
        # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = F.relu(self.bn4(self.conv4(s)))
        s = s.view(
            -1,
            self.args.num_channels * (self.board_x - 4) * (self.board_y - 4),
        )

        # batch_size x 1024
        s = F.dropout(
            F.relu(self.fc_bn1(self.fc1(s))),
            p=self.args.dropout,
            training=self.training,
        )
        # batch_size x 512
        s = F.dropout(
            F.relu(self.fc_bn2(self.fc2(s))),
            p=self.args.dropout,
            training=self.training,
        )

        # batch_size x action_size
        pi = self.fc3(s)
        # batch_size x 1
        v = self.fc4(s)

        return F.log_softmax(pi, dim=1), torch.tanh(v)
