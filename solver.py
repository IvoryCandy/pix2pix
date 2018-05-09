from __future__ import print_function
import os
from math import log10

import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from dataset import get_training_set, get_test_set

from model import Generator, Discriminator


class Solver(object):
    def __init__(self, args):
        # model
        self.g_optimizer = None
        self.d_optimizer = None
        self.generator = None
        self.discriminator = None
        self.MSELoss = None
        self.L1loss = None
        self.GPU_IN_USE = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.GPU_IN_USE else 'cpu')

        # Training settings
        self.dataset = args.dataset
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.threads = args.threads
        self.g_conv_dim = args.g_conv_dim
        self.d_conv_dim = args.d_conv_dim
        self.in_channel = args.in_channel
        self.out_channel = args.out_channel
        self.use_sigmoid = False

        # hyper-parameters
        self.lr = args.lr
        self.beta_1 = args.beta_1
        self.lamb = args.lamb

        # dataloader
        self.training_data_loader = None
        self.testing_data_loader = None

    def build_model(self):
        self.generator = Generator(in_channel=self.in_channel, out_channel=self.out_channel, g_conv_dim=self.g_conv_dim,
                                   norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9).to(self.device)
        self.generator.normal_init()
        self.discriminator = Discriminator(in_channel=self.in_channel + self.out_channel, d_conv_dim=self.d_conv_dim,
                                           num_layers=3, norm_layer=nn.BatchNorm2d,
                                           use_sigmoid=self.use_sigmoid).to(self.device)
        self.discriminator.normal_init()
        self.MSELoss = nn.MSELoss()
        self.L1loss = nn.L1Loss()

        if self.GPU_IN_USE:
            self.MSELoss.cuda()
            self.L1loss.cuda()
            cudnn.benchmark = True

        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta_1, 0.999))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta_1, 0.999))

    def build_dataset(self):
        root_path = "datasets/"
        train_set = get_training_set(root_path + self.dataset)
        test_set = get_test_set(root_path + self.dataset)
        self.training_data_loader = DataLoader(dataset=train_set, num_workers=self.threads, batch_size=self.batch_size,
                                               shuffle=True)
        self.testing_data_loader = DataLoader(dataset=test_set, num_workers=self.threads, batch_size=self.batch_size,
                                              shuffle=False)

    @staticmethod
    def to_data(x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

    @staticmethod
    def de_normalize(x):
        """Convert range (-1, 1) to (0, 1)"""
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def checkpoint(self, epoch):
        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")
        if not os.path.exists(os.path.join("checkpoint", self.dataset)):
            os.mkdir(os.path.join("checkpoint", self.dataset))
        net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(self.dataset, epoch)
        net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(self.dataset, epoch)
        torch.save(self.generator, net_g_model_out_path)
        torch.save(self.discriminator, net_d_model_out_path)
        print("Checkpoint saved to {}".format("checkpoint" + self.dataset))

    def mode_switch(self, mode):
        if mode == 'train':
            self.discriminator.train()
            self.generator.train()

        elif mode == 'eval':
            self.discriminator.eval()
            self.generator.eval()

    def train(self):
        self.mode_switch('train')
        for i, (data, target) in enumerate(self.training_data_loader):
            # forward
            data, target = data.to(self.device), target.to(self.device)
            fake_target = self.generator(data)

            ###########################
            # (1) train D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
            ###########################
            self.reset_grad()

            # train with fake
            fake_combined = torch.cat((data, fake_target), 1)
            fake_prediction = self.discriminator(fake_combined.detach())
            fake_d_loss = self.MSELoss(fake_prediction, torch.zeros(1, 1, fake_prediction.size(2),
                                                                    fake_prediction.size(3), device=self.device))

            # train with real
            real_combined = torch.cat((data, target), 1)
            real_prediction = self.discriminator(real_combined)
            real_d_loss = self.MSELoss(real_prediction, torch.ones(1, 1, real_prediction.size(2),
                                                                   real_prediction.size(3), device=self.device))

            # Combined loss
            loss_d = (fake_d_loss + real_d_loss) * 0.5
            loss_d.backward()
            self.d_optimizer.step()

            ##########################
            # (2) train G network: maximize log(D(x,G(x))) + L1(y,G(x))
            ##########################
            self.reset_grad()
            # First, G(A) should fake the discriminator
            fake_combined = torch.cat((data, fake_target), 1)
            fake_prediction = self.discriminator(fake_combined)
            g_loss_mse = self.MSELoss(fake_prediction, torch.ones(1, 1, fake_prediction.size(2),
                                                                  fake_prediction.size(3), device=self.device))

            # Second, G(A) = B
            g_loss_l1 = self.L1loss(fake_target, target) * self.lamb
            loss_g = g_loss_mse + g_loss_l1
            loss_g.backward()
            self.g_optimizer.step()

            print("({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(i, len(self.training_data_loader),
                                                                  loss_d.item(), loss_g.item()))

    def test(self):
        self.mode_switch('eval')
        avg_psnr = 0
        with torch.no_grad():
            for (data, target) in self.testing_data_loader:
                data, target = data.to(self.device), target.to(self.device)

                prediction = self.generator(data)
                mse = self.MSELoss(prediction, target)
                psnr = 10 * log10(1 / mse.data[0])
                avg_psnr += psnr

        print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(self.testing_data_loader)))

    def run(self):
        self.build_model()
        self.build_dataset()
        for e in range(1, self.num_epochs + 1):
            print("===> Epoch {}/{}".format(e, self.num_epochs))
            self.train()
            self.checkpoint(e)
            self.test()
