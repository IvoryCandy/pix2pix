from __future__ import print_function
import argparse
import os

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms

from datasets import is_image_file, load_img, save_img

# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--datasets', required=True, help='facades')
parser.add_argument('--model', type=str, default='checkpoint/facades/netG_model_epoch_200.pth',
                    help='model file to use')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()
print(opt)

netG = torch.load(opt.model)

image_dir = "datasets/{}/test/a/".format(opt.dataset)
image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transform = transforms.Compose(transform_list)

for image_name in image_filenames:
    image = load_img(image_dir + image_name)
    image = transform(image)
    data = Variable(image, volatile=True).view(1, -1, 256, 256)

    if opt.cuda:
        netG = netG.cuda()
        data = data.cuda()

    out = netG(data)
    out = out.cpu()
    out_img = out.data[0]
    if not os.path.exists(os.path.join("result", opt.dataset)):
        os.mkdir(os.path.join("result", opt.dataset))
    save_img(out_img, "result/{}/{}".format(opt.dataset, image_name))
