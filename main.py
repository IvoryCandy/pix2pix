import argparse
from solver import Solver


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='facades', help='input dataset')
parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--g_conv_dim', type=int, default=64)
parser.add_argument('--d_conv_dim', type=int, default=64)
parser.add_argument('--input_size', type=int, default=256, help='input size')
parser.add_argument('--in_channel', type=int, default=3, help='input channel size')
parser.add_argument('--out_channel', type=int, default=3, help='output channel size')
parser.add_argument('--threads', type=int, default=4, help='number of threads')
parser.add_argument('--num_epochs', type=int, default=200, help='number of train epochs')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lamb', type=float, default=100, help='lambda for L1 loss')
parser.add_argument('--beta_1', type=float, default=0.5, help='beta1 for Adam optimizer')
args = parser.parse_args()
print(args)


def main():
    solver = Solver(args)
    solver.run()


if __name__ == '__main__':
    main()
