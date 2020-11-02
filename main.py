import os
import argparse
from torch.backends import cudnn


def main(args):
    print(args)
    print("hallo")
    cudnn.benchmark = True

    if args.mode == 'train':
        print("train")

    elif args.mode == 'align':
        from core.align import align_faces
        align_faces(args, args.inp_dir, args.out_dir)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image resolution')

    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'sample', 'eval', 'align'],
                        help='This argument is used in solver')

    parser.add_argument('--wing_path', type=str, default='wing.ckpt')
    parser.add_argument('--lm_path', type=str, default='celeba_lm_mean.npz')

    parser.add_argument('--inp_dir', type=str, default='assets/representative/custom/female',
                        help='input directory when aligning faces')
    parser.add_argument('--out_dir', type=str, default='assets/representative/celeba_hq/src/female',
                        help='output directory when aligning faces')

    args = parser.parse_args()
    main(args)