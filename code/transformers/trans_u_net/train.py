import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from backbones.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from backbones.vit_seg_modeling import VisionTransformer as ViT_seg
from constants import EXPERIMENT_DIR, EXPORT_DIR
from trainer import trainer_isic
from transformers.trans_u_net.my_trainer import my_trainer_isic

parser = argparse.ArgumentParser()
# parser.add_argument('--root_path', type=str,
#                     default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='ISIC', help='experiment_name')
parser.add_argument('--train_dataset_size', type=int,
                    default=2700, help='Train dataset size. full dataset is 2700 images.')
parser.add_argument('--valid_dataset_size', type=int,
                    default=300, help='Validation dataset size. full dataset is 300 images.')
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=11000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=40, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        'ISIC': {
            'num_classes': 1,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.is_pretrain = True
    args.experiment = f"TU_{dataset_name}_{str(args.img_size)}"
    snapshot_path = os.path.join(EXPORT_DIR, "trans_u_net", "original", f"{args.experiment}")
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(
        args.vit_patches_size) if args.vit_patches_size != 16 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[
                                          0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.img_size)
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size)
        )
    model = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).to(device)
    # load backbone (pretrained on ImageNet)
    model.load_from(weights=np.load(config_vit.pretrained_path))

    trainer = {'ISIC': trainer_isic}
    trainer[dataset_name](
        args=args,
        model=model,
        snapshot_path=snapshot_path,
        device=device
    )

    # my_trainer = {'ISIC': my_trainer_isic}
    # my_trainer[dataset_name](
    #     args=args,
    #     transformer=model,
    #     dataset_size=args.dataset_size
    # )
