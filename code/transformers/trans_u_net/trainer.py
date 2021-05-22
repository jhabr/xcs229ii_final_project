import logging
import os
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import VALIDATION_DIR, TRAIN_DIR
from transformers.trans_u_net.datasets.isic_dataset import ISICDataset
from transformers.trans_u_net.losses import JaccardLoss


def trainer_isic(args, model, snapshot_path, dataset_size=None, device="cpu"):
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = ISICDataset(resize_to=(args.img_size, args.img_size), size=dataset_size, image_dir=TRAIN_DIR)
    db_validation = ISICDataset(resize_to=(args.img_size, args.img_size), size=dataset_size, image_dir=VALIDATION_DIR)
    print("The length of train set is: {}".format(len(db_train)))

    train_loader = DataLoader(
        dataset=db_train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model.train()
    # TODO: maybe use BCEWITHLOGITSLOSS here: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    # https://discuss.pytorch.org/t/understanding-channels-in-binary-segmentation/79966
    # ce_loss = CrossEntropyLoss()
    ce_loss = BCEWithLogitsLoss()
    # ce_loss = SoftBCEWithLogitsLoss()
    jaccard_loss = JaccardLoss(mode='binary')
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(train_loader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(train_loader):
            image_batch, label_batch = sampled_batch
            # is: (batch_size, height, width, channels)
            # input: (batch_size, channels, height, width)
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)

            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch)
            loss_jaccard = jaccard_loss(outputs, label_batch)
            loss = 0.5 * loss_ce + 0.5 * loss_jaccard
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.sigmoid(outputs).round()
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...] * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
