import os
import random
import numpy as np
import torch
import torch.nn as nn
import slackweb
import hashlib


def cutmix(batch_images, batch_labels, p=0.4):
    def rand_bbox(size, lambda_):
        W = size[-2]
        H = size[-1]
        cut_rat = np.sqrt(1. - lambda_)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    if random.random() > p:
        beta = 1
        lambda_ = np.random.beta(beta, beta)
        rand_index = torch.randperm(batch_images.size()[0])
        bbx1, bby1, bbx2, bby2 = rand_bbox(batch_images.size(), lambda_)

        batch_images[:, :, bbx1:bbx2, bby1:bby2] = batch_images[rand_index, :, bbx1:bbx2, bby1:bby2]
        batch_labels[:, :, bbx1:bbx2, bby1:bby2] = batch_labels[rand_index, :, bbx1:bbx2, bby1:bby2]
    return batch_images, batch_labels


def set_seed(seed=None, cudnn_deterministic=True):
    if seed is None:
        seed = 42

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False


def TTA_rotate(x: torch.Tensor, model: nn.Module, apply_TTA=True):
    if apply_TTA:
        shape = x.shape
        x = [torch.rot90(x, k=i, dims=(-2, -1)) for i in [1]]
        x = torch.cat(x, dim=0)
        x = model(x)
        x = torch.sigmoid(x)
        x = x.reshape(1, shape[0], *shape[2:])
        x = [torch.rot90(x[0], k=-i, dims=(-2, -1)) for i in [1]]
        x = torch.stack(x, dim=0)
        return x.mean(0)
    else:
        x = model(x)
        x = torch.sigmoid(x)
        return x


def calculate_file_hash(file_path, algorithm='md5'):
    hash_object = hashlib.new(algorithm)
    with open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(4096*4096), b''):
            hash_object.update(chunk)
    return hash_object.hexdigest()


def has_overlap(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1 & set2
    return len(intersection) > 0


def slack_notify(text, mention=False):
    slack = slackweb.Slack(url="https://hooks.slack.com/services/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")
    if mention:
        text = "<@U05G5J71DT4>\n" + text
    slack.notify(text=text)
