import os
import argparse
import torch
from torch.utils.data import *
import torchvision.datasets as dset
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn as nn
import copy

import pandas as pd
import numpy as np

## import BFA module
import models
from models.quantization import quan_Conv2d, quan_Linear

import warnings
warnings.filterwarnings("ignore")


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(
    description='Check vulnerable',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--arch',
                    metavar='ARCH',
                    default='resnet20_quan',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet20_quan)')

parser.add_argument('--test_batch_size',
                    type=int,
                    default=256,
                    help='Batch size.')

parser.add_argument('--seed',
                    type=int, 
                    default=0, 
                    help='manual seed')

parser.add_argument('--resume',
                    default="./pretrained/resnet20_8bit_cifar10_92_41.pth.tar",
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: pretrained/resnet20_8bit_92_41.pth.tar)')

parser.add_argument('--fine_tune',
    dest='fine_tune',
    action='store_true',
    help='fine tuning from the pre-trained model, force the start epoch be zero'
)

parser.add_argument(
    '--reset_weight',
    dest='reset_weight',
    action='store_true',
    help='enable the weight replacement with the quantized weight')

parser.add_argument(
    '--quan_bitwidth',
    type=int,
    default=None,
    help='the bitwidth used for quantization')

parser.add_argument(
    '--dataset',
    type=str,
    choices=['cifar10', 'cifar100', 'imagenet'],
    default='cifar10',
    help='Choose between Cifar10/100 and ImageNet.')

parser.add_argument(
    '--dataset_ratio',
    type=float,
    default=1,
    help='using test dataset ratio (default=1)') 

parser.add_argument('--save_path',
                    type=str,
                    default='./save/',
                    help='Folder to save csv.')


parser.add_argument('--data_path',
                    default='../dataset/',
                    type=str,
                    help='Path to dataset')

parser.add_argument('--workers',
                    type=int,
                    default=4,
                    help='number of data loading workers (default: 4)')


# Optimization options

args = parser.parse_args()

args.use_cuda = torch.cuda.is_available()  # check GPU
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def bin2int(input, num_bits):
    '''
    convert the unsigned integer (2's complement equivantly) back to the signed integer format
    with the bitwise operations. Note that, in order to perform the bitwise operation, the input
    tensor has to be in the integer format.
    '''
    if num_bits == 1:
        output = input*2-1
    elif num_bits > 1:
        mask = 2**(num_bits - 1) - 1
        output = -(input & ~mask) + (input & mask)
    return output

def int2bin(input, num_bits):
    '''
    convert the signed integer value into unsigned integer (2's complement equivalently).
    Note that, the conversion is different depends on number of bit used.
    '''
    output = input.clone()
    if num_bits == 1: # when it is binary, the conversion is different
        output = output/2 + .5
    elif num_bits > 1:
        output[input.lt(0)] = 2**num_bits + output[input.lt(0)]

    return output


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def validate(val_loader, model, criterion, summary_output=False, dataset_ratio=1.0):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    output_summary = [] # init a list for output summary

    if dataset_ratio < 1:
        max_count = round(len(val_loader.dataset) * dataset_ratio)
        max_iter = max_count // args.test_batch_size
    else:
        max_iter = 1000000

    with torch.no_grad():
        for i, (input, target) in enumerate(tqdm(val_loader)):
            if args.use_cuda:
                target = target.cuda()
                input = input.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)
            
            # summary the output
            if summary_output:
                tmp_list = output.max(1, keepdim=True)[1].flatten().cpu().numpy() # get the index of the max log-probability
                output_summary.append(tmp_list)


            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            if i > max_iter:
                break

        print(
            '  **Test (# of data {number:.2f})** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'
            .format(number=i*args.test_batch_size, top1=top1, top5=top5, error1=100 - top1.avg))
        
    if summary_output:
        output_summary = np.asarray(output_summary).flatten()
        return top1.avg, top5.avg, losses.avg, output_summary
    else:
        return top1.avg, top5.avg, losses.avg


def main():
    net = models.__dict__[args.arch]()
    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif args.dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        print("Don't find Datasets")

    if args.dataset == 'imagenet':
        if "inception" in args.arch:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(299),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            test_transform = transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])  # here is actually the validation dataset
        else :
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])  # here is actually the validation dataset

    else:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean, std)])

    if args.dataset == 'cifar10':
        test_data = dset.CIFAR10(args.data_path,
                                 train=False,
                                 transform=test_transform,
                                 download=True)

    elif args.dataset == 'cifar100':
        test_data = dset.CIFAR100(args.data_path,
                                 train=False,
                                 transform=test_transform,
                                 download=True)
    
    elif args.dataset == "imagenet":
        test_dir = os.path.join(args.data_path, 'val')
        test_data = dset.ImageFolder(test_dir, transform=test_transform)

    test_loader = DataLoader(test_data,
                    batch_size=args.test_batch_size,
                    shuffle=True,
                    num_workers=args.workers,
                    pin_memory=True)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            state_tmp = net.state_dict()
            if 'state_dict' in checkpoint.keys():
                state_tmp.update(checkpoint['state_dict'])
            else:
                state_tmp.update(checkpoint)
            net.load_state_dict(state_tmp)
        else:
            print("=>can't find loading checkpoint '{}'".format(args.resume))
    
        # update the step_size once the model is loaded. This is used for quantization.
    for m in net.modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            # simple step size update based on the pretrained model or weight init
            m.__reset_stepsize__()
            m.__reset_weight__()

    if not args.use_cuda:
        return "This program using only GPU"

    net_clean = copy.deepcopy(net)
    total_df = pd.DataFrame()
    count=0
    net.cuda()
    origin_top1, origin_top5, loss = validate(test_loader, net, nn.CrossEntropyLoss(), False, dataset_ratio=args.dataset_ratio)
    
    for i, (n, m) in enumerate(net.named_modules()):
        if isinstance(m, quan_Conv2d):
            count+=1
            full_length = m.weight.nelement()
            
            print("==*40")
            print(f"{n} module test")
            print("==*40")
            for weight_idx in range(full_length):
                net.cuda()
                original_m_weight = m.weight.data.clone()
                filter_idx = idx_to_filter(weight_idx, m)
                print(f"weight idx {weight_idx} / {full_length} ({filter_idx}) flip and acc check")
                flatten_weight = m.weight.cpu().detach().flatten()
                print(f"original {filter_idx} value : \t {flatten_weight[weight_idx]}")
                bin_w = int2bin(flatten_weight[weight_idx], m.N_bits).short()
                # select msb bit
                mask = (bin_w.clone().zero_() + 1) * (2**(m.N_bits-1))
                bin_w = bin_w ^ mask
                int_w = bin2int(bin_w, m.N_bits).float()
                print(f"Flip {filter_idx} value : \t {int_w}")
                flatten_weight[weight_idx] = int_w
                m.weight.data = flatten_weight.view(m.weight.data.size()).cuda()
                top1, top5, loss = validate(test_loader, net, nn.CrossEntropyLoss(), False, dataset_ratio=args.dataset_ratio)
                print("--"*40)
                dict_result = {"model_name": args.arch, "ori_top1":origin_top1, "ori_top5":origin_top5, "atk_top1":top1, "atk_top5":top5, "module":n, "module_idx":i, "weight_idx":filter_idx, "weight_idx_flatten":weight_idx}
                total_df= total_df.append(dict_result, ignore_index=True)
                save_path = os.path.join(args.save_path, "result.csv")
                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
                total_df.to_csv(save_path)
                m.weight.data = original_m_weight
                #net = copy.deepcopy(net_clean)
            print("==*40")
            print("end module test ", n)
            print("==*40")
    print("end network check")
    return         


def idx_to_filter(weight_idx, m):
    n, c, h, w = m.weight.shape
    out_n = weight_idx // (c*h*w)
    out_chw = weight_idx % (c*h*w)
    out_c = out_chw // (h*w)
    out_hw = out_chw % (h*w)
    out_h = out_hw // w
    out_w = out_hw % w

    return out_n, out_c, out_h, out_w


if __name__ == "__main__":
    main()
    print("end")
