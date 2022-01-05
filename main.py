import os, sys, shutil, time, random
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time, clustering_loss, change_quan_bitwidth
from tensorboardX import SummaryWriter
from tqdm import tqdm

import torch.nn.functional as F
import copy

import pandas as pd
import numpy as np

## import BFA module
import models
from models.quantization import quan_Conv2d, quan_Linear, quantize
from attack.BFA import *

## import distilled module

from distilled.distilled_data import distilled_target_dataset

import warnings
warnings.filterwarnings("ignore")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

################# Options ##################################################
############################################################################
parser = argparse.ArgumentParser(
    description='Training network for image classification',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data_path',
                    default='./dataset/',
                    type=str,
                    help='Path to dataset')

parser.add_argument(
    '--dataset',
    type=str,
    choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10', 'mnist'],
    default='cifar10',
    help='Choose between Cifar10/100 and ImageNet.')

parser.add_argument('--arch',
                    metavar='ARCH',
                    default='resnet20_quan',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet20_quan)')
# Optimization options

parser.add_argument('--epochs',
                    type=int,
                    default=300,
                    help='Number of epochs to train.')

parser.add_argument('--optimizer',
                    type=str,
                    default='SGD',
                    choices=['SGD', 'Adam', 'YF'])

parser.add_argument('--test_batch_size',
                    type=int,
                    default=256,
                    help='Batch size.')

parser.add_argument('--learning_rate',
                    type=float,
                    default=0.001,
                    help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')

parser.add_argument('--decay',
                    type=float,
                    default=1e-4,
                    help='Weight decay (L2 penalty).')

parser.add_argument('--schedule',
                    type=int,
                    nargs='+',
                    default=[80, 120],
                    help='Decrease learning rate at these epochs.')

parser.add_argument(
    '--gammas',
    type=float,
    nargs='+',
    default=[0.1, 0.1],
    help=
    'LR is multiplied by gamma on schedule, number of gammas should be equal to schedule'
)
# Checkpoints
parser.add_argument('--print_freq',
                    default=100,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 200)')

parser.add_argument('--save_path',
                    type=str,
                    default='./save/',
                    help='Folder to save checkpoints and log.')

parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--fine_tune',
    dest='fine_tune',
    action='store_true',
    help='fine tuning from the pre-trained model, force the start epoch be zero'
)

parser.add_argument('--quan_training',
    dest='qt',
    action='store_true',
    help="quantization aware training to change_bitwidth"
)

parser.add_argument('--pq',
    dest='pq',
    action='store_true',
    default=False,
    help="Post quantization. change quan bitwidth"
)

parser.add_argument('--model_only',
                    dest='model_only',
                    action='store_true',
                    help='only save the model without external utils_')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--gpu_id',
                    type=int,
                    default=0,
                    help='device range [0,ngpu-1]')
parser.add_argument('--workers',
                    type=int,
                    default=4,
                    help='number of data loading workers (default: 4)')
# random seed
parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')
# quantization
parser.add_argument(
    '--quan_bitwidth',
    type=int,
    default=None,
    help='the bitwidth used for quantization')
parser.add_argument(
    '--reset_weight',
    dest='reset_weight',
    action='store_true',
    help='enable the weight replacement with the quantized weight')
# Bit Flip Attack
parser.add_argument('--bfa',
                    dest='enable_bfa',
                    action='store_true',
                    help='enable the Bit Flip Attack')

parser.add_argument('--attack_sample_size',
                    type=int,
                    default=128,
                    help='attack sample size')
parser.add_argument('--n_iter',
                    type=int,
                    default=20,
                    help='number of attack iterations')
parser.add_argument(
    '--k_top',
    type=int,
    default=10,
    help='k weight with top ranking gradient used for bit-level gradient check.'
)
parser.add_argument('--random_bfa',
                    dest='random_bfa',
                    action='store_true',
                    help='perform the bit-flips randomly on weight bits')

parser.add_argument('--random_msb_bfa',
                    dest='random_msb_bfa',
                    action='store_true',
                    help='perform the bit-flips randomly on weight msb bits')

# Piecewise clustering
parser.add_argument('--clustering',
                    dest='clustering',
                    action='store_true',
                    help='add the piecewise clustering term.')
parser.add_argument('--lambda_coeff',
                    type=float,
                    default=1e-3,
                    help='lambda coefficient to control the clustering term')


#########################################################################
# Distilled dataset
parser.add_argument('--zebra',
                    dest='enable_zebra',
                    action='store_true',
                    help="enable the Zero data Based Rowhammer Attack")

parser.add_argument('--datafree_zebra',
                    dest='datafree_zebra',
                    action='store_true',
                    help="enable to Data-Free(can't access test data) ZeBRA")

parser.add_argument('--n_search',
                    type=float,
                    default=20,
                    help="generated distilled target data sample"
                    )

parser.add_argument('--distilled_batch_size',
                    type=int,
                    default=16,
                    help="make distilled dataset batch size"
                    )

parser.add_argument('--total_loss_bound',
                    type=float,
                    default=10,
                    help="generating distilled_data")

parser.add_argument('--CE_loss_lambda',
                    type=float,
                    default=0.1,
                    help="parameter to control the influence of the cross entropy of the generated distilled_data")

parser.add_argument('--distilled_loss_lambda',
                    type=float,
                    default=0.1,
                    help="parameter to control the influence of the cross entropy of the generated distilled_data")

parser.add_argument('--achieve_bit', 
                    type=int, 
                    default=10, 
                    help="attack success bit flip count")

parser.add_argument('--drop_acc', 
                    type=float,
                    default=10.0,
                    help="attack success accuracy & cifar10")

#################################################################

###distilled_target_valid_dataset hyper_param###

parser.add_argument('--distilled_target_valid_batch_size',
                type=int,
                default=64,
                help="make distilled_target_valid_dataset batch_size"
)

parser.add_argument('--distilled_target_valid_CE_loss_lambda',
                type=float,
                default=2,
                help="make distilled_target_valid_dataset CE_loss_lambda"
)

parser.add_argument('--distilled_target_valid_distill_loss_lambda',
                type=float,
                default=1,
                help="make distilled_target_valid_dataset distilled_loss_lambda"
)

parser.add_argument('--distilled_target_valid_total_loss_bound',
                type=float,
                default=1,
                help="make distilled_target_valid_dataset total_loss_bound"
)

parser.add_argument("--distilled_target_drop_acc1",
            type=float,
            default=11.0,
            help="attack success distilled target valid accuracy & cifar10"
)

parser.add_argument("--distilled_target_drop_acc5",
            type=float,
            default=52.0,
            help="attack success distilled target valid accuracy top-5"
)

parser.add_argument("--n_test_zebra_data",
        type=float,
        default=10,
        help="# of making distilled target data using validation"
)

##########################################################################


args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.ngpu == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        args.gpu_id)  # make only device #gpu_id visible, then

args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()  # check GPU

# Give a random seed if no manual configuration
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

cudnn.benchmark = True

###############################################################################
###############################################################################


def main():
    # Init logger6
    csv_save_path = os.path.join(args.save_path, "csv")
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.isdir(csv_save_path):
        os.makedirs(csv_save_path)
    log = open(
        os.path.join(args.save_path,
                     'log_seed_{}.txt'.format(args.manualSeed)), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')),
              log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()),
              log)

    # Init the tensorboard path and writer
    tb_path = os.path.join(args.save_path, 'tb_log',
                           'run_' + str(args.manualSeed))
    # logger = Logger(tb_path)
    writer = SummaryWriter(tb_path)

    # Init dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif args.dataset == 'svhn':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.dataset == 'mnist':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        assert False, "Unknown dataset : {}".format(args.dataset)

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

    if args.dataset == 'mnist':
        train_data = dset.MNIST(args.data_path,
                                train=True,
                                transform=train_transform,
                                download=True)
        test_data = dset.MNIST(args.data_path,
                               train=False,
                               transform=test_transform,
                               download=True)
        num_classes = 10
    elif args.dataset == 'cifar10':
        train_data = dset.CIFAR10(args.data_path,
                                  train=True,
                                  transform=train_transform,
                                  download=True)
        test_data = dset.CIFAR10(args.data_path,
                                 train=False,
                                 transform=test_transform,
                                 download=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_data = dset.CIFAR100(args.data_path,
                                   train=True,
                                   transform=train_transform,
                                   download=True)
        test_data = dset.CIFAR100(args.data_path,
                                  train=False,
                                  transform=test_transform,
                                  download=True)
        num_classes = 100
    elif args.dataset == 'svhn':
        train_data = dset.SVHN(args.data_path,
                               split='train',
                               transform=train_transform,
                               download=True)
        test_data = dset.SVHN(args.data_path,
                              split='test',
                              transform=test_transform,
                              download=True)
        num_classes = 10
    elif args.dataset == 'stl10':
        train_data = dset.STL10(args.data_path,
                                split='train',
                                transform=train_transform,
                                download=True)
        test_data = dset.STL10(args.data_path,
                               split='test',
                               transform=test_transform,
                               download=True)
        num_classes = 10

    elif args.dataset == 'imagenet':
        """
        We do not use ImageNet train data.
        BFA is using the attack data sample on ImageNet valid dataset
        ZeBRA is using the attack data sample on synthetic data
        """
        #train_dir = os.path.join(args.data_path, 'train')
        #train_data = dset.ImageFolder(train_dir, transform=train_transform)
        test_dir = os.path.join(args.data_path, 'val')
        train_data = dset.ImageFolder(test_dir, transform=train_transform)
        test_data = dset.ImageFolder(test_dir, transform=test_transform)
        num_classes = 1000
    else:
        assert False, 'Do not support dataset : {}'.format(args.dataset)


    print_log("=> creating model '{}'".format(args.arch), log)
    # Init model, criterion, and optimizer
    net = models.__dict__[args.arch](num_classes)
    print_log("=> network :\n {}".format(net), log)

    if args.use_cuda:
        if args.ngpu > 1:
            net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    # separate the parameters thus param groups can be updated by different optimizer
    all_param = [
        param for name, param in net.named_parameters()
        if not 'step_size' in name
    ]

    step_param = [
        param for name, param in net.named_parameters() if 'step_size' in name 
    ]

    if args.optimizer == "SGD":
        print("using SGD as optimizer")
        optimizer = torch.optim.SGD(all_param,
                                    lr=state['learning_rate'],
                                    momentum=state['momentum'],
                                    weight_decay=state['decay'],
                                    nesterov=True)

    elif args.optimizer == "Adam":
        print("using Adam as optimizer")
        optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad,
                                            all_param),
                                     lr=state['learning_rate'],
                                     weight_decay=state['decay'])

    elif args.optimizer == "RMSprop":
        print("using RMSprop as optimizer")
        optimizer = torch.optim.RMSprop(
            filter(lambda param: param.requires_grad, net.parameters()),
            lr=state['learning_rate'],
            alpha=0.99,
            eps=1e-08,
            weight_decay=0,
            momentum=0)

    if args.use_cuda:
        net.cuda()
        criterion.cuda()

    recorder = RecorderMeter(args.epochs)  # count number of epoches

    # considered training quantization bit-width (quantization based training parameter)
    if args.qt == True:
        if args.quan_bitwidth is not None:
            change_quan_bitwidth(net, args.quan_bitwidth)

        
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            recorder = RecorderMeter(args.epochs)  # count number of epoches

            if not (args.fine_tune):
                args.start_epoch = checkpoint['epoch']
                recorder = checkpoint['recorder']
                optimizer.load_state_dict(checkpoint['optimizer'])

            state_tmp = net.state_dict()
            if 'state_dict' in checkpoint.keys():
                state_tmp.update(checkpoint['state_dict'])
            else:
                state_tmp.update(checkpoint)

            net.load_state_dict(state_tmp)

            print_log(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, args.start_epoch), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume),
                    log)
    else:
        print_log(
            "=> do not use any checkpoint for {} model".format(args.arch), log)

    # Post quantization. So change quan_bitwidth    
    if args.pq == True:
        if args.quan_bitwidth is not None:
            change_quan_bitwidth(net, args.quan_bitwidth)


    # update the step_size once the model is loaded. This is used for quantization.
    for m in net.modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            # simple step size update based on the pretrained model or weight init
            m.__reset_stepsize__()

    # block for weight reset
    if args.reset_weight:
        for m in net.modules():
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
                m.__reset_weight__()
                # print(m.weight)

    attacker = BFA(criterion, net, args.k_top)
    net_clean = copy.deepcopy(net)
    
    # imagenet is not load train_data only use test data
    
    train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.attack_sample_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=False)

    

    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=args.test_batch_size,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              pin_memory=False)


    if args.enable_bfa:
        perform_attack(attacker, net, net_clean, train_loader, test_loader,
                       args.n_iter, log, writer, csv_save_path=csv_save_path,
                       random_attack=args.random_bfa)
        return


    if args.enable_zebra:
        print_log("attack_distilled_profile", log)
        print_log("==========================================", log)
        print_log("distilled_batch size : {}".format(args.distilled_batch_size), log)
        print_log("attack_sample size : {}".format(args.attack_sample_size), log)
        print_log("CE_loss_lambda {}".format(args.CE_loss_lambda), log)
        print_log("distilled_loss_lambda {}".format(args.distilled_loss_lambda), log)
        print_log("total loss bound {}".format(args.total_loss_bound), log)
        print_log("==========================================", log)
        
        if "inception" in args.arch:
            print("num_batch : ", args.n_search * (args.attack_sample_size // args.distilled_batch_size))
            distilled_attack_dataset = distilled_target_dataset(teacher_model=net_clean, dataset=args.dataset, num_classes = num_classes,batch_size=args.distilled_batch_size, 
                num_batch=args.n_search * (args.attack_sample_size // args.distilled_batch_size), 
                for_inception=True,
                total_loss_bound=args.total_loss_bound, CE_loss_lambda=args.CE_loss_lambda, distilled_loss_lambda=args.distilled_loss_lambda)
        else:
            print("num_batch : ", args.n_search * (args.attack_sample_size // args.distilled_batch_size))
            distilled_attack_dataset = distilled_target_dataset(teacher_model=net_clean, dataset=args.dataset, num_classes=num_classes,batch_size=args.distilled_batch_size, 
                num_batch=args.n_search * (args.attack_sample_size // args.distilled_batch_size), 
                total_loss_bound=args.total_loss_bound, CE_loss_lambda=args.CE_loss_lambda, distilled_loss_lambda=args.distilled_loss_lambda)
        
            
        
        distilled_attack_loader = torch.utils.data.DataLoader(
        distilled_attack_dataset,
        batch_size=args.attack_sample_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False)

        zebra_attack(attacker=attacker, model_clean=net_clean, attack_dataloader=distilled_attack_loader, test_loader=test_loader, N_iter=args.n_iter, 
        achieve_bit=args.achieve_bit, drop_acc=args.drop_acc, log=log, writer=writer, csv_save_path=csv_save_path)
        
        return


    if args.datafree_zebra:


        print_log("play datafree_zebra", log)
        print_log("attack_distilled_profile", log)
        print_log("==========================================", log)
        print_log("distilled_batch size : {}".format(args.distilled_batch_size), log)
        print_log("attack_sample size : {}".format(args.attack_sample_size), log)
        print_log("CE_loss_lambda {}".format(args.CE_loss_lambda), log)
        print_log("distilled_loss_lambda {}".format(args.distilled_loss_lambda), log)
        print_log("total loss bound {}".format(args.total_loss_bound), log)
        print_log("==========================================", log)

        distilled_attack_dataset = distilled_target_dataset(teacher_model=net_clean, dataset=args.dataset, batch_size=args.distilled_batch_size, 
        num_batch=args.n_search * (args.attack_sample_size // args.distilled_batch_size),  
        CE_loss_lambda=args.CE_loss_lambda, 
        distilled_loss_lambda=args.distilled_loss_lambda,
        total_loss_bound=args.total_loss_bound
        )
        distilled_attack_loader = torch.utils.data.DataLoader(
        distilled_attack_dataset,
        batch_size=args.attack_sample_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False)

        distill_target_valid_batch_size = args.distilled_target_valid_batch_size
        distilled_target_valid_CE_loss_lambda = args.distilled_target_valid_CE_loss_lambda
        distilled_target_valid_distill_loss_lambda = args.distilled_target_valid_distill_loss_lambda
        distilled_target_valid_total_loss_bound = args.distilled_target_valid_total_loss_bound
        distilled_drop_acc_1 = args.distilled_target_drop_acc1
        distilled_drop_acc_5 = args.distilled_target_drop_acc5
        
        print_log("test_distilled_profile", log)
        print_log("==========================================", log)
        print_log("test_batch size : {}".format(distill_target_valid_batch_size), log)
        print_log("CE_loss_lambda {}".format(distilled_target_valid_CE_loss_lambda), log)
        print_log("distilled_loss_lambda {}".format(distilled_target_valid_distill_loss_lambda), log)
        print_log("total loss bound {}".format(distilled_target_valid_total_loss_bound), log)
        print_log("success attack top-1 accuracy : {}".format(distilled_drop_acc_1), log)
        print_log("success attack top-5 accuracy : {}".format(distilled_drop_acc_5), log)

        print_log("==========================================", log)


        distilled_test_dataset = distilled_target_dataset(teacher_model=net_clean, 
        dataset=args.dataset, 
        batch_size=distill_target_valid_batch_size, 
        num_batch=args.n_test_zebra_data,
        CE_loss_lambda=distilled_target_valid_CE_loss_lambda, 
        distilled_loss_lambda=distilled_target_valid_distill_loss_lambda,
        total_loss_bound=distilled_target_valid_total_loss_bound
        )

        distilled_test_loader = torch.utils.data.DataLoader(
        distilled_test_dataset,
        batch_size=args.attack_sample_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False)

        datafree_zebra_attack(attacker=attacker, model_clean=net_clean, 
        attack_dataloader=distilled_attack_loader, 
        distilled_test_loader=distilled_test_loader,
        test_loader=test_loader, N_iter=args.n_iter, 
        achieve_bit=args.achieve_bit, drop_acc1=distilled_drop_acc_1, drop_acc5=distilled_drop_acc_5, log=log, writer=writer, csv_save_path=csv_save_path)

    if args.evaluate:
        _,_,_, output_summary = validate(test_loader, net, criterion, log, summary_output=True)
        pd.DataFrame(output_summary).to_csv(os.path.join(args.save_path, 'output_summary_{}.csv'.format(args.arch)),
                                            header=['top-1 output'], index=False)
        return


    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()

    for epoch in range(args.start_epoch, args.epochs):
        current_learning_rate, current_momentum = adjust_learning_rate(
            optimizer, epoch, args.gammas, args.schedule)
        # Display simulation time
        need_hour, need_mins, need_secs = convert_secs2time(
            epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(
            need_hour, need_mins, need_secs)

        print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [LR={:6.4f}][M={:1.2f}]'.format(time_string(), epoch, args.epochs,
                                                                                   need_time, current_learning_rate,
                                                                                   current_momentum) \
            + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False),
                                                               100 - recorder.max_accuracy(False)), log)

        # train for one epoch
        train_acc, train_los = train(train_loader, net, criterion, optimizer,
                                     epoch, log)

        # evaluate on validation set
        val_acc, _, val_los = validate(test_loader, net, criterion, log)
        recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        is_best = val_acc >= recorder.max_accuracy(False)

        if args.model_only:
            checkpoint_state = {'state_dict': net.state_dict}
        else:
            checkpoint_state = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': net.state_dict(),
                'recorder': recorder,
                'optimizer': optimizer.state_dict(),
            }

        save_checkpoint(checkpoint_state, is_best, args.save_path,
                        'checkpoint.pth.tar', log)

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        recorder.plot_curve(os.path.join(args.save_path, 'curve.png'))

        # save addition accuracy log for plotting
        accuracy_logger(base_dir=args.save_path,
                        epoch=epoch,
                        train_accuracy=train_acc,
                        test_accuracy=val_acc)

        # ============ TensorBoard logging ============#

        ## Log the graidents distribution
        for name, param in net.named_parameters():
            name = name.replace('.', '/')
            try:
                writer.add_histogram(name + '/grad',
                                    param.grad.clone().cpu().data.numpy(),
                                    epoch + 1,
                                    bins='tensorflow')
            except:
                pass
            
            try:
                writer.add_histogram(name, param.clone().cpu().data.numpy(),
                                      epoch + 1, bins='tensorflow')
            except:
                pass
            
        total_weight_change = 0 
            
        for name, module in net.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                try:
                    writer.add_histogram(name+'/bin_weight', module.bin_weight.clone().cpu().data.numpy(), epoch + 1,
                                        bins='tensorflow')
                    writer.add_scalar(name + '/bin_weight_change', module.bin_weight_change, epoch+1)
                    total_weight_change += module.bin_weight_change
                    writer.add_scalar(name + '/bin_weight_change_ratio', module.bin_weight_change_ratio, epoch+1)
                except:
                    pass
                
        writer.add_scalar('total_weight_change', total_weight_change, epoch + 1)
        print('total weight changes:', total_weight_change)

        writer.add_scalar('loss/train_loss', train_los, epoch + 1)
        writer.add_scalar('loss/test_loss', val_los, epoch + 1)
        writer.add_scalar('accuracy/train_accuracy', train_acc, epoch + 1)
        writer.add_scalar('accuracy/test_accuracy', val_acc, epoch + 1)
    # ============ TensorBoard logging ============#

    log.close()


def perform_attack(attacker, model, model_clean, train_loader, test_loader,
                   N_iter, log, writer, csv_save_path=None, random_attack=False):
    # Note that, attack has to be done in evaluation model due to batch-norm.
    # see: https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146
    model.eval()
    losses = AverageMeter()
    iter_time = AverageMeter()
    attack_time = AverageMeter()


    # attempt to use the training data to conduct BFA
    for _, (data, target) in enumerate(train_loader):
        if args.use_cuda:
            target = target.cuda()
            data = data.cuda()
        # Override the target to prevent label leaking
        _, target = model(data).data.max(1)
        break

    # evaluate the test accuracy of clean model
    val_acc_top1, val_acc_top5, val_loss, output_summary = validate(test_loader, model,
                                                    attacker.criterion, log, summary_output=True)
    tmp_df = pd.DataFrame(output_summary, columns=['top-1 output'])
    tmp_df['BFA iteration'] = 0
    tmp_df.to_csv(os.path.join(args.save_path, 'output_summary_{}_BFA_0.csv'.format(args.arch)),
                                        index=False)

    writer.add_scalar('attack/val_top1_acc', val_acc_top1, 0)
    writer.add_scalar('attack/val_top5_acc', val_acc_top5, 0)
    writer.add_scalar('attack/val_loss', val_loss, 0)

    print_log('k_top is set to {}'.format(args.k_top), log)
    print_log('Attack sample size is {}'.format(data.size()[0]), log)
    end = time.time()
    
    df = pd.DataFrame() #init a empty dataframe for logging
    last_val_acc_top1 = val_acc_top1
        # Stop the attack if the accuracy is below the configured break_acc.
    if args.drop_acc is None:
        if args.dataset == 'cifar10':
            break_acc = 10.0
        elif args.dataset == 'imagenet':
            break_acc = 0.2
        else:
            print("default setting break acc 10")
            break_acc = 10.0
    else:
        break_acc = args.drop_acc

    for i_iter in range(N_iter):
        print_log('**********************************', log)
        if not random_attack:
            attack_log = attacker.progressive_bit_search(model, data, target)
        else:
            attack_log = attacker.random_flip_one_bit(model)
            
        
        # measure data loading time
        attack_time.update(time.time() - end)
        end = time.time()

        h_dist = hamming_distance(model, model_clean)

        # record the loss
        if hasattr(attacker, "loss_max"):
            losses.update(attacker.loss_max, data.size(0))

        print_log(
            'Iteration: [{:03d}/{:03d}]   '
            'Attack Time {attack_time.val:.3f} ({attack_time.avg:.3f})  '.
            format((i_iter + 1),
                   N_iter,
                   attack_time=attack_time,
                   iter_time=iter_time) + time_string(), log)
        try:
            print_log('loss before attack: {:.4f}'.format(attacker.loss.item()),
                    log)
            print_log('loss after attack: {:.4f}'.format(attacker.loss_max), log)
        except:
            pass
        
        #print_log('bit flips: {:.0f}'.format(attacker.bit_counter), log)
        print_log('hamming_dist: {:.0f}'.format(h_dist), log)

        writer.add_scalar('attack/bit_flip', attacker.bit_counter, i_iter + 1)
        writer.add_scalar('attack/h_dist', h_dist, i_iter + 1)
        writer.add_scalar('attack/sample_loss', losses.avg, i_iter + 1)

        # exam the BFA on entire val dataset
        val_acc_top1, val_acc_top5, val_loss, output_summary = validate(
            test_loader, model, attacker.criterion, log, summary_output=True)
        
        
        # add additional info for logging
        acc_drop = last_val_acc_top1 - val_acc_top1
        last_val_acc_top1 = val_acc_top1
        
        for i in range(attack_log.__len__()):
            attack_log[i].append(val_acc_top1)
            attack_log[i].append(acc_drop)
        
        df = df.append(attack_log, ignore_index=True)

        writer.add_scalar('attack/val_top1_acc', val_acc_top1, i_iter + 1)
        writer.add_scalar('attack/val_top5_acc', val_acc_top5, i_iter + 1)
        writer.add_scalar('attack/val_loss', val_loss, i_iter + 1)

        # measure elapsed time
        iter_time.update(time.time() - end)
        print_log(
            'iteration Time {iter_time.val:.3f} ({iter_time.avg:.3f})'.format(
                iter_time=iter_time), log)
        end = time.time()

        if val_acc_top1 <= break_acc:
            break
        
    # attack profile
    column_list = ['module idx', 'bit-flip idx', 'module name', 'weight idx',
                  'weight before attack', 'weight after attack', 'validation accuracy',
                  'accuracy drop']
    df.columns = column_list
    df['trial seed'] = args.manualSeed
    if csv_save_path is not None:
        csv_file_name = 'bfa_attack_profile_{}.csv'.format(args.manualSeed)
        export_csv = df.to_csv(os.path.join(csv_save_path, csv_file_name), index=None)

    return

def zebra_attack(attacker, model_clean, attack_dataloader, test_loader, N_iter, achieve_bit, drop_acc, log, writer, csv_save_path=None, random_attack=False):
    """
    zebra attack algorithm
    attacker : defined attacker (BFA attacker)
    model_clean : pretrained victim model 
    attack_dataloader : Dataloader to use to find the vulnerable bit
    test_loader : Dataloader to validate attack performance
    N_iter : # of iteration to bit-flips
    achieve_bit : The objective of the number of bit flips
    drop_acc : The objective of accuracy
    log : logging
    writer : SummaryWriter
    csv_save_path : return the result to csv format
    random_attack : random bit flip attack
    """
    # attack success and flip bit count
    success_bit = N_iter
    print_log("ZeBRA Attack start", log)
    break_acc = 0
    temp_acc = 100
    column_list = ['module idx', 'bit-flip idx', 'module name', 'weight idx',
                        'weight before attack', 'weight after attack', 'validation accuracy', 'top5 validation accuracy',
                        'accuracy drop', 'distilled data batch idx']
            
    # success drop_acc but do not goal achieve bit trigger
    drop_acc_success_trigger = False
    attack_df = pd.DataFrame() #init a empty dataframe for attack_log logging

    for distilled_batch_i, (data, target) in enumerate(attack_dataloader):
        print_log("=================================", log)
        print_log("ZeBRA attack iter {}".format(distilled_batch_i), log)
        print_log("using distilled target data mini batch {}".format(distilled_batch_i), log)
        # bfa_temp_df setting
        bfa_temp_df = pd.DataFrame()
        
        if args.use_cuda:
            data = data.cuda()
            target = target.cuda()

        # clean model initialize
        model = copy.deepcopy(model_clean)
        
        model.eval()
        losses = AverageMeter()
        iter_time = AverageMeter()
        attack_time = AverageMeter()

        # evaluate the test accuracy of clean model
        val_acc_top1, val_acc_top5, val_loss, output_summary = validate(test_loader, model,
                                                        attacker.criterion, log, summary_output=True)

        writer.add_scalar('attack/val_top1_acc', val_acc_top1, 0)
        writer.add_scalar('attack/val_top5_acc', val_acc_top5, 0)
        writer.add_scalar('attack/val_loss', val_loss, 0)

        print_log('k_top is set to {}'.format(args.k_top), log)
        print_log('Attack BFA sample size is {}'.format(data.size()[0]), log)
        end = time.time()
        
        last_val_acc_top1 = val_acc_top1
        
        if drop_acc is None:
            if args.dataset == 'cifar10':
                break_acc = 10.0
            elif args.dataset == 'imagenet':
                break_acc = 0.2
            else:
                print("default setting break acc 10")
                break_acc = 10.0
        else:
            break_acc = drop_acc
        
        # repeat until accuracy drops to defined drop_acc
        for i_iter in range(success_bit):
            print_log('**********************************', log)
            print_log('success_bit : {}'.format(success_bit), log)

            attack_log = attacker.progressive_bit_search(model, data, target)            
            # measure data loading time
            attack_time.update(time.time() - end)
            end = time.time()

            h_dist = hamming_distance(model, model_clean)

            # record the loss
            if hasattr(attacker, "loss_max"):
                losses.update(attacker.loss_max, data.size(0))

            print_log(
                'Iteration: [{:03d}/{:03d}]   '
                'Attack Time {attack_time.val:.3f} ({attack_time.avg:.3f})  '.
                format((i_iter + 1),
                    success_bit,
                    attack_time=attack_time,
                    iter_time=iter_time) + time_string(), log)
            try:
                print_log('loss before attack: {:.4f}'.format(attacker.loss.item()),
                        log)
                print_log('loss after attack: {:.4f}'.format(attacker.loss_max), log)
            except:
                pass
            
            print_log('hamming_dist: {:.0f}'.format(h_dist), log)

            writer.add_scalar('attack/bit_flip', attacker.bit_counter, i_iter + 1)
            writer.add_scalar('attack/h_dist', h_dist, i_iter + 1)
            writer.add_scalar('attack/sample_loss', losses.avg, i_iter + 1)

            # exam the BFA on entire val dataset
            val_acc_top1, val_acc_top5, val_loss, output_summary = validate(
                test_loader, model, attacker.criterion, log, summary_output=True)
            
            # add additional info for logging
            acc_drop = last_val_acc_top1 - val_acc_top1
            last_val_acc_top1 = val_acc_top1
            
            for i in range(attack_log.__len__()):
                attack_log[i].append(val_acc_top1)
                attack_log[i].append(val_acc_top5)
                attack_log[i].append(acc_drop)
                attack_log[i].append(distilled_batch_i)
            
            temp_df = pd.DataFrame(attack_log, columns=column_list)
            attack_df = attack_df.append(temp_df, ignore_index=True)
            bfa_temp_df = bfa_temp_df.append(temp_df, ignore_index=True)


            writer.add_scalar('attack/val_top1_acc', val_acc_top1, i_iter + 1)
            writer.add_scalar('attack/val_top5_acc', val_acc_top5, i_iter + 1)
            writer.add_scalar('attack/val_loss', val_loss, i_iter + 1)

            # measure elapsed time
            iter_time.update(time.time() - end)
            print_log(
                'iteration Time {iter_time.val:.3f} ({iter_time.avg:.3f})'.format(
                    iter_time=iter_time), log)
            end = time.time()

            if val_acc_top1 <= temp_acc:
                temp_acc = val_acc_top1
                temp_hamming_dist = h_dist
                drop_acc_top_idx = distilled_batch_i
            
            if val_acc_top1 <= break_acc:
                # for reducing search time, change success_bit to h_dist (local_minima) 
                success_bit = h_dist
                local_success_data_idx = distilled_batch_i
                drop_acc_success_trigger = True
                break
            
            if success_bit < i_iter:
                break
        
        # After the iteration is finished, attacker's bit_count set zero
        attacker.bit_counter = 0

        # attack profile
        # success attack
        if success_bit <= achieve_bit:

            success_save_path = os.path.join(args.save_path, 'success')    
            if not os.path.exists(success_save_path):
                os.makedirs(success_save_path)
            
            success_log = open(
                os.path.join(success_save_path,
                'success_log_{}.txt'.format(args.manualSeed)), 'w')

            print_log('save path : {}'.format(success_save_path), success_log)
            print_log("===================================", success_log)
            print_log("success ZeBRA attack", success_log)
            print_log("distilled_target_dataset_batch_idx {}".format(distilled_batch_i), success_log)
            print_log("Success Bit Flip : \t {}".format(success_bit), success_log)
            print_log("Success drop acc : \t {}".format(temp_acc), success_log)
            print_log("attack batch size : \t {}".format(args.attack_sample_size), success_log)
            print_log("total_loss_bound : {}".format(args.total_loss_bound), success_log)
            print_log("distilled_loss_lambda {}".format(args.distilled_loss_lambda), success_log)
            print_log("CE_Loss_lambda {}".format(args.CE_loss_lambda), success_log)
            print_log("=========================", success_log)
            

            
            attack_df['trial seed'] = args.manualSeed
            
            save_df = attack_df[attack_df['distilled data batch idx'] == distilled_batch_i]

            if csv_save_path is not None:
                csv_file_name = 'ZeBRA_attack_profile_{}.csv'.format(args.manualSeed)
                export_csv = save_df.to_csv(os.path.join(csv_save_path, csv_file_name), index=None)

            return

    # attack failed
    fail_save_path = os.path.join(args.save_path, 'fail')
    if not os.path.exists(fail_save_path):
        os.makedirs(fail_save_path)

    fail_log = open(os.path.join(fail_save_path, 'fail_log_{}.txt'.format(args.manualSeed)), 'w')

    print_log('fail save path : {}'.format(fail_save_path), fail_log)
    print_log('==================================', fail_log)
    print_log('fail Zebra attack', fail_log)
    
    # The attack was successful but required more bit flips than the achieve_bits. 
    if drop_acc_success_trigger:

        print_log("success acc drop but number of bits do not goal\n", fail_log)
        print_log('drop best acc : {}'.format(temp_acc), fail_log)
        print_log('drop best acc dataset idx : {}'.format(drop_acc_top_idx), fail_log)
        print_log('The minimum number of bit flips during attacks to reduce accuracy uintil drop_acc  : {}'.format(success_bit), fail_log)
        attack_df['trial seed'] = args.manualSeed
        save_df = attack_df[attack_df['distilled data batch idx'] == local_success_data_idx]
        if csv_save_path is not None:
            csv_file_name = 'ZeBRA_fail_attack_profile_{}.csv'.format(args.manualSeed)
            export_csv = save_df.to_csv(os.path.join(csv_save_path, csv_file_name), index=None)

    else:
        print_log('drop best acc : {}'.format(temp_acc), fail_log)
        print_log('drop best acc dataset idx : {}'.format(drop_acc_top_idx), fail_log)
        
        attack_df['trial seed'] = args.manualSeed

        save_df = attack_df[attack_df['distilled data batch idx'] == drop_acc_top_idx]

        if csv_save_path is not None:
            csv_file_name = 'ZeBRA_fail_attack_profile_{}.csv'.format(args.manualSeed)
            export_csv = save_df.to_csv(os.path.join(csv_save_path, csv_file_name), index=None)
        
    print_log('# of bit flip of drop best acc: {}'.format(temp_hamming_dist), fail_log)
    print_log('achieve bit : {}'.format(achieve_bit), fail_log)
    print_log('minimum flip bit : {}'.format(success_bit), fail_log)
    print_log("attack batch size : \t {}".format(args.attack_sample_size), fail_log)
    print_log("total_loss_bound : {}".format(args.total_loss_bound), fail_log)
    print_log("distilled_loss_lambda {}".format(args.distilled_loss_lambda), fail_log)
    print_log("CE_Loss_lambda {}".format(args.CE_loss_lambda), fail_log)
    print_log("=========================", fail_log)

    return


def datafree_zebra_attack(attacker, model_clean, attack_dataloader, distilled_test_loader, test_loader, N_iter, achieve_bit, drop_acc1, drop_acc5, log, writer, csv_save_path=None, random_attack=False):
    """
    zebra attack algorithm
    attacker : defined attacker (BFA attacker)
    model_clean : pretrained victim model 
    attack_dataloader : Dataloader to use to find the vulnerable bit
    test_loader : Dataloader to validate attack performance
    N_iter : # of iteration to bit-flips
    achieve_bit : The objective of the number of bit flips
    drop_acc1 : The objective of accuracy(top-1)
    drop_acc5 : The objective of accuracy(top-5)
    log : logging
    writer : SummaryWriter
    csv_save_path : return the result to csv format
    random_attack : random bit flip attack
    """
    # attack success and flip bit count
    success_bit = N_iter
    print_log("DataFree ZeBRA Attack start", log)
    temp_acc = 100
    column_list = ['module idx', 'bit-flip idx', 'module name', 'weight idx',
                        'weight before attack', 'weight after attack', 'validation accuracy', 'top5 validation accuracy',
                        'accuracy drop', 'distilled data batch idx']
            
    # success drop_acc but do not goal achieve bit trigger
    drop_acc_success_trigger = False
    attack_df = pd.DataFrame() #init a empty dataframe for attack_log logging


    for distilled_batch_i, (data, target) in enumerate(attack_dataloader):
        print_log("=================================", log)
        print_log("ZeBRA attack iter {}".format(distilled_batch_i), log)
        print_log("using distilled target data mini batch {}".format(distilled_batch_i), log)
    
        # bfa_temp_df setting
        bfa_temp_df = pd.DataFrame()
        
        if args.use_cuda:
            data = data.cuda()
            target = target.cuda()

       # clean model initialize
        model = copy.deepcopy(model_clean)
        
        model.eval()
        losses = AverageMeter()
        iter_time = AverageMeter()
        attack_time = AverageMeter()

        # evaluate the test accuracy of clean model
        print_log("real data accuracy", log)
        val_acc_top1, val_acc_top5, val_loss, output_summary = validate(test_loader, model,
                                                        attacker.criterion, log, summary_output=True)

        writer.add_scalar('attack/val_top1_acc', val_acc_top1, 0)
        writer.add_scalar('attack/val_top5_acc', val_acc_top5, 0)
        writer.add_scalar('attack/val_loss', val_loss, 0)

        print_log('k_top is set to {}'.format(args.k_top), log)
        print_log('Attack BFA sample size is {}'.format(data.size()[0]), log)
        end = time.time()
        
        last_val_acc_top1 = val_acc_top1
        
        # repeat until accuracy drops to defined drop_acc
        for i_iter in range(success_bit):
            
            print_log('**********************************', log)
            print_log('success_bit : {}'.format(success_bit), log)

            attack_log = attacker.progressive_bit_search(model, data, target)            
            # measure data loading time
            attack_time.update(time.time() - end)
            end = time.time()

            h_dist = hamming_distance(model, model_clean)

            # record the loss
            if hasattr(attacker, "loss_max"):
                losses.update(attacker.loss_max, data.size(0))

            print_log(
                'Iteration: [{:03d}/{:03d}]   '
                'Attack Time {attack_time.val:.3f} ({attack_time.avg:.3f})  '.
                format((i_iter + 1),
                    success_bit,
                    attack_time=attack_time,
                    iter_time=iter_time) + time_string(), log)
            try:
                print_log('loss before attack: {:.4f}'.format(attacker.loss.item()),
                        log)
                print_log('loss after attack: {:.4f}'.format(attacker.loss_max), log)
            except:
                pass
            
            print_log('hamming_dist: {:.0f}'.format(h_dist), log)

            writer.add_scalar('attack/bit_flip', attacker.bit_counter, i_iter + 1)
            writer.add_scalar('attack/h_dist', h_dist, i_iter + 1)
            writer.add_scalar('attack/sample_loss', losses.avg, i_iter + 1)

            # exam the BFA on entire val dataset
            print_log("distilled_test_loader accuracy", log)
            val_acc_top1, val_acc_top5, val_loss, output_summary = validate(
                distilled_test_loader, model, attacker.criterion, log, summary_output=True)
            
            if args.dataset == "cifar10":
                real_acc_top1, real_acc_top5, real_loss, _ = validate(test_loader, model, attacker.criterion, log, summary_output=True)
            
            # add additional info for logging
            acc_drop = last_val_acc_top1 - val_acc_top1
            last_val_acc_top1 = val_acc_top1
            
            for i in range(attack_log.__len__()):
                attack_log[i].append(val_acc_top1)
                attack_log[i].append(val_acc_top5)
                attack_log[i].append(acc_drop)
                attack_log[i].append(distilled_batch_i)
                
            
            temp_df = pd.DataFrame(attack_log, columns=column_list)
            attack_df = attack_df.append(temp_df, ignore_index=True)
            bfa_temp_df = bfa_temp_df.append(temp_df, ignore_index=True)


            writer.add_scalar('attack/val_top1_acc', val_acc_top1, i_iter + 1)
            writer.add_scalar('attack/val_top5_acc', val_acc_top5, i_iter + 1)
            writer.add_scalar('attack/val_loss', val_loss, i_iter + 1)

            # measure elapsed time
            iter_time.update(time.time() - end)
            print_log(
                'iteration Time {iter_time.val:.3f} ({iter_time.avg:.3f})'.format(
                    iter_time=iter_time), log)
            end = time.time()

            if val_acc_top1 <= temp_acc:
                temp_acc = val_acc_top1
                temp_hamming_dist = h_dist
                drop_acc_top_idx = distilled_batch_i
            
            if val_acc_top1 <= drop_acc1 and val_acc_top5 <= drop_acc5:
                success_bit = h_dist
                local_success_data_idx = distilled_batch_i
                last_df = attack_df.iloc[-1]

                test_val_acc_top1, test_val_acc_top5, test_val_loss, test_output_summary = validate(
                test_loader, model, attacker.criterion, log, summary_output=True)
                
                acc_top1_gap = test_val_acc_top1 - last_df['validation accuracy']
                acc_top5_gap = test_val_acc_top5 - last_df['top5 validation accuracy']

                test_data_dict = {"test_val_acc1":test_val_acc_top1, "test_val_acc5":test_val_acc_top5, 
                "test_val_loss":test_val_loss, "acc_top1_gap":acc_top1_gap, "acc_top5_gap":acc_top5_gap,
                "distilled data batch idx": distilled_batch_i
                }
                attack_df = attack_df.append(test_data_dict, ignore_index=True)
                bfa_temp_df = bfa_temp_df.append(test_data_dict, ignore_index=True)
                drop_acc_success_trigger = True
                break
            
            if success_bit < i_iter:
                break
        
        attacker.bit_counter = 0

        # attack profile


        if success_bit <= achieve_bit:

            success_save_path = os.path.join(args.save_path, 'success')    
            if not os.path.exists(success_save_path):
                os.makedirs(success_save_path)
            
            success_log = open(
                os.path.join(success_save_path,
                'success_log_{}.txt'.format(args.manualSeed)), 'w')

            print_log('save path : {}'.format(success_save_path), success_log)
            print_log("===================================", success_log)
            print_log("success ZeBRA attack", success_log)
            print_log("distilled_target_dataset_batch_idx {}".format(distilled_batch_i), success_log)
            print_log("Success Bit Flip : \t {}".format(success_bit), success_log)
            print_log("Success drop acc : \t {}".format(temp_acc), success_log)
            print_log("attack batch size : \t {}".format(args.attack_sample_size), success_log)
            print_log("total_loss_bound : {}".format(args.total_loss_bound), success_log)
            print_log("distilled_loss_lambda {}".format(args.distilled_loss_lambda), success_log)
            print_log("CE_Loss_lambda {}".format(args.CE_loss_lambda), success_log)
            print_log("=========================", success_log)
            
            
            attack_df['trial seed'] = args.manualSeed
            save_df = attack_df[attack_df['distilled data batch idx'] == distilled_batch_i]

            if csv_save_path is not None:
                csv_file_name = 'ZeBRA_attack_profile_{}.csv'.format(args.manualSeed)
                export_csv = save_df.to_csv(os.path.join(csv_save_path, csv_file_name), index=None)
            return

    fail_save_path = os.path.join(args.save_path, 'fail')
    if not os.path.exists(fail_save_path):
        os.makedirs(fail_save_path)

    fail_log = open(os.path.join(fail_save_path, 'fail_log_{}.txt'.format(args.manualSeed)), 'w')

    print_log('fail save path : {}'.format(fail_save_path), fail_log)
    print_log('==================================', fail_log)
    print_log('fail Zebra attack', fail_log)
    
    if drop_acc_success_trigger:
        print_log("success acc drop but number of bits do not goal\n", fail_log)
        print_log('drop best acc : {}'.format(temp_acc), fail_log)
        print_log('drop best acc dataset idx : {}'.format(local_success_data_idx), fail_log)

        attack_df['trial seed'] = args.manualSeed
        
        save_df = attack_df[attack_df['distilled data batch idx'] == local_success_data_idx]

        if csv_save_path is not None:
            csv_file_name = 'ZeBRA_fail_attack_profile_{}.csv'.format(args.manualSeed)
            export_csv = save_df.to_csv(os.path.join(csv_save_path, csv_file_name), index=None)

    else:
        print_log('drop best acc : {}'.format(temp_acc), fail_log)
        print_log('drop best acc dataset idx : {}'.format(drop_acc_top_idx), fail_log)
        
        
        attack_df['trial seed'] = args.manualSeed

        save_df = attack_df[attack_df['distilled data batch idx'] == drop_acc_top_idx]

        if csv_save_path is not None:
            csv_file_name = 'ZeBRA_fail_attack_profile_{}.csv'.format(args.manualSeed)
            export_csv = save_df.to_csv(os.path.join(csv_save_path, csv_file_name), index=None)
        
    print_log('# of bit flip of drop best acc: {}'.format(temp_hamming_dist), fail_log)
    print_log('achieve bit : {}'.format(achieve_bit), fail_log)
    print_log('minimum flip bit : {}'.format(success_bit), fail_log)
    print_log("attack batch size : \t {}".format(args.attack_sample_size), fail_log)
    print_log("total_loss_bound : {}".format(args.total_loss_bound), fail_log)
    print_log("distilled_loss_lambda {}".format(args.distilled_loss_lambda), fail_log)
    print_log("CE_Loss_lambda {}".format(args.CE_loss_lambda), fail_log)
    print_log("=========================", fail_log)

    return

# train function (forward, backward, update)
def train(train_loader, model, criterion, optimizer, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.use_cuda:
            target = target.cuda()  # the copy will be asynchronous with respect to the host.
            input = input.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)
        if args.clustering:
            loss += clustering_loss(model, args.lambda_coeff)

        # measure accuracy and record loss

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log(
                '  Epoch: [{:03d}][{:03d}/{:03d}]   '
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top5=top5) + time_string(), log)
    print_log(
        '  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'
        .format(top1=top1, top5=top5, error1=100 - top1.avg), log)
    return top1.avg, losses.avg


def validate(val_loader, model, criterion, log, summary_output=False):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    output_summary = [] # init a list for output summary

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

        print_log(
            '  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'
            .format(top1=top1, top5=top5, error1=100 - top1.avg), log)
        
    if summary_output:
        output_summary = np.asarray(output_summary).flatten()
        return top1.avg, top5.avg, losses.avg, output_summary
    else:
        return top1.avg, top5.avg, losses.avg


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def save_checkpoint(state, is_best, save_path, filename, log):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:  # copy the checkpoint to the best model if it is the best_accuracy
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)
        print_log("=> Obtain best accuracy, and update the best model", log)


def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    mu = args.momentum

    if args.optimizer != "YF":
        assert len(gammas) == len(
            schedule), "length of gammas and schedule should be equal"
        for (gamma, step) in zip(gammas, schedule):
            if (epoch >= step):
                lr = lr * gamma
            else:
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    elif args.optimizer == "YF":
        lr = optimizer._lr
        mu = optimizer._mu

    return lr, mu


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


def accuracy_logger(base_dir, epoch, train_accuracy, test_accuracy):
    file_name = 'accuracy.txt'
    file_path = "%s/%s" % (base_dir, file_name)
    # create and format the log file if it does not exists
    if not os.path.exists(file_path):
        create_log = open(file_path, 'w')
        create_log.write('epochs train test\n')
        create_log.close()

    recorder = {}
    recorder['epoch'] = epoch
    recorder['train'] = train_accuracy
    recorder['test'] = test_accuracy
    # append the epoch index, train accuracy and test accuracy:
    with open(file_path, 'a') as accuracy_log:
        accuracy_log.write(
            '{epoch}       {train}    {test}\n'.format(**recorder))


if __name__ == '__main__':
    main()