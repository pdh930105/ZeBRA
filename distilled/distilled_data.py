from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torchvision
import torch.nn as nn
import torch.optim as optim
import copy
import torch
import random
import argparse
import os

def own_loss(A, B):
    return (A - B).norm()**2 / B.size(0)

class output_hook(object):
    """
	Forward_hook used to get the output of intermediate layer. 
	"""
    def __init__(self):
        super(output_hook, self).__init__()
        self.outputs = None

    def hook(self, module, input, output):
        self.outputs = output

    def clear(self):
        self.outputs = None


class UniformDataset(Dataset):
    """
    get random uniform samples with mean 0 and variance 1
    """
    def __init__(self, length, size, transform):
        self.length = length
        self.transform = transform
        self.size = size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # var[U(-128, 127)] = (127 - (-128))**2 / 12 = 5418.75
        sample = (torch.randint(high=255, size=self.size).float() -
                  127.5) / 5418.75
        return sample

class UniformTargetDataset(Dataset):
    """
    get random uniform sample & random target
    """
    def __init__(self, length, size, num_classes=10, transform=None, dataset="cifar10"):
        self.length=length
        self.transform = transform
        self.size = size
        self.dataset = dataset
        self.num_classes = num_classes

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # var[U(-128, 127)] = (127 - (-128))**2 / 12 = 5418.75
        sample = (torch.randint(high=255, size=self.size).float() -
                  127.5) / 5418.75
        
        if self.dataset == "cifar10":
            #target = idx % 10
            target = int(torch.randint(high=10, size=(1,)))

        
        elif self.dataset =="imagenet":
#            target = idx % 1000
            target = int(torch.randint(high=1000, size=(1,)))
        
        elif self.num_classes >=1:
            target = int(torch.randint(high=1000, size=(1,)))

        else:
            target = 0

        return sample, target


def getRandomData(dataset='cifar10', batch_size=512, transform=None,for_inception=False):
    """
    get random sample dataloader 
    dataset: name of the dataset 
    batch_size: the batch size of random data
    for_inception: whether the data is for Inception because inception has input size 299 rather than 224
    """
    if dataset == 'cifar10':
        size = (3, 32, 32)
        num_data = 10000
    elif dataset == 'imagenet':
        num_data = 10000
        if not for_inception:
            size = (3, 224, 224)
        else:
            size = (3, 299, 299)
    else:
        raise NotImplementedError
    dataset = UniformDataset(length=10000, size=size, transform=transform)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=32)
    return data_loader

def getRandomTargetData(dataset='cifar10', batch_size=512, num_classes=10, transform=None,for_inception=False):
    """
    get random sample dataloader 
    dataset: name of the dataset 
    batch_size: the batch size of random data
    for_inception: whether the data is for Inception because inception has input size 299 rather than 224
    """
    if dataset == 'cifar10':
        size = (3, 32, 32)
        num_data = 10000
    elif dataset == 'imagenet':
        num_data = 10000
        if not for_inception:
            size = (3, 224, 224)
        else:
            size = (3, 299, 299)
    else:
        raise NotImplementedError

    make_dataset = UniformTargetDataset(length=10000, size=size, num_classes=num_classes, transform=transform, dataset=dataset)
    data_loader = DataLoader(make_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=32)
    return data_loader




def getReconData(teacher_model,
                 dataset,
                 batch_size,
                 num_batch=1,
                 for_inception=False,
                 prevent_overfit=5):
    """
	Generate distilled data according to the BatchNorm statistics in pretrained single-precision model.
	Only support single GPU.
	teacher_model: pretrained single-precision model
	dataset: the name of dataset
	batch_size: the batch size of generated distilled data
	num_batch: the number of batch of generated distilled data
	for_inception: whether the data is for Inception because inception has input size 299 rather than 224
	"""
    print("generate distilled data")
    # initialize distilled data with random noise according to the dataset
    dataloader = getRandomData(dataset=dataset,
                               batch_size=batch_size,
                               for_inception=for_inception)

    eps = 1e-6
    # initialize hooks and single-precision model
    hooks, hook_handles, bn_stats, refined_gaussian = [], [], [], []
    teacher_model = teacher_model.cuda()
    teacher_model = teacher_model.eval()

    # get number of BatchNorm layers in the model
    layers = sum([
        1 if isinstance(layer, nn.BatchNorm2d) else 0
        for layer in teacher_model.modules()
    ])

    for n, m in teacher_model.named_modules():
        if isinstance(m, nn.Conv2d) and len(hook_handles) < layers:
            # register hooks on the convolutional layers to get the intermediate output after convolution and before BatchNorm.
            hook = output_hook()
            hooks.append(hook)
            hook_handles.append(m.register_forward_hook(hook.hook))
        if isinstance(m, nn.BatchNorm2d):
            # get the statistics in the BatchNorm layers
            bn_stats.append(
                (m.running_mean.detach().clone().flatten().cuda(),
                 torch.sqrt(m.running_var +
                            eps).detach().clone().flatten().cuda()))
    assert len(hooks) == len(bn_stats)

    for i, gaussian_data in enumerate(dataloader):
        if i == num_batch:
            break
        # initialize the criterion, optimizer, and scheduler
        gaussian_data = gaussian_data.cuda()
        gaussian_data.requires_grad = True
        crit = nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam([gaussian_data], lr=0.1)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         min_lr=1e-4,
                                                         verbose=False,
                                                         patience=100)

        input_mean = torch.zeros(1, 3).cuda()
        input_std = torch.ones(1, 3).cuda()

        for it in range(2000):
            teacher_model.zero_grad()
            optimizer.zero_grad()
            for hook in hooks:
                hook.clear()
            output = teacher_model(gaussian_data)
            mean_loss = 0
            std_loss = 0

            # loss에 cateogorical loss 를 추가해보는 것
            # 해당 아이디어는 아래 링크에서 차용 
            # https://openaccess.thecvf.com/content_CVPRW_2020/papers/w40/Choi_Data-Free_Network_Quantization_With_Adversarial_Knowledge_Distillation_CVPRW_2020_paper.pdf

            # compute the loss according to the BatchNorm statistics and the statistics of intermediate output
            for cnt, (bn_stat, hook) in enumerate(zip(bn_stats, hooks)):
                tmp_output = hook.outputs
                bn_mean, bn_std = bn_stat[0], bn_stat[1]
                tmp_mean = torch.mean(tmp_output.view(tmp_output.size(0),
                                                      tmp_output.size(1), -1),
                                      dim=2)

                tmp_std = torch.sqrt(
                    torch.var(tmp_output.view(tmp_output.size(0),
                                              tmp_output.size(1), -1),
                              dim=2) + eps)

                mean_loss += own_loss(bn_mean, tmp_mean)
                std_loss += own_loss(bn_std, tmp_std)
            tmp_mean = torch.mean(gaussian_data.view(gaussian_data.size(0), 3,-1), dim=2)
            tmp_std = torch.sqrt(torch.var(gaussian_data.view(gaussian_data.size(0), 3, -1), dim=2) + eps)
            mean_loss += own_loss(tmp_mean, input_mean)
            std_loss += own_loss(tmp_std, input_std)
            total_loss = mean_loss + std_loss
            # update the distilled data
            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss.item())

            if it % 200 == 0:
                print("epoch : {} total loss : {}".format(it, total_loss))

            # early stop to prevent overfit
            if total_loss <= (layers + 1) * prevent_overfit:
                print("end")
                break

        refined_gaussian.append(gaussian_data.detach().clone())

    for handle in hook_handles:
        handle.remove()
    return refined_gaussian


def getReconTargetData(teacher_model,
                 dataset,
                 batch_size,
                 num_batch=1,
                 num_classes=10,
                 for_inception=False,
                 target_classes = None,
                 CE_loss_lambda = 0.1,
                 distilled_loss_lambda = 0.1,
                 total_loss_bound=10):
    """
	Generate distilled data according to the BatchNorm statistics in pretrained single-precision model.
	Only support single GPU.
	teacher_model: pretrained single-precision model
	dataset: the name of dataset
	batch_size: the batch size of generated distilled data
	num_batch: the number of batch of generated distilled data
	for_inception: whether the data is for Inception because inception has input size 299 rather than 224
	"""

    print("generate distilled_target_data generate")
    # initialize distilled data with random noise according to the dataset
    dataloader = getRandomTargetData(dataset=dataset,
                               batch_size=batch_size,
                               for_inception=for_inception)

    eps = 1e-6
    # initialize hooks and single-precision model
    hooks, hook_handles, bn_stats, refined_gaussian, refined_target = [], [], [], [], []
    teacher_model = teacher_model.cuda()
    teacher_model = teacher_model.eval()

    # get number of BatchNorm layers in the model
    layers = sum([
        1 if isinstance(layer, nn.BatchNorm2d) else 0
        for layer in teacher_model.modules()
    ])

    for n, m in teacher_model.named_modules():
        if isinstance(m, nn.Conv2d) and len(hook_handles) < layers:
            # register hooks on the convolutional layers to get the intermediate output after convolution and before BatchNorm.
            hook = output_hook()
            hooks.append(hook)
            hook_handles.append(m.register_forward_hook(hook.hook))
        if isinstance(m, nn.BatchNorm2d):
            # get the statistics in the BatchNorm layers
            bn_stats.append(
                (m.running_mean.detach().clone().flatten().cuda(),
                 torch.sqrt(m.running_var +
                            eps).detach().clone().flatten().cuda()))
    assert len(hooks) == len(bn_stats)

    for i, (gaussian_data, target) in enumerate(dataloader):
        # initialize the criterion, optimizer, and scheduler
        gaussian_data = gaussian_data.cuda()
        gaussian_data.requires_grad = True
        
        # initiallize specified target
        if target_classes is not None:
            target = torch.ones_like(target) * target_classes

        target = target.cuda()
        
        crit = nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam([gaussian_data], lr=0.1)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         min_lr=1e-4,
                                                         verbose=False,
                                                         patience=100)
        input_mean = torch.zeros(1, 3).cuda()
        input_std = torch.ones(1, 3).cuda()

        for it in range(2000):
            teacher_model.zero_grad()
            optimizer.zero_grad()
            for hook in hooks:
                hook.clear()
            output = teacher_model(gaussian_data)
            mean_loss = 0
            std_loss = 0

            # loss에 cateogorical loss 를 추가해보는 것
            # 해당 아이디어는 아래 링크에서 차용 
            # https://openaccess.thecvf.com/content_CVPRW_2020/papers/w40/Choi_Data-Free_Network_Quantization_With_Adversarial_Knowledge_Distillation_CVPRW_2020_paper.pdf

            # compute the loss according to the BatchNorm statistics and the statistics of intermediate output
            for cnt, (bn_stat, hook) in enumerate(zip(bn_stats, hooks)):
                tmp_output = hook.outputs
                bn_mean, bn_std = bn_stat[0], bn_stat[1]
                tmp_mean = torch.mean(tmp_output.view(tmp_output.size(0),
                                                      tmp_output.size(1), -1),
                                      dim=2)

                tmp_std = torch.sqrt(
                    torch.var(tmp_output.view(tmp_output.size(0),
                                              tmp_output.size(1), -1),
                              dim=2) + eps)

                mean_loss += own_loss(bn_mean, tmp_mean)
                std_loss += own_loss(bn_std, tmp_std)
            tmp_mean = torch.mean(gaussian_data.view(gaussian_data.size(0), 3,-1), dim=2)
            tmp_std = torch.sqrt(torch.var(gaussian_data.view(gaussian_data.size(0), 3, -1), dim=2) + eps)
            mean_loss += own_loss(tmp_mean, input_mean)
            std_loss += own_loss(tmp_std, input_std)
            
            distilled_loss = mean_loss + std_loss
            CEloss = crit(output, target)
            total_loss = CE_loss_lambda*CEloss + distilled_loss_lambda * distilled_loss

            if it % 200 == 0:
                print("CE_loss : {}\t disilled_loss : {}\t".format(CEloss, distilled_loss))
                print("total loss : {}".format(total_loss))
            
            
            # update the distilled data
            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss.item())

            # early stop to prevent overfit
            if total_loss <= total_loss_bound:
                break
        
        print("=======================================")
        print("Success")
        print("CE_loss : {}\t disilled_loss : {}\t".format(CEloss, distilled_loss))
        print("total loss : {}".format(total_loss))
                    
        refined_gaussian.append(gaussian_data.detach().clone())
        refined_target.append(target.detach().clone())

        if i == num_batch:
            break
        
    for handle in hook_handles:
        handle.remove()
    
    return refined_gaussian, refined_target

class distilled_dataset(torch.utils.data.Dataset):
    def __init__(self, teacher_model, dataset="cifar10", batch_size=64, num_batch=1, for_inception=False, 
    target_batch=0, target_classes=0, prevent_overfit=5):
        self.distilled_images = getReconData(teacher_model, dataset, batch_size, num_batch, for_inception, prevent_overfit)
        self.target_classes = target_classes
        self.target_batch = target_batch
        self.x_data = self.distilled_images[self.target_batch].cpu().detach()
        self.y_data = torch.ones(self.x_data.shape[0]) * self.target_classes
        if dataset == "cifar10":
            self.classes = ['airplane',
                'automobile',
                'bird',
                'cat',
                'deer',
                'dog',
                'frog',
                'horse',
                'ship',
                'truck'
                ]
        else: # imagenet
            self.classes = []
    def reselect_batch(self, target_batch):
        self.target_batch = target_batch
        self.x_data = self.distilled_images[self.target_batch].cpu().detach()


    def reselect_target(self, target_classes):
        self.target_classes = target_classes
        self.y_data = torch.ones(self.x_data.shape[0]) * self.target_classes

    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]

        return x, y

class distilled_target_dataset(torch.utils.data.Dataset):
    def __init__(self, teacher_model, dataset="cifar10", batch_size=64, num_batch=1, num_classes=10, for_inception=False, target_classes = None, CE_loss_lambda=0.1, distilled_loss_lambda=1,total_loss_bound=1):
        self.teacher_model = teacher_model
        self.distilled_images, self.distilled_target = getReconTargetData(teacher_model=teacher_model, dataset=dataset, batch_size=batch_size,num_batch=num_batch, 
                                                    num_classes=num_classes,for_inception=for_inception, target_classes=target_classes, 
                                                    CE_loss_lambda=CE_loss_lambda, distilled_loss_lambda=distilled_loss_lambda,total_loss_bound=total_loss_bound)
        self.x_data = torch.cat(self.distilled_images, dim=0).cpu().detach()
        self.y_data = torch.cat(self.distilled_target, dim=0).cpu().detach()
        if dataset == "cifar10":
            self.classes = ['airplane',
                'automobile',
                'bird',
                'cat',
                'deer',
                'dog',
                'frog',
                'horse',
                'ship',
                'truck'
                ]
        else: # imagenet
            self.classes = []

    def reselect_distilled_data(self, teacher_model=None, dataset="cifar10", batch_size=64, num_batch=1, for_inception=False, target_classes=None, target_batch=0, CE_loss_lambda=0.1, distilled_loss_lambda=1,total_loss_min=1):
        if teacher_model == None:
            teacher_model =  self.teacher_model
        
        self.distilled_images, self.distilled_target = getReconTargetData(teacher_model, dataset, batch_size, num_batch, for_inception, target_classes,CE_loss_lambda, distilled_loss_lambda, total_loss_min)
        
        self.x_data = torch.cat(self.distilled_images, dim=0)
        self.y_data = torch.cat(self.distilled_target, dim=0)

    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]

        return x, y
