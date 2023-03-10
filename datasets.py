import os
import random
import json
from scipy.io import loadmat
from PIL import Image
import xml.etree.ElementTree as ET
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import random_split, ConcatDataset, Subset

from transforms import MultiView, RandomResizedCrop, ColorJitter, GaussianBlur, RandomRotation, KRandomResizedCrop
from torchvision import transforms as T
from torchvision.datasets import STL10, CIFAR10, CIFAR100, ImageFolder, ImageNet, Caltech101, Caltech256
from datasets_newer_torch import Flowers102, Food101, DTD, OxfordIIITPet, StanfordCars, FGVCAircraft

import kornia.augmentation as K

class ImageList(torch.utils.data.Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.samples)

class ImageNet100(ImageFolder):
    def __init__(self, root, split, transform):
        with open('splits/imagenet100.txt') as f:
            classes = [line.strip() for line in f]
            class_to_idx = { cls: idx for idx, cls in enumerate(classes) }

        super().__init__(os.path.join(root, split), transform=transform)
        samples = []
        for path, label in self.samples:
            cls = self.classes[label]
            if cls not in class_to_idx:
                continue
            label = class_to_idx[cls]
            samples.append((path, label))

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.targets = [s[1] for s in samples]

class Pets(ImageList):
    def __init__(self, root, split, transform=None):
        with open(os.path.join(root, 'annotations', f'{split}.txt')) as f:
            annotations = [line.split() for line in f]

        samples = []
        for sample in annotations:
            path = os.path.join(root, 'images', sample[0] + '.jpg')
            label = int(sample[1])-1
            samples.append((path, label))

        super().__init__(samples, transform)

# class Food101(ImageList):
#     def __init__(self, root, split, transform=None):
#         with open(os.path.join(root, 'meta', 'classes.txt')) as f:
#             classes = [line.strip() for line in f]
#         with open(os.path.join(root, 'meta', f'{split}.json')) as f:
#             annotations = json.load(f)

#         samples = []
#         for i, cls in enumerate(classes):
#             for path in annotations[cls]:
#                 samples.append((os.path.join(root, 'images', f'{path}.jpg'), i))

#         super().__init__(samples, transform)

# class DTD(ImageList):
#     def __init__(self, root, split, transform=None):
#         with open(os.path.join(root, 'labels', f'{split}1.txt')) as f:
#             paths = [line.strip() for line in f]

#         classes = sorted(os.listdir(os.path.join(root, 'images')))
#         samples = [(os.path.join(root, 'images', path), classes.index(path.split('/')[0])) for path in paths]
#         super().__init__(samples, transform)

class SUN397(ImageList):
    def __init__(self, root, split, transform=None):
        # some files exists only in /storage/shared/datasets/mimit67_indoor_scenes/indoorCVPR_09/images_train_test/Images/
        root = os.path.join(root, "sun397")
        with open(os.path.join(root, 'ClassName.txt')) as f:
            classes = [line.strip() for line in f]

        with open(os.path.join(root, f'{split}_01.txt')) as f:
            samples = []
            for line in f:
                path = line.strip()
                for y, cls in enumerate(classes):
                    if path.startswith(cls+'/'):
                        samples.append((os.path.join(root, path[1:]), y))
                        break
        super().__init__(samples, transform)

def load_pretrain_datasets(dataset='cifar10',
                           datadir='/data',
                           color_aug='default'):

    if dataset == 'imagenet100':
        mean = torch.tensor([0.485, 0.456, 0.406])
        std  = torch.tensor([0.229, 0.224, 0.225])
        train_transform = MultiView(RandomResizedCrop(224, scale=(0.2, 1.0)))
        test_transform = T.Compose([T.Resize(224),
                                    T.CenterCrop(224),
                                    T.ToTensor(),
                                    T.Normalize(mean, std)])
        t1 = nn.Sequential(K.RandomHorizontalFlip(),
                           ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                           K.RandomGrayscale(p=0.2),
                           GaussianBlur(23, (0.1, 2.0)),
                           K.Normalize(mean, std))
        t2 = nn.Sequential(K.RandomHorizontalFlip(),
                           ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                           K.RandomGrayscale(p=0.2),
                           GaussianBlur(23, (0.1, 2.0)),
                           K.Normalize(mean, std))

        trainset = ImageNet100(datadir, split='train', transform=train_transform)
        valset   = ImageNet100(datadir, split='train', transform=test_transform)
        testset  = ImageNet100(datadir, split='val', transform=test_transform)

    elif dataset == 'stl10':
        mean = torch.tensor([0.43, 0.42, 0.39])
        std  = torch.tensor([0.27, 0.26, 0.27])
        train_transform = MultiView(RandomResizedCrop(96, scale=(0.2, 1.0)))

        if color_aug == 'default':
            s = 1
        elif color_aug == 'strong':
            s = 2.
        elif color_aug == 'weak':
            s = 0.5
        test_transform = T.Compose([T.Resize(96),
                                    T.CenterCrop(96),
                                    T.ToTensor(),
                                    T.Normalize(mean, std)])
        t1 = nn.Sequential(K.RandomHorizontalFlip(),
                           ColorJitter(0.4*s, 0.4*s, 0.4*s, 0.1*s, p=0.8),
                           K.RandomGrayscale(p=0.2*s),
                           GaussianBlur(9, (0.1, 2.0)),
                           K.Normalize(mean, std))
        t2 = nn.Sequential(K.RandomHorizontalFlip(),
                           ColorJitter(0.4*s, 0.4*s, 0.4*s, 0.1*s, p=0.8),
                           K.RandomGrayscale(p=0.2*s),
                           GaussianBlur(9, (0.1, 2.0)),
                           K.Normalize(mean, std))

        trainset = STL10(datadir, split='train+unlabeled', transform=train_transform, download=True)
        valset   = STL10(datadir, split='train',           transform=test_transform, download=True)
        testset  = STL10(datadir, split='test',            transform=test_transform, download=True)

    elif dataset == 'stl10_rot':
        mean = torch.tensor([0.43, 0.42, 0.39])
        std  = torch.tensor([0.27, 0.26, 0.27])
        train_transform = MultiView(RandomResizedCrop(96, scale=(0.2, 1.0)))
        test_transform = T.Compose([T.Resize(96),
                                    T.CenterCrop(96),
                                    T.ToTensor(),
                                    T.Normalize(mean, std)])
        t1 = nn.Sequential(K.RandomHorizontalFlip(),
                           ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                           K.RandomGrayscale(p=0.2),
                           GaussianBlur(9, (0.1, 2.0)),
                           RandomRotation(p=0.5),
                           K.Normalize(mean, std))
        t2 = nn.Sequential(K.RandomHorizontalFlip(),
                           ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                           K.RandomGrayscale(p=0.2),
                           GaussianBlur(9, (0.1, 2.0)),
                           RandomRotation(p=0.5),
                           K.Normalize(mean, std))

        trainset = STL10(datadir, split='train+unlabeled', transform=train_transform)
        valset   = STL10(datadir, split='train',           transform=test_transform)
        testset  = STL10(datadir, split='test',            transform=test_transform)

    elif dataset == 'stl10_sol':
        mean = torch.tensor([0.43, 0.42, 0.39])
        std  = torch.tensor([0.27, 0.26, 0.27])
        train_transform = MultiView(RandomResizedCrop(96, scale=(0.2, 1.0)))

        test_transform = T.Compose([T.Resize(96),
                                    T.CenterCrop(96),
                                    T.ToTensor(),
                                    T.Normalize(mean, std)])
        t1 = nn.Sequential(K.RandomHorizontalFlip(),
                           ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                           K.RandomSolarize(0.5, 0.0, p=0.5),
                           K.RandomGrayscale(p=0.2),
                           GaussianBlur(9, (0.1, 2.0)),
                           K.Normalize(mean, std))
        t2 = nn.Sequential(K.RandomHorizontalFlip(),
                           ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                           K.RandomSolarize(0.5, 0.0, p=0.5),
                           K.RandomGrayscale(p=0.2),
                           GaussianBlur(9, (0.1, 2.0)),
                           K.Normalize(mean, std))

        trainset = STL10(datadir, split='train+unlabeled', transform=train_transform, download=True)
        valset   = STL10(datadir, split='train',           transform=test_transform, download=True)
        testset  = STL10(datadir, split='test',            transform=test_transform, download=True)

    else:
        raise Exception(f'Unknown dataset {dataset}')

    return dict(train=trainset,
                val=valset,
                test=testset,
                t1=t1, t2=t2)

def load_datasets(dataset='cifar10',
                  datadir='/data',
                  pretrain_data='stl10'):

    if pretrain_data == 'imagenet100':
        mean = torch.tensor([0.485, 0.456, 0.406])
        std  = torch.tensor([0.229, 0.224, 0.225])
        transform = T.Compose([T.Resize(224, interpolation=Image.BICUBIC),
                               T.CenterCrop(224),
                               T.ToTensor(),
                               T.Normalize(mean, std)])

    elif pretrain_data == 'stl10':
        mean = torch.tensor([0.43, 0.42, 0.39])
        std  = torch.tensor([0.27, 0.26, 0.27])
        transform = T.Compose([T.Resize(96, interpolation=Image.BICUBIC),
                               T.CenterCrop(96),
                               T.ToTensor(),
                               T.Normalize(mean, std)])

    generator = lambda seed: torch.Generator().manual_seed(seed)
    if dataset == 'imagenet100':
        """
        https://github.com/HobbitLong/CMC/blob/master/imagenet100.txt
        """
        trainval = ImageNet100(datadir, split='train', transform=transform)
        train, val = None, None
        test     = ImageNet100(datadir, split='val', transform=transform)
        num_classes = 100

    elif dataset == 'food101':
        trainval   = Food101(root=datadir, split='train', transform=transform, download=True)
        train, val = random_split(trainval, [68175, 7575], generator=generator(42))
        test       = Food101(root=datadir, split='test',  transform=transform, download=True)
        num_classes = 101

    elif dataset == 'cifar10':
        trainval   = CIFAR10(root=datadir, train=True,  transform=transform, download=True)
        train, val = random_split(trainval, [45000, 5000], generator=generator(43))
        test       = CIFAR10(root=datadir, train=False, transform=transform, download=True)
        num_classes = 10

    elif dataset == 'cifar100':
        trainval   = CIFAR100(root=datadir, train=True,  transform=transform, download=True)
        train, val = random_split(trainval, [45000, 5000], generator=generator(44))
        test       = CIFAR100(root=datadir, train=False, transform=transform, download=True)
        num_classes = 100

    elif dataset == 'sun397':
        trn_indices, val_indices = torch.load('splits/sun397.pth')
        trainval = SUN397(root=datadir, split='Training', transform=transform)
        train    = Subset(trainval, trn_indices)
        val      = Subset(trainval, val_indices)
        test     = SUN397(root=datadir, split='Testing',  transform=transform)
        num_classes = 397

    elif dataset == 'dtd':
        train    = DTD(root=datadir, split='train', transform=transform, download=True)
        val      = DTD(root=datadir, split='val',   transform=transform, download=True)
        trainval = ConcatDataset([train, val])
        test     = DTD(root=datadir, split='test',  transform=transform, download=True)
        num_classes = 47

    elif dataset == 'pets':
        trainval   = OxfordIIITPet(root=datadir, split='trainval', transform=transform, download=True)
        train, val = random_split(trainval, [2940, 740], generator=generator(49))
        test       = OxfordIIITPet(root=datadir, split='test',     transform=transform, download=True)
        num_classes = 37

    elif dataset == 'caltech_101':
        transform.transforms.insert(0, T.Lambda(lambda img: img.convert('RGB')))
        D = Caltech101(datadir, transform=transform, download=False) # changed flag download
        trn_indices, val_indices, tst_indices = torch.load('splits/caltech101.pth')
        train    = Subset(D, trn_indices)
        val      = Subset(D, val_indices)
        trainval = ConcatDataset([train, val])
        test     = Subset(D, tst_indices)
        num_classes = 101

    elif dataset == 'flowers':
        train = Flowers102(datadir, split="train", transform=transform, download=True)
        val = Flowers102(datadir, split="val", transform=transform, download=True)
        test = Flowers102(datadir, split="test", transform=transform, download=True)

        trainval = ConcatDataset([train, val])
        num_classes = 102

    elif dataset in ['flowers-5shot', 'flowers-10shot']:
        if dataset == 'flowers-5shot':
            n = 5
        else:
            n = 10
        train = Flowers102(datadir, split="train", transform=transform, download=True)
        val = Flowers102(datadir, split="val", transform=transform, download=True)
        trainval = Flowers102(datadir, split="train", transform=transform, download=True)
        trainval._image_files += val._image_files
        trainval._labels += val._labels

        test = Flowers102(datadir, split="test", transform=transform, download=True)

        trainval = ConcatDataset([train, val])

        indices = defaultdict(list)
        for i, y in enumerate(trainval._labels):
            indices[y].append(i)
        indices = sum([random.sample(indices[y], n) for y in indices.keys()], [])
        trainval = Subset(trainval, indices)
        # test     = ImageFolder(os.path.join(datadir, 'tst'), transform=transform)
        num_classes = 102

    elif dataset == 'stl10':
        trainval   = STL10(root=datadir, split='train', transform=transform, download=True)
        test       = STL10(root=datadir, split='test',  transform=transform, download=True)
        train, val = random_split(trainval, [4500, 500], generator=generator(50))
        num_classes = 10


    elif dataset == 'mit67':
        """
        https://www.kaggle.com/datasets/itsahmad/indoor-scenes-cvpr-2019
        """
        trainval = ImageFolder(os.path.join(datadir, 'train'), transform=transform)
        test     = ImageFolder(os.path.join(datadir, 'test'),  transform=transform)
        train, val = random_split(trainval, [4690, 670], generator=generator(51))
        num_classes = 67

    elif dataset == 'cub200':
        trn_indices, val_indices = torch.load('splits/cub200.pth')
        trainval = ImageFolder(os.path.join(datadir, 'train'), transform=transform)
        train    = Subset(trainval, trn_indices)
        val      = Subset(trainval, val_indices)
        test     = ImageFolder(os.path.join(datadir, 'test'),  transform=transform)
        num_classes = 200

    elif dataset == 'cars':
        trainval = StanfordCars(datadir, "train", transform=transform, download=True)
        test =  StanfordCars(datadir, "test", transform=transform, download=True)
        train, val = random_split(trainval, [7000, 1144], generator=generator(51))
        num_classes = 196

    elif dataset == 'aircraft':
        trainval = FGVCAircraft(datadir, "trainval", transform=transform, download=True)
        train = FGVCAircraft(datadir, "train", transform=transform, download=True)
        val = FGVCAircraft(datadir, "val", transform=transform, download=True)

        test =  FGVCAircraft(datadir, "test", transform=transform, download=True)
        num_classes = 100

    elif dataset == 'dog':
        # not mentioned in the paper?
        trn_indices, val_indices = torch.load('splits/dog.pth')
        trainval = ImageFolder(os.path.join(datadir, 'train'), transform=transform)
        train    = Subset(trainval, trn_indices)
        val      = Subset(trainval, val_indices)
        test     = ImageFolder(os.path.join(datadir, 'test'),  transform=transform)
        num_classes = 120

    return dict(trainval=trainval,
                train=train,
                val=val,
                test=test,
                num_classes=num_classes)


def load_fewshot_datasets(dataset='cifar10',
                          datadir='/data',
                          pretrain_data='stl10'):

    if pretrain_data == 'imagenet100':
        mean = torch.tensor([0.485, 0.456, 0.406])
        std  = torch.tensor([0.229, 0.224, 0.225])
        transform = T.Compose([T.Resize(224, interpolation=Image.BICUBIC),
                               T.CenterCrop(224),
                               T.ToTensor(),
                               T.Normalize(mean, std)])

    elif pretrain_data == 'stl10':
        mean = torch.tensor([0.43, 0.42, 0.39])
        std  = torch.tensor([0.27, 0.26, 0.27])
        transform = T.Compose([T.Resize(96, interpolation=Image.BICUBIC),
                               T.CenterCrop(96),
                               T.ToTensor(),
                               T.Normalize(mean, std)])

    if dataset == 'cub200':
        train = ImageFolder(os.path.join(datadir, 'train'), transform=transform)
        test  = ImageFolder(os.path.join(datadir, 'test'),  transform=transform)
        test.samples = train.samples + test.samples
        test.targets = train.targets + test.targets

    elif dataset == 'fc100':
        train = ImageFolder(os.path.join(datadir, 'train'), transform=transform)
        test  = ImageFolder(os.path.join(datadir, 'test'),  transform=transform)

    elif dataset == 'plant_disease':
        test = ImageFolder(os.path.join(datadir, 'train'), transform=transform)
        # test  = ImageFolder(os.path.join(datadir, 'test'),  transform=transform)
        # test.samples = train.samples + test.samples
        # test.targets = train.targets + test.targets

    else:
        raise NotImplementedError(dataset)

    return dict(test=test)

def load_datasets_for_cosine_sim(
        dataset='cifar10',
        datadir='/data',
        pretrain_data="stl10",
        color_aug='default'):

    if pretrain_data == 'imagenet100':
        mean = torch.tensor([0.485, 0.456, 0.406])
        std  = torch.tensor([0.229, 0.224, 0.225])
        test_transform = T.Compose([
                T.Resize(224),
                T.CenterCrop(224,),
                T.ToTensor(),
                                    ])
        transforms = dict(
            flip=K.RandomHorizontalFlip(p=1),
            color=ColorJitter(0.4, 0.4, 0.4, 0.1, p=1),
            grayscale=K.RandomGrayscale(p=1),
            blur= GaussianBlur(23, (0.1, 2.0), p=1),
            # crop=RandomResizedCrop(224, scale=(0.2, 1.0)),
            identity=T.Compose([]),
            # normalize=T.Normalize(mean, std), #TODO bug?
        )

        transforms = {
            k: T.Compose([v, T.Normalize(mean, std)])
            for (k,v) in transforms.items()
        }

        # testset  = ImageNet100(datadir, split='val', transform=test_transform)

    elif pretrain_data == 'stl10':
        mean = torch.tensor([0.43, 0.42, 0.39])
        std  = torch.tensor([0.27, 0.26, 0.27])

        s = 1

        test_transform = T.Compose([
            T.Resize(96),
            T.CenterCrop(96),
            T.ToTensor(),
                                    ])

        transforms = dict(
            flip=K.RandomHorizontalFlip(p=1),
            color=ColorJitter(0.4, 0.4, 0.4, 0.1, p=1),
            grayscale=K.RandomGrayscale(p=1),
            blur=GaussianBlur(9, (0.1, 2.0), p=1),
            crop=K.RandomResizedCrop((96, 96), scale=(0.2, 1.0)),
            normalize=T.Normalize(mean, std),
        )
    else:
        raise NotImplementedError(pretrain_data)
        # testset  = STL10(datadir, split='test',            transform=test_transform, download=True)
    if dataset == 'imagenet100':
        """
        https://github.com/HobbitLong/CMC/blob/master/imagenet100.txt
        """
        test     = ImageNet100(datadir, split='val', transform=test_transform)

    elif dataset == 'food101':
        test       = Food101(root=datadir, split='test',  transform=test_transform, download=True)

    elif dataset == 'cifar10':
        test       = CIFAR10(root=datadir, train=False, transform=test_transform, download=True)

    elif dataset == 'cifar100':
        test       = CIFAR100(root=datadir, train=False, transform=test_transform, download=True)

    elif dataset == 'sun397':
        test     = SUN397(root=datadir, split='Testing',  transform=test_transform)

    elif dataset == 'dtd':
        test     = DTD(root=datadir, split='test',  transform=test_transform, download=True)

    elif dataset == 'pets':
        test       = OxfordIIITPet(root=datadir, split='test',     transform=test_transform, download=True)

    elif dataset == 'caltech101':
        test_transform.transforms.insert(0, T.Lambda(lambda img: img.convert('RGB')))
        D = Caltech101(datadir, transform=test_transform, download=True)
        trn_indices, val_indices, tst_indices = torch.load('splits/caltech101.pth')
        test     = Subset(D, tst_indices)

    elif dataset == 'flowers':
        test = Flowers102(datadir, split="test", transform=test_transform, download=True)

    elif dataset == 'stl10':
        test = STL10(root=datadir, split='test',  transform=test_transform, download=True)

    elif dataset == 'mit67':
        """
        https://www.kaggle.com/datasets/itsahmad/indoor-scenes-cvpr-2019
        """
        test     = ImageFolder(os.path.join(datadir, 'test'),  transform=test_transform)

    elif dataset == 'cub200':
        test     = ImageFolder(os.path.join(datadir, 'test'),  transform=test_transform)

    elif dataset == 'cars':
        test =  StanfordCars(datadir, "test", transform=test_transform, download=True)

    elif dataset == 'aircraft':
        test =  FGVCAircraft(datadir, "test", transform=test_transform, download=True)

    else:
        raise Exception(f'Unknown dataset {dataset}')

    return dict(
        test=test,
        transforms=transforms
    )

def load_datasets_for_augm_interpolation(
        dataset='cifar10',
        datadir='/data',
        pretrain_data="stl10",
        augmentation='colorjitter'):

    if pretrain_data == 'imagenet100':
        mean = torch.tensor([0.485, 0.456, 0.406])
        std  = torch.tensor([0.229, 0.224, 0.225])
        test_transform = T.Compose([
                T.Resize(224),
                T.CenterCrop(224,),
                T.ToTensor(),
                                    ])
        standard_transforms = dict(
            #flip=K.RandomHorizontalFlip(p=1),
            #color=ColorJitter(0.4, 0.4, 0.4, 0.1, p=1),
            #grayscale=K.RandomGrayscale(p=1),
            blur= GaussianBlur(23, [0.1, 2.0], p=1),
            identity=T.Compose([]),
        )

        transforms = dict()

        if augmentation in ["colorjitter", "blur"]:
            for (k,v) in standard_transforms.items():
                modified = v
                if k == "color":
                    for i in range(1,9):
                        modified.brightness = (i/8) * v.brightness
                        modified.contrast = (i/8) * v.contrast
                        modified.saturation = (i/8) * v.saturation
                        modified.hue = (i/8) * v.hue
                        transforms[f'color_{i}'] = T.Compose([modified, T.Normalize(mean, std)])
                elif k == "blur":
                    for i in range(1,9):
                        #modified.sigma = (i/8) * v.sigma
                        for j in range(len(modified.sigma)):
                            modified.sigma[j] = (i/8) * v.sigma[j]
                        transforms[f'blur_{i}'] = T.Compose([modified, T.Normalize(mean, std)])
                elif k=="identity":
                    transforms[k] = v
        # else TODO

        #transforms = {
        #    k: T.Compose([v, T.Normalize(mean, std)])
        #    for (k,v) in transforms.items()
        #}

        # testset  = ImageNet100(datadir, split='val', transform=test_transform)

    elif pretrain_data == 'stl10':
        mean = torch.tensor([0.43, 0.42, 0.39])
        std  = torch.tensor([0.27, 0.26, 0.27])

        s = 1

        test_transform = T.Compose([
            T.Resize(96),
            T.CenterCrop(96),
            T.ToTensor(),
                                    ])

        transforms = dict(
            flip=K.RandomHorizontalFlip(p=1),
            color=ColorJitter(0.4, 0.4, 0.4, 0.1, p=1),
            grayscale=K.RandomGrayscale(p=1),
            blur=GaussianBlur(9, (0.1, 2.0), p=1),
            crop=K.RandomResizedCrop((96, 96), scale=(0.2, 1.0)),
            normalize=T.Normalize(mean, std),
        )
    else:
        raise NotImplementedError(pretrain_data)
        # testset  = STL10(datadir, split='test',            transform=test_transform, download=True)
    if dataset == 'imagenet100':
        """
        https://github.com/HobbitLong/CMC/blob/master/imagenet100.txt
        """
        test     = ImageNet100(datadir, split='val', transform=test_transform)

    elif dataset == 'food101':
        test       = Food101(root=datadir, split='test',  transform=test_transform, download=True)

    elif dataset == 'cifar10':
        test       = CIFAR10(root=datadir, train=False, transform=test_transform, download=True)

    elif dataset == 'cifar100':
        test       = CIFAR100(root=datadir, train=False, transform=test_transform, download=True)

    elif dataset == 'sun397':
        test     = SUN397(root=datadir, split='Testing',  transform=test_transform)

    elif dataset == 'dtd':
        test     = DTD(root=datadir, split='test',  transform=test_transform, download=True)

    elif dataset == 'pets':
        test       = OxfordIIITPet(root=datadir, split='test',     transform=test_transform, download=True)

    elif dataset == 'caltech101':
        test_transform.transforms.insert(0, T.Lambda(lambda img: img.convert('RGB')))
        D = Caltech101(datadir, transform=test_transform, download=True)
        trn_indices, val_indices, tst_indices = torch.load('splits/caltech101.pth')
        test     = Subset(D, tst_indices)

    elif dataset == 'flowers':
        test = Flowers102(datadir, split="test", transform=test_transform, download=True)

    elif dataset == 'stl10':
        test = STL10(root=datadir, split='test',  transform=test_transform, download=True)

    elif dataset == 'mit67':
        """
        https://www.kaggle.com/datasets/itsahmad/indoor-scenes-cvpr-2019
        """
        test     = ImageFolder(os.path.join(datadir, 'test'),  transform=test_transform)

    elif dataset == 'cub200':
        test     = ImageFolder(os.path.join(datadir, 'test'),  transform=test_transform)

    elif dataset == 'cars':
        test =  StanfordCars(datadir, "test", transform=test_transform, download=True)

    elif dataset == 'aircraft':
        test =  FGVCAircraft(datadir, "test", transform=test_transform, download=True)

    else:
        raise Exception(f'Unknown dataset {dataset}')

    return dict(
        test=test,
        transforms=transforms
    )
