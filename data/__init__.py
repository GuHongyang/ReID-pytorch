from torch.utils.data import dataloader
from torchvision import transforms
from .dataset import Dataset

from .transforms import *
from .samplers import *
from torchvision.transforms import *
from copy import deepcopy



class Data:
    def __init__(self, args):

        transform = {'train': transforms.Compose([
            transforms.Resize((args.height, args.width), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
        ]),
            'test': transforms.Compose([
                transforms.Resize((args.height, args.width), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'query': transforms.Compose([
                transforms.Resize((args.height, args.width), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])}


        self.transforms=transform

        self.dataset={ name:Dataset(args,transform[name],name) for name in ['train','test','query'] }
        self.dataset['test_query']=deepcopy(self.dataset['test'])
        self.dataset['test_query'].extend(self.dataset['query'])
        self.dataloader={}

        if args.triplet_sapmler:
            self.dataloader['train'] = dataloader.DataLoader(self.dataset['train'],
                                                             sampler=RandomIdentitySampler2(self.dataset['train'],args.batch_id,args.batch_image),
                                                             batch_size=args.batch_id*args.batch_image,
                                                             num_workers=args.nThread)

        else:
            self.dataloader['train']=dataloader.DataLoader(self.dataset['train'],
                                                              shuffle=True,
                                                              batch_size=args.batch_train,
                                                              num_workers=args.nThread)

        self.dataloader['test_query'] = dataloader.DataLoader(self.dataset['test_query'],
                                                              shuffle=False,
                                                              batch_size=args.batch_test,
                                                              num_workers=args.nThread)

        args.num_classes=len(self.dataset['train'].unique_ids)
