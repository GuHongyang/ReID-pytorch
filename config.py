import argparse
import os

parser = argparse.ArgumentParser(description='Person ReID Frame')

"""
System parameters
"""
parser.add_argument('--nThread', type=int, default=4, help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true', help='use cpu only',default=False)
parser.add_argument('--GPUs', type=str, default='3', help='the number of GPUS')

"""
Data parameters
"""
#DukeMTMC-reID   cuhk03-np/labeled  /detected
parser.add_argument("--data_dir", type=str, default="/home/guhongyang/DATASETS/Market-1501-v15.09.15", help='dataset directory')
parser.add_argument("--batch_train", type=int, default=64, help='input batch size for train')
parser.add_argument("--batch_test", type=int, default=128, help='input batch size for test')
parser.add_argument('--height', type=int, default=256, help='height of the input image')
parser.add_argument('--width', type=int, default=128, help='width of the input image')
parser.add_argument('--batch_id',type=int,default=16)
parser.add_argument('--batch_image',type=int,default=4)
parser.add_argument('--triplet_sapmler',type=bool,default=False)

"""
Model parameters
"""
parser.add_argument('--model',type=str,default='ResNet_50')
parser.add_argument('--load_dir',type=str,default='')


"""
Loss parameters
"""
#weight*name+weight*name
#CE_Loss,Triplet_Loss-1.5,Center_Loss
parser.add_argument('--loss',type=str,default='1*CE_Loss')


"""
Optimizer parameters
"""
parser.add_argument('--optimizer',type=str,default='Adam',choices=['SGD','Adam'])
parser.add_argument('--lrs',type=str,default='3.5e-4')
parser.add_argument('--momentum',type=float,default=0.9)
parser.add_argument('--dampening',type=float,default=0)
parser.add_argument('--weight_decay',type=float,default=5e-4)
parser.add_argument('--nesterov',type=bool,default=True)


"""
Train parameters
"""
parser.add_argument('--epochs',type=int,default=100)
parser.add_argument('--test_every',type=int,default=40)
parser.add_argument('--milestones',type=str,default='40-70')
parser.add_argument('--gamma',type=float,default=0.1)
parser.add_argument('--warm_training',type=int,default=10,help='warm up epochs, 0 for no warm up')
parser.add_argument('--warm_lr',type=float,default=3.5e-6,help='warm up start lr')



args = parser.parse_args()
args.nGPU = len(args.GPUs.split(','))
os.environ['CUDA_VISIBLE_DEVICES']=args.GPUs
args.lrs=[float(args.lrs.split(',')[i]) for i in range(len(args.lrs.split(',')))]



