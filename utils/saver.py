import torch
import time
import os

class Saver():
    def __init__(self, args, model, opti):
        self.load_dir=args.load_dir
        self.save_dir=time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        if not os.path.exists('./exps'):
            os.mkdir('./exps')
        if not os.path.exists('./exps/{}'.format(self.save_dir)):
            os.mkdir('./exps/{}'.format(self.save_dir))
        self.save_dir='./exps/{}'.format(self.save_dir)

        self.model=model
        self.opti=opti

        f1 = open(self.save_dir + '/config.txt', 'w')
        keys = args.__dict__.keys()
        for k in keys:
            f1.write('{}:{}\n'.format(k, args.__dict__[k]))



    def save(self, epoch):
        ckpt={
            'model':self.model.get_module().state_dict(),
            'opti':self.opti.state_dict(),
            'epoch':epoch,
        }
        torch.save(ckpt,self.save_dir+'/md_{}.pkl'.format(epoch))


    def load(self):
        if self.load_dir != '':
            print('loading from {}'.format(self.load_dir))
            ckpt=torch.load(self.load_dir)
            self.model.get_module().load_state_dict(ckpt['model'])
            self.opti.load_state_dict(ckpt['opti'])




