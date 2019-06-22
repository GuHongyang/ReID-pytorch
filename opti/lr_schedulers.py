
class LR():
    def __init__(self,args,opti):
        self.warm_training=args.warm_training
        self.warm_lr=args.warm_lr
        self.gamma=args.gamma
        self.milestones=[int(i) for i in args.milestones.split('-')]
        self.epochs=args.epochs
        self.lrs=args.lrs
        self.opti=opti


    def __call__(self, epoch):
        opti=self.opti
        if epoch<self.warm_training:
            alpha=((self.lrs[0]-self.warm_lr)/(self.warm_training-1)*epoch+self.warm_lr)/self.lrs[0]
            for i, param_group in enumerate(opti.param_groups):
                param_group['lr'] = self.lrs[i] * alpha
        else:
            if epoch == 0:
                for j, param_group in enumerate(opti.param_groups):
                    param_group['lr'] = self.lrs[j]
            else:
                i=0
                while i<len(self.milestones) and self.milestones[i] <= epoch:
                    i+=1
                if i>0 and self.milestones[i-1] == epoch:
                    alpha=0.1**i
                    for j, param_group in enumerate(opti.param_groups):
                        param_group['lr'] = self.lrs[j] * alpha

        return {'LR_{}'.format(i):param_group['lr'] for i, param_group in enumerate(opti.param_groups)}




