from torch.optim import *
from .lr_schedulers import *


def Opti(args,model):
    params=model.get_module().params_groups

    if args.optimizer == 'SGD':
        opti = SGD(params=params,
                   lr=args.lrs[0],
                   momentum=args.momentum,
                   dampening=args.dampening,
                   weight_decay=args.weight_decay,
                   nesterov=args.nesterov)
    elif args.optimizer == 'Adam':
        opti = Adam(params=params,
                    lr=args.lrs[0],
                    weight_decay=args.weight_decay
                    )
    else:
        raise Exception

    lr_s=LR(args,opti)


    return opti, lr_s
