import torch

def get_optimizer(args, model):
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001,nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise Exception('unknown optimizer')
    return optimizer


def get_scheduler(args, optimizer):
    if args.scheduler == 'none':
        return None
    elif args.scheduler =='clr':
        return torch.optim.lr_scheduler.CyclicLR(optimizer, 0.01, 0.015, mode='triangular2', step_size_up=250000, cycle_momentum=False)
    elif args.scheduler =='exp':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999283, last_epoch=-1)
    elif args.scheduler =='mlr':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5*args.ep, 0.75*args.ep], gamma=0.1)
    elif args.scheduler =='cos':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.ep)
    else:
        raise Exception('unknown scheduler')
