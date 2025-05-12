import torch
def get_optimizer(args, net):
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                         weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError
    return optimizer

