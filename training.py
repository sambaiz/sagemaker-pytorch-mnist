import logging
import torch
from torchvision import datasets, transforms
import torch.distributed as dist
import os
import model as md
import argparse
import sys
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def _make_train_loader(batch_size, data_dir, is_distributed, **kwargs):
    dataset = datasets.MNIST(data_dir, train=True, transform=transforms.ToTensor(), download=False)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed else None
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train_sampler is None, sampler=train_sampler, **kwargs)


def _make_test_loader(batch_size, data_dir, **kwargs):
    dataset = datasets.MNIST(data_dir, train=False, transform=transforms.ToTensor(), download=False)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, **kwargs)


def train(args):
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    use_cuda = args.num_gpus > 0
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ['WORLD_SIZE'] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ['RANK'] = str(host_rank)
        dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    model = md.Model(args.dropout).to(device)
    if is_distributed and use_cuda:
        # multi-machine multi-gpu case
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        # single-machine multi-gpu case or single-machine or multi-machine cpu case
        model = torch.nn.DataParallel(model)

    train_loader = _make_train_loader(args.batch_size, args.data_dir, is_distributed, **kwargs)
    test_loader = _make_test_loader(args.test_batch_size, args.data_dir, **kwargs)

    for epoch in range(1, args.epochs + 1):
        logger.info(f'epoch: {epoch}/{args.epochs}')
        md.train(model, train_loader, device, is_distributed, args.lr, args.momentum)
        test_accuracy, test_loss = md.test(model, test_loader, device)
        logger.info(f'test accuracy: {test_accuracy}, test loss: {test_loss};')
    save_model(model, args.model_dir)


def save_model(model, model_dir):
    path = os.path.join(model_dir, 'model.pt')
    torch.jit.save(torch.jit.script(model.module), path)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # hyperparameters
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--backend', type=str, default=None, help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')
    # model params
    parser.add_argument('--dropout', type=float, default=0.5, metavar='DROP', help='dropout rate (default: 0.5)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')

    # Container environment
    parser.add_argument('--hosts', type=str, default=os.environ.get('SM_HOSTS', '[]'))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '.'))
    parser.add_argument('--num-gpus', type=int, default=os.environ.get('SM_NUM_GPUS', 0))

    # fit() inputs (SM_CHANNEL_XXXX)
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', 'mnist'))

    args = parser.parse_args()
    args.hosts = json.loads(args.hosts)

    train(args)
