import argparse
import logging
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as T

import core.nets.fc as FC
import core.nets.cnn as CNN
import core.utils as U

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
    

logger = logging.getLogger(__name__)


def parse_args():
    '''
    Parse arguments from command line.

    '''
    parser = argparse.ArgumentParser(description='Train a GAN on the MNIST dataset.')

    parser.add_argument('--cfg',
                        help='configuration file.',
                        required=True,
                        type=str)
    parser.add_argument('--out_dir',
                        help='output directory.',
                        required=True,
                        type=str)
    
    return parser.parse_args()

def main():
    '''
    Train a GAN on the MNIST dataset.

    '''
    args = parse_args()

    # Set logger
    U.setup_logger()

    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info('using {} device'.format(device))

    # Read configuration file
    with open(args.cfg, 'r') as f:
        params = json.load(f)
    logger.info('loaded configuration from {}'.format(args.cfg))

    # Create output directory
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    logger.info('checkpoints will be saved at {}'.format(args.out_dir))

    # Set random seed
    U.set_seed(params['seed'])

    # Create networks
    if params['model']['name'] == 'fc':
        disc = FC.Discriminator().to(device)
        gen = FC.Generator(params['z_dim']).to(device)
    elif params['model']['name'] == 'cnn':
        disc = CNN.Discriminator().to(device)
        gen = CNN.Generator(params['z_dim']).to(device)
    else:
        raise ValueError('the selected model type is not implemented')
    logger.info('created networks')
    logger.info('discriminator: {} parameters'.format(U.num_total_parameters(disc)))
    logger.info('generator: {} parameters'.format(U.num_total_parameters(gen)))

    # Create a fixed noise vector to visualize the evolution of the output during training
    fixed_noise = torch.randn((params['batch_size'], params['z_dim'])).to(device)

    # Define transforms
    transforms = T.Compose([
        T.ToTensor(), 
        T.Normalize((0.1307,), (0.3081,))
    ])

    # Create dataloader
    dataset = datasets.MNIST(root=os.path.join(args.out_dir, 'dataset'), transform=transforms, download=True)
    dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)
    logger.info('created dataloader... images are downloaded to {}'.format(os.path.join(args.out_dir, 'dataset')))

    # Create optimizers
    opt_disc = optim.Adam(disc.parameters(), lr=params['lr'])
    opt_gen = optim.Adam(gen.parameters(), lr=params['lr'])
    logger.info('using adam optimizer with base learning rate {}'.format(params['lr']))

    # Define criterion
    criterion = nn.BCELoss()
    logger.info('using BCE loss')

    # Define tensorboard writer
    writer_fake = SummaryWriter(log_dir=os.path.join(args.out_dir, 'log', 'fake'))
    writer_real = SummaryWriter(log_dir=os.path.join(args.out_dir, 'log', 'real'))
    step = 0

    for epoch in range(params['epochs']):

        lossD, lossG, step = U.train_one_epoch(
            params, 
            dataloader, 
            disc, gen, 
            criterion, 
            opt_disc, opt_gen, 
            epoch, 
            device, 
            fixed_noise, 
            writer_fake, writer_real,
            step,
            args.out_dir
        )

        # Save checkpoint
        U.save_checkpoint({
            'epoch': epoch + 1,
            'model': params['model']['name'],
            'disc_state_dict': disc.state_dict(),
            'gen_state_dict': gen.state_dict(),
            'lossD': lossD,
            'lossG': lossG,
            'disc_optimizer': opt_disc.state_dict(),
            'gen_optimizer': opt_gen.state_dict(),
        }, args.out_dir)


if __name__=='__main__':
    main()