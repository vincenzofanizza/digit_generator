import os
import torch
import numpy as np
import random
import logging
import torchvision
import cv2


logger = logging.getLogger(__name__)


def setup_logger():
    '''
    Configure logger.

    '''
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=head, datefmt='%Y/%m/%d %H:%M:%S')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

def set_seed(seed):
    '''
    Set random seed.

    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    logger.info('set random seed value to {}'.format(seed))

def num_total_parameters(model):
    '''
    Print total number of parameters for a certain model.

    '''
    return sum(p.numel() for p in model.parameters())

def train_one_epoch(params, dataloader, disc, gen, criterion, opt_disc, opt_gen, epoch, device, fixed_noise, writer_fake, writer_real, step, out_dir):
    '''
    Train a GAN for a full epoch.
    
    '''
    # Switch to train mode
    disc.train()
    gen.train()

    for batch_idx, (real, _) in enumerate(dataloader):
        # Flatten real image for a FC model
        if params['model']['name'] == 'fc':
            real = real.view(-1, 28*28)
        real = real.to(device)
        
        # Generate fake images from random noise
        noise = torch.randn(params['batch_size'], params['z_dim']).to(device)
        fake = gen(noise)

        # Train discriminator
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        lossD = (lossD_real + lossD_fake)/2

        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        # Make inference on fake images
        output = disc(fake).view(-1)

        # Train generator    
        lossG = criterion(output, torch.ones_like(output))
        
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx % 500 == 0:
            logger.info(
                f"Epoch [{epoch}/{params['epochs']}] Batch {batch_idx}/{len(dataloader)} \
                        Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                # Save image grid with tensorboard
                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )

                # Save image grid for future processing
                fake_grid_array = np.moveaxis(img_grid_fake.cpu().numpy(), 0, -1) * 255.
                fake_save_dir = os.path.join(out_dir, 'images', 'fake')
                if not os.path.isdir(fake_save_dir):
                    os.makedirs(fake_save_dir)
                cv2.imwrite(os.path.join(fake_save_dir, '{0:04}.jpg'.format(step)), fake_grid_array)

                real_grid_array = np.moveaxis(img_grid_real.cpu().numpy(), 0, -1) * 255.
                real_save_dir = os.path.join(out_dir, 'images', 'real')
                if not os.path.isdir(real_save_dir):
                    os.makedirs(real_save_dir)
                cv2.imwrite(os.path.join(real_save_dir, '{0:04}.jpg'.format(step)), real_grid_array)

                step += 1
        
    return lossD, lossG,  step
    
def save_checkpoint(states, output_dir, filename='checkpoint.pth'):
    '''
    Save Pytorch checkpoint.

    '''
    logger.info('=> saving checkpoint to {}'.format(output_dir))

    # Save model from last epoch
    torch.save(states, os.path.join(output_dir, filename))
