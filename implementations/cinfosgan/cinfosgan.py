import argparse
import os, sys
import numpy as np
import itertools
import datetime
import time


import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import EncoderResNet, DecoderResNet, Discriminator, ContentCodeDiscriminator, weights_init_normal
from datasets import ImageDataset
from utils import ReplayBuffer, LambdaLR

import torch


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="monet2photo", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=2, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

cuda = torch.cuda.is_available()

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize generator and discriminator
EC_A = EncoderResNet(input_shape, opt.n_residual_blocks)
EC_B = EncoderResNet(input_shape, opt.n_residual_blocks)
ES_A = EncoderResNet(input_shape, opt.n_residual_blocks)
ES_B = EncoderResNet(input_shape, opt.n_residual_blocks)
channels, height, width = EC_A.output_shape
G_A = DecoderResNet((channels * 2, height, width), input_shape, opt.n_residual_blocks)
G_B = DecoderResNet((channels * 2, height, width), input_shape, opt.n_residual_blocks)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)
D_S = ContentCodeDiscriminator((channels, height, width))

if cuda:
    EC_A = EC_A.cuda()
    EC_B = EC_B.cuda()
    ES_A = ES_A.cuda()
    ES_B = ES_B.cuda()
    G_A = G_A.cuda()
    G_B = G_B.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    D_S = D_S.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

if opt.epoch != 0:
    # Load pretrained models
    EC_A.load_state_dict(torch.load("saved_models/%s/EC_A_%d.pth" % (opt.dataset_name, opt.epoch)))
    EC_B.load_state_dict(torch.load("saved_models/%s/EC_B_%d.pth" % (opt.dataset_name, opt.epoch)))
    ES_A.load_state_dict(torch.load("saved_models/%s/EC_A_%d.pth" % (opt.dataset_name, opt.epoch)))
    ES_B.load_state_dict(torch.load("saved_models/%s/EC_B_%d.pth" % (opt.dataset_name, opt.epoch)))
    G_A.load_state_dict(torch.load("saved_models/%s/G_A_%d.pth" % (opt.dataset_name, opt.epoch)))
    G_B.load_state_dict(torch.load("saved_models/%s/G_B_%d.pth" % (opt.dataset_name, opt.epoch)))
    D_A.load_state_dict(torch.load("saved_models/%s/D_A_%d.pth" % (opt.dataset_name, opt.epoch)))
    D_B.load_state_dict(torch.load("saved_models/%s/D_B_%d.pth" % (opt.dataset_name, opt.epoch)))
    D_S.load_state_dict(torch.load("saved_models/%s/D_S_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    EC_A.apply(weights_init_normal)
    EC_B.apply(weights_init_normal)
    ES_A.apply(weights_init_normal)
    ES_B.apply(weights_init_normal)
    G_A.apply(weights_init_normal)
    G_B.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)
    D_S.apply(weights_init_normal)

# Optimizers
optimizer_EG = torch.optim.Adam(
    itertools.chain(EC_A.parameters(), ES_A.parameters(), G_A.parameters(), 
                    EC_B.parameters(), ES_B.parameters(), G_B.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_EG = torch.optim.lr_scheduler.LambdaLR(
    optimizer_EG, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

Tensor = torch.Tensor
if cuda:
    Tensor = torch.cuda.FloatTensor

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Image transformations
transforms_ = [
    transforms.Resize(int(opt.img_height * 1.12), InterpolationMode.BICUBIC),
    transforms.RandomCrop((opt.img_height, opt.img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

# Training data loader
dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
# Test data loader
val_dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True, mode="test"),
    batch_size=5,
    shuffle=True,
    num_workers=1,
)


def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    EC_A.eval()
    ES_A.eval()
    EC_B.eval()
    ES_B.eval()

    real_A = Variable(imgs["A"].type(Tensor))
    real_B = Variable(imgs["B"].type(Tensor))

    C_A = EC_A(real_A)
    S_A = ES_A(real_A)
    C_B = EC_B(real_B)
    S_B = ES_B(real_B)

    fake_A = G_A(C_B, S_A)
    fake_B = G_B(C_A, S_B)
    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, "images/%s/%s.png" % (opt.dataset_name, batches_done), normalize=False)


# ----------
#  Training
# ----------

def main():
    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):

            # Set model input
            real_A = Variable(batch["A"].type(Tensor))
            real_B = Variable(batch["B"].type(Tensor))

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            EC_A.train()
            ES_A.train()
            EC_B.train()
            ES_B.train()

            optimizer_EG.zero_grad()

            ccode_a = EC_A(real_A)
            scode_a = ES_A(real_A)
            ccode_b = EC_B(real_B)
            scode_b = ES_B(real_B)

            # Identity loss
            loss_id_A = criterion_identity(G_A(ccode_a, scode_a), real_A)
            loss_id_B = criterion_identity(G_B(ccode_b, scode_b), real_B)

            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN loss
            fake_A = G_A(ccode_b, scode_a)
            fake_B = G_B(ccode_a, scode_b)
            loss_GAN_A = criterion_GAN(D_A(fake_A), valid)
            loss_GAN_B = criterion_GAN(D_B(fake_B), valid)

            loss_GAN = (loss_GAN_A + loss_GAN_B) / 2

            # Cycle loss
            f_ccode_a = EC_A(fake_A)
            f_scode_a = ES_A(fake_A)
            f_ccode_b = EC_B(fake_B)
            f_scode_b = ES_B(fake_B)
            recov_A = G_A(f_ccode_b, f_scode_a)
            recov_B = G_B(f_ccode_a, f_scode_b)

            loss_cycle_A = criterion_cycle(recov_A, real_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total loss
            loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity

            loss_G.backward()
            optimizer_EG.step()

            # -----------------------
            #  Train Discriminator A
            # -----------------------

            optimizer_D_A.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_A(real_A), valid)
            # Fake loss (on batch of previously generated samples)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
            # Total loss
            loss_D_A = (loss_real + loss_fake) / 2

            loss_D_A.backward()
            optimizer_D_A.step()

            # -----------------------
            #  Train Discriminator B
            # -----------------------

            optimizer_D_B.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_B(real_B), valid)
            # Fake loss (on batch of previously generated samples)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
            # Total loss
            loss_D_B = (loss_real + loss_fake) / 2

            loss_D_B.backward()
            optimizer_D_B.step()

            loss_D = (loss_D_A + loss_D_B) / 2

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_cycle.item(),
                    loss_identity.item(),
                    time_left,
                )
            )

            # If at sample interval save image
            if batches_done % opt.sample_interval == 0:
                sample_images(batches_done)

        # Update learning rates
        lr_scheduler_EG.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(EC_A.state_dict(), "saved_models/%s/EC_A_%d.pth" % (opt.dataset_name, epoch))
            torch.save(ES_A.state_dict(), "saved_models/%s/ES_A_%d.pth" % (opt.dataset_name, epoch))
            torch.save(EC_B.state_dict(), "saved_models/%s/EC_B_%d.pth" % (opt.dataset_name, epoch))
            torch.save(ES_B.state_dict(), "saved_models/%s/ES_B_%d.pth" % (opt.dataset_name, epoch))
            torch.save(G_A.state_dict(), "saved_models/%s/G_A_%d.pth" % (opt.dataset_name, epoch))
            torch.save(G_B.state_dict(), "saved_models/%s/G_B_%d.pth" % (opt.dataset_name, epoch))
            torch.save(D_A.state_dict(), "saved_models/%s/D_A_%d.pth" % (opt.dataset_name, epoch))
            torch.save(D_B.state_dict(), "saved_models/%s/D_B_%d.pth" % (opt.dataset_name, epoch))
            torch.save(D_S.state_dict(), "saved_models/%s/D_S_%d.pth" % (opt.dataset_name, epoch))

if __name__ == "__main__":
    main()
