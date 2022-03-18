import argparse
import os, sys
import numpy as np
import itertools
import datetime
import time
import signal


import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import DomainEncoder, DomainDecoder, DomainDiscriminator, weights_init_normal
from datasets import ImageDataset
from utils import ReplayBuffer, LambdaLR

import torch


parser = argparse.ArgumentParser()
parser.add_argument("--epoch",               type=int,   default=0,             help="epoch to start training from")
parser.add_argument("--n_epochs",            type=int,   default=200,           help="number of epochs of training")
parser.add_argument("--dataset_name",        type=str,   default="monet2photo", help="name of the dataset")
parser.add_argument("--batch_size",          type=int,   default=1,             help="size of the batches")
parser.add_argument("--lr",                  type=float, default=0.0002,        help="adam: learning rate")
parser.add_argument("--b1",                  type=float, default=0.5,           help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2",                  type=float, default=0.999,         help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch",         type=int,   default=100,           help="epoch from which to start lr decay")
parser.add_argument("--n_cpu",               type=int,   default=2,             help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height",          type=int,   default=256,           help="size of image height")
parser.add_argument("--img_width",           type=int,   default=256,           help="size of image width")
parser.add_argument("--channels",            type=int,   default=3,             help="number of image channels")
parser.add_argument("--sample_interval",     type=int,   default=100,           help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int,   default=1,             help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks",   type=int,   default=2,             help="number of residual blocks in generator")
parser.add_argument("--lambda_ae_id",       type=float, default=10,            help="autoencoder identity loss weight")
parser.add_argument("--lambda_ae_gan",      type=float, default=1,             help="autoencoder GAN loss weight")
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_identity = torch.nn.L1Loss()
criterion_cycle = torch.nn.L1Loss()

cuda = torch.cuda.is_available()

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize encoder decoder and discriminator
DomainEncoder_A = DomainEncoder(3, 2)
DomainDecoder_A = DomainDecoder(DomainEncoder_A.out_features, 2)
DomainEncoder_B = DomainEncoder(3, 2)
DomainDecoder_B = DomainDecoder(DomainEncoder_B.out_features, 2)
DomainDiscriminator_A = DomainDiscriminator((3, opt.img_height, opt.img_width))
DomainDiscriminator_B = DomainDiscriminator((3, opt.img_height, opt.img_width))
model_components = [ 
        [ DomainEncoder_A, "DomainEncoder_A" ],
        [ DomainDecoder_A, "DomainDecoder_A" ],
        [ DomainEncoder_B, "DomainEncoder_B" ],
        [ DomainDecoder_B, "DomainDecoder_B" ],
        [ DomainDiscriminator_A, "DomainDiscriminator_A" ],
        [ DomainDiscriminator_B, "DomainDiscriminator_B" ],
        ]

if cuda:
    for model in model_components:
        model[0].cuda()

for model in model_components:
    model_filename = "saved_models/%s/%s_%d.pth" % (opt.dataset_name, model[1], opt.epoch)
    be_load = False
    be_init = False
    if os.path.exists(model_filename):
        if __name__ == "__main__":
            print("loading model %s" % (model_filename))
        model[0].load_state_dict(torch.load(model_filename))
        be_load = True
    else:
        model[0].apply(weights_init_normal)
        be_init = True
    if be_load and be_init:
        print("Inconsistent model, some parts were loaded from previous saved, some not")

def save_models(epoch):
    for model in model_components:
        torch.save(model[0].state_dict(), "saved_models/%s/%s_%d.pth" % (opt.dataset_name, model[1], epoch))
    
def clear_gradient():
    for model in model_components:
        for p in model[0].parameters():
            if p.grad is not None:
                del p.grad
    torch.cuda.empty_cache()

# Optimizers
optimizer_DomainA = torch.optim.Adam(
    itertools.chain(DomainEncoder_A.parameters(), DomainDecoder_A.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_DomainB = torch.optim.Adam(
    itertools.chain(DomainEncoder_B.parameters(), DomainDecoder_B.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_DomainA_Dis = torch.optim.Adam(DomainDiscriminator_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_DomainB_Dis = torch.optim.Adam(DomainDiscriminator_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_DomainA= torch.optim.lr_scheduler.LambdaLR(
    optimizer_DomainA, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_DomainB= torch.optim.lr_scheduler.LambdaLR(
    optimizer_DomainB, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_DomainA_Dis = torch.optim.lr_scheduler.LambdaLR(
    optimizer_DomainA_Dis, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_DomainB_Dis = torch.optim.lr_scheduler.LambdaLR(
    optimizer_DomainB_Dis, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
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
    DomainEncoder_A.eval()
    DomainDecoder_A.eval()
    DomainEncoder_B.eval()
    DomainDecoder_B.eval()

    real_A = Variable(imgs["A"].type(Tensor))
    real_B = Variable(imgs["B"].type(Tensor))

    fake_A = DomainDecoder_A(DomainEncoder_A(real_A))
    fake_B = DomainDecoder_B(DomainEncoder_B(real_B))

    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_A, real_B, fake_B), 1)
    save_image(image_grid, "images/%s/%s.png" % (opt.dataset_name, batches_done), normalize=False)


saved_epoch = 0
is_training_process = False
def interrupt_exit(signum, frame):
    if not is_training_process:
        sys.exit(1)
    print("saving models...")
    save_models(saved_epoch)
    sys.exit(0)

signal.signal(signal.SIGINT, interrupt_exit)

# ----------
#  Training
# ----------
def main():
    global saved_epoch, is_training_process
    is_training_process = True
    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs):
        saved_epoch = epoch
        for i, batch in enumerate(dataloader):
            optimizer_DomainA.zero_grad()
            optimizer_DomainB.zero_grad()
            optimizer_DomainA_Dis.zero_grad()
            optimizer_DomainB_Dis.zero_grad()

            # Set model input
            real_A = Variable(batch["A"].type(Tensor))
            real_B = Variable(batch["B"].type(Tensor))

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_A.size(0), *DomainDiscriminator_A.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_A.size(0), *DomainDiscriminator_A.output_shape))), requires_grad=False)

            # ------------------
            #  Train Auto-Encoder
            # ------------------
            DomainEncoder_A.train()
            DomainDecoder_A.train()
            DomainEncoder_B.train()
            DomainDecoder_B.train()

            optimizer_DomainA.zero_grad()
            optimizer_DomainB.zero_grad()

            interm_a = DomainEncoder_A(real_A)
            interm_b = DomainEncoder_B(real_B)
            fake_A   = DomainDecoder_A(interm_a)
            fake_B   = DomainDecoder_B(interm_b)

            # Identity loss
            loss_aec_id_A = criterion_identity(fake_A, real_A)
            loss_aec_id_B = criterion_identity(fake_B, real_B)

            # GAN loss
            loss_GAN_A = criterion_GAN(DomainDiscriminator_A(fake_A), valid)
            loss_GAN_B = criterion_GAN(DomainDiscriminator_B(fake_B), valid)

            loss_domain_A = loss_GAN_A * opt.lambda_ae_gan + loss_aec_id_A * opt.lambda_ae_id
            loss_domain_B = loss_GAN_B * opt.lambda_ae_gan + loss_aec_id_B * opt.lambda_ae_id

            loss_domain_A.backward()
            loss_domain_B.backward()
            optimizer_DomainA.step()
            optimizer_DomainB.step()


            # -----------------------
            #  Train Discriminator A
            # -----------------------
            DomainDiscriminator_A.train()
            loss_real = criterion_GAN(DomainDiscriminator_A(real_A), valid)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(DomainDiscriminator_A(fake_A_.detach()), fake)
            loss_D_A = (loss_real + loss_fake) / 2
            loss_D_A.backward()
            optimizer_DomainA_Dis.step()


            # -----------------------
            #  Train Discriminator B
            # -----------------------
            DomainDiscriminator_B.train()
            loss_real = criterion_GAN(DomainDiscriminator_B(real_B), valid)
            fake_B_ = fake_A_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(DomainDiscriminator_B(fake_B_.detach()), fake)
            loss_D_B = (loss_real + loss_fake) / 2
            loss_D_B.backward()
            optimizer_DomainB_Dis.step()

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
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [AE loss: %f, adv: %f, identity: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    (loss_D_A + loss_D_B).item(),
                    (loss_domain_A + loss_domain_B).item(),
                    (loss_GAN_A + loss_GAN_B).item(),
                    (loss_aec_id_A + loss_aec_id_B).item(),
                    time_left,
                )
            )

            # If at sample interval save image
            if batches_done % opt.sample_interval == 0:
                sample_images(batches_done)

        # Update learning rates
        lr_scheduler_DomainA.step()
        lr_scheduler_DomainB.step()
        lr_scheduler_DomainA_Dis.step()
        lr_scheduler_DomainB_Dis.step()

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            save_models(epoch)


if __name__ == "__main__":
    while True:
        try:
            main()
        except RuntimeError as e:
            if 'out of meomory' in str(e):
                print("|Warning: out of memory")
                clear_gradient()
                torch.cuda.empty_cache()
            else:
                raise e 
