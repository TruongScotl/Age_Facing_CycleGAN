import torch
from dataset import AgeDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator

#zebra = young, hourse = old
def train_fn(disc_O, disc_Y, gen_Y, gen_O, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    O_reals = 0
    O_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (young, old) in enumerate(loop):
        young = young.to(config.DEVICE)
        old = old.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_old = gen_O(young)
            D_O_real = disc_O(old)
            D_O_fake = disc_O(fake_old.detach())
            O_reals += D_O_real.mean().item()
            O_fakes += D_O_fake.mean().item()
            D_O_real_loss = mse(D_O_real, torch.ones_like(D_O_real))
            D_O_fake_loss = mse(D_O_fake, torch.zeros_like(D_O_fake))
            D_O_loss = D_O_real_loss + D_O_fake_loss

            fake_young = gen_Y(old)
            D_Y_real = disc_Y(young)
            D_Y_fake = disc_Y(fake_young.detach())
            D_Y_real_loss = mse(D_Y_real, torch.ones_like(D_Y_real))
            D_Y_fake_loss = mse(D_Y_fake, torch.zeros_like(D_Y_fake))
            D_Y_loss = D_Y_real_loss + D_Y_fake_loss

            # put it togethor
            D_loss = (D_O_loss + D_Y_loss)/2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_O_fake = disc_O(fake_old)
            D_Y_fake = disc_Y(fake_young)
            loss_G_O = mse(D_O_fake, torch.ones_like(D_O_fake))
            loss_G_Y = mse(D_Y_fake, torch.ones_like(D_Y_fake))

            # cycle loss
            cycle_young = gen_Y(fake_old)
            cycle_old = gen_O(fake_young)
            cycle_young_loss = l1(young, cycle_young)
            cycle_old_loss = l1(old, cycle_old)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_young = gen_Y(young)
            identity_old = gen_O(old)
            identity_young_loss = l1(young, identity_young)
            identity_old_loss = l1(old, identity_old)

            # add all togethor
            G_loss = (
                loss_G_Y
                + loss_G_O
                + cycle_young_loss * config.LAMBDA_CYCLE
                + cycle_old_loss * config.LAMBDA_CYCLE
                + identity_old_loss * config.LAMBDA_IDENTITY
                + identity_young_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            # save_image(fake_old*0.5+0.5, f"saved_images/old_{idx}.png")
            # save_image(fake_young*0.5+0.5, f"saved_images/young_{idx}.png")
            save_image(fake_old*0.5+0.5, f"old_{idx}.png")
            save_image(fake_young*0.5+0.5, f"young_{idx}.png")

        loop.set_postfix(O_real=O_reals/(idx+1), O_fake=O_fakes/(idx+1))


# torch.compile is avaiable with torch 2.0
def main():
    disc_O = torch.compile(Discriminator(in_channels=3), mode="reduce-overhead").to(config.DEVICE)
    disc_Y = torch.compile(Discriminator(in_channels=3),mode="reduce-overhead").to(config.DEVICE)
    gen_Y = torch.compile(Generator(img_channels=3, num_residuals=9), mode="reduce-overhead").to(config.DEVICE)
    gen_O = torch.compile(Generator(img_channels=3, num_residuals=9), mode="reduce-overhead").to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_O.parameters()) + list(disc_Y.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_Y.parameters()) + list(gen_O.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_O, gen_O, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_Y, gen_Y, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_O, disc_O, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_Y, disc_Y, opt_disc, config.LEARNING_RATE,
        )

    dataset = AgeDataset(
        root_old=config.TRAIN_DIR+"/trainA", root_young=config.TRAIN_DIR+"/trainE", transform=config.transforms
    )
    # val_dataset = AgeDataset(
    #    root_old="/testA", root_young="/testE", transform=config.transforms
    # )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     pin_memory=True)

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc_O, disc_Y, gen_Y, gen_O, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)

        if config.SAVE_MODEL:
            save_checkpoint(gen_O, opt_gen, filename=config.CHECKPOINT_GEN_O)
            save_checkpoint(gen_Y, opt_gen, filename=config.CHECKPOINT_GEN_Y)
            save_checkpoint(disc_O, opt_disc, filename=config.CHECKPOINT_CRITIC_O)
            save_checkpoint(disc_Y, opt_disc, filename=config.CHECKPOINT_CRITIC_Y)

if __name__ == "__main__":
    main()