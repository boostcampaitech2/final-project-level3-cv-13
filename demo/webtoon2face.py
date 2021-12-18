import argparse
import math
import os

import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms

from PIL import Image
from tqdm import tqdm

import lpips
from model import Generator


def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise

def denorm(x):
    return x * 0.5 + 0.5

def get_generated_image(img, device):

    toon_ckpt = "./data/561000.pt"
    real_ckpt = "./data/566000.pt"
    toon_fact = "./data/561000_factorization.pt"

    transform = transforms.Compose([
        transforms.CenterCrop(_min),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    imgs = transform(img).unsqueeze(0).to(device)

    eigvec = torch.load(toon_fact)["eigvec"].to(device)
    eigvec.requires_grad = False

    g_ema1 = Generator(256, 512, 8)
    g_ema1.load_state_dict(torch.load(toon_ckpt)["g_ema"], strict=False)
    g_ema1.eval()
    g_ema1 = g_ema1.to(device)

    g_ema2 = Generator(256, 512, 8)
    g_ema2.load_state_dict(torch.load(real_ckpt, map_location=device)["g_ema"], strict=False)
    g_ema2.eval()
    g_ema2 = g_ema2.to(device)

    with torch.no_grad():
        noise_sample = torch.randn(1, 512, device=device)
        latent = g_ema1.get_latent(noise_sample)

        degree_sample = torch.randn(10000, 1, device=device) * 1000
        weight_sample = torch.randn(10000, 512, device=device).unsqueeze(1) # row of factor

        degree_mean = degree_sample.mean(0)
        degree_std = ((degree_sample - degree_mean).pow(2).sum() / 10000) ** 0.5

        weight_mean = weight_sample.mean(0)
        weight_std = ((weight_sample - weight_mean).pow(2).sum() / 10000) ** 0.5 

        direction = torch.mm(eigvec, weight_mean.T) 

    percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=device.startswith(device))

    noises_single = g_ema1.make_noise()
    noises = []
    for noise in noises_single:
        noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())

    direction_in = direction.T
    weight_in = weight_mean 

    weight_mean.requires_grad = True

    for noise in noises:
        noise.requires_grad = True

    optimizer = optim.Adam([weight_mean] + noises, lr = 0.3)
    pbar = tqdm(range(300))
    latent_path = []
    weight_path = []

    imgs.detach()

    for i in pbar:
        t = i / 300
        lr = get_lr(t, 0.3)
        optimizer.param_groups[0]["lr"] = lr
        noise_strength = weight_std * 0.05 * max(0, 1 - t / 0.75) ** 2
        weight_n = latent_noise(weight_in, noise_strength.item())
        direction_n = torch.mm(weight_in, eigvec) 

        img_gen, _ = g_ema1([direction_n], input_is_latent=True, noise=noises)

        batch, channel, height, width = img_gen.shape

        if height > 256:
            factor = height // 256

            img_gen = img_gen.reshape(
                batch, channel, height // factor, factor, width // factor, factor
            )
            img_gen = img_gen.mean([3, 5])


        p_loss = percept(img_gen, imgs).sum()
        n_loss = noise_regularize(noises)
        mse_loss = F.l1_loss(img_gen, imgs)

        loss = p_loss + 1e5 * n_loss + 1 * mse_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        noise_normalize_(noises)
    
    vec = weight_in.detach().clone()
    input_latent = torch.mm(vec, eigvec)

    g_ema1.load_state_dict(torch.load(toon_ckpt, map_location=device)["g_ema"], strict=False)
    g_ema1.eval()
    g_ema1 = g_ema1.to(device)

    noises_single = g_ema2.make_noise()
    noises = []
    for noise in noises_single:
        noises.append(noise.repeat(1, 1, 1, 1).normal_())
    noise_normalize_(noises)

    with torch.no_grad():
        mean_latent = g_ema2.mean_latent(4096)

    img1, swap_res = g_ema1([input_latent], input_is_latent=True, save_for_swap=True, swap_layer=3)

    img_style, _ = g_ema2([input_latent], truncation=0.2, truncation_latent=mean_latent, swap=True, swap_layer=3,  swap_tensor=swap_res)

    img1 = denorm(img1.detach().cpu())
    img_style = denorm(img_style.detach().cpu())

    return img_style, img1
