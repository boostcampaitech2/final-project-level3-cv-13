import argparse
from webtoon2face import tune_weight, load_models, denorm
from torchvision import transforms
import torch
from PIL import Image
from face2celeb import get_embedding, get_nearest_image
import os
import cv2
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--toon_ckpt", type=str, required=True)
    parser.add_argument("--real_ckpt", type=str, required=True)
    parser.add_argument("--toon_fact", type=str, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--webtoon_image_path", type = str, required = True)
    parser.add_argument("-o", "--output", type=str, default="output")
   # parser.add_argument("--img_path",  type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_k", type = int, default = 1)
    args = parser.parse_args()

    transform = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    img_path = [args.webtoon_image_path]

    imgs = []
    for imgfile in img_path:
        img = transform(Image.open(imgfile).convert("RGB"))
        imgs.append(img)

    imgs = torch.stack(imgs, 0).to(args.device)

    g_ema1, g_ema2 = load_models(args.toon_ckpt, args.real_ckpt, args.device)
    eigvec = torch.load(args.toon_fact)["eigvec"].to(args.device)
    eigvec.requires_grad = False

    noises_single = g_ema1.make_noise()
    noises = []
    for noise in noises_single:
        n = noise.repeat(imgs.shape[0], 1, 1, 1).normal_()
        n.requires_grad = True
        noises.append(n)
    weight_sample = torch.randn(10000, 512, device=args.device).unsqueeze(1)
    weight_mean = weight_sample.mean(0)

    weight_mean = tune_weight(noises, weight_mean, eigvec, imgs, g_ema1, args.device, 300)
    input_latent = torch.mm(weight_mean, eigvec)

    with torch.no_grad():
        mean_latent = g_ema2.mean_latent(4096)
        _, swap_res = g_ema1([input_latent], input_is_latent=True,
                             save_for_swap=True, swap_layer=3)
        img_style, _ = g_ema2([input_latent], truncation=0.2, truncation_latent=mean_latent,
                              swap=True, swap_layer=3,  swap_tensor=swap_res)

    img_style = denorm(img_style.detach().cpu())
    img_style = img_style[0].permute(1,2,0).numpy()*255
    emb = get_embedding(img_style, args.device)
    img_style = cv2.cvtColor((img_style).astype(np.uint8), cv2.COLOR_BGR2RGB)
    distances, infos = get_nearest_image(emb, args.max_k)
    distances = distances.tolist()
    cv2.imwrite(args.output+'/'+infos[0][1], cv2.imread(os.path.join(args.img_path,infos[0][1])))
    cv2.imwrite(args.output+'/result.jpg',img_style)
 