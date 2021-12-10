import torch
from torchvision import transforms
from module import ResnetGenerator
import streamlit as st

img_size = 256


def load_model(path, device):
    ch = 64
    n_res = 4
    light = False

    genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=ch,
                             n_blocks=n_res, img_size=img_size, light=light).to(device)
    genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=ch,
                             n_blocks=n_res, img_size=img_size, light=light).to(device)

    params = torch.load(path)
    genA2B.load_state_dict(params['genA2B'])
    genB2A.load_state_dict(params['genB2A'])

    return genA2B, genB2A

def denorm(x):
    return x * 0.5 + 0.5

@st.cache(allow_output_mutation=True)
def get_generated_image(img, device):
    trfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    _input = trfm(img).unsqueeze(0).to(device)
    A2B, B2A = load_model("./data/webtoon2selfie_params_latest.pt", device)

    with torch.no_grad():
        A2B.eval()
        B2A.eval()
        A2B_output, _, _ = A2B(_input)
        B2A_output, _, _ = B2A(_input)

        A2B_output = denorm(A2B_output.cpu())
        B2A_output = denorm(B2A_output.cpu())
        
    return A2B_output, B2A_output
