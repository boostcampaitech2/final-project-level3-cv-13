import torch
from torchvision import transforms
from module import ResnetGenerator

img_size = 256


def load_model(path, device):
    ch = 64
    n_res = 4
    light = False

    genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=ch,
                             n_blocks=n_res, img_size=img_size, light=light).to(device)
  
    params = torch.load(path)
    genA2B.load_state_dict(params['genA2B'])
  
    return genA2B

def denorm(x):
    return x * 0.5 + 0.5

def get_generated_image(img, device):
    width, height = img.size[0], img.size[1]
    _min = min(width, height)

    trfm = transforms.Compose([
        transforms.CenterCrop(_min),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    _input = trfm(img).unsqueeze(0).to(device)
    A2B = load_model("./data/toon2real_params_latest.pt", device)

    with torch.no_grad():
        A2B.eval()
        A2B_output, _, _ = A2B(_input)
        A2B_output = denorm(A2B_output.cpu())
        
    return A2B_output
