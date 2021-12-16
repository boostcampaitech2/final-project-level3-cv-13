from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np
import pickle
from torchvision import transforms


def crop_face(img, device):
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )

    with torch.no_grad():
        mtcnn.eval()
        x_aligned = mtcnn(img, return_prob=False)

    return x_aligned


def get_embedding(img, device):
    x_aligned = crop_face(img, device)

    # 입력된 이미지에서 얼굴을 찾지 못할 경우 입력된 이미지 자체로 임베딩 추출
    if x_aligned == None:
        x_aligned = transforms.ToTensor()(img)

    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    with torch.no_grad():
        input_embedding = resnet(
            x_aligned.unsqueeze(0).to(device)).cpu()

    return input_embedding


def caculate_embedding_distance(input_embedding, embeddings):
    dist = np.array([(input_embedding - embedding).norm().item()
                    for embedding in embeddings])
    
    idx_sort = dist.argsort()
    dist = np.sort(dist)

    return dist, idx_sort


def get_nearest_image(input_embedding, k=1):
    embedding_saving_path = "./data/embedding.data"
    embedding_info_path = "./data/embedding_info.data"

    with open(embedding_saving_path, 'rb') as f:
        embeddings = torch.tensor(pickle.load(f))

    with open(embedding_info_path, 'rb') as f:
        path_infos = pickle.load(f)

    dist, idx_sort = caculate_embedding_distance(input_embedding, embeddings)

    return dist[:k], [path_infos[i] for i in idx_sort[:k]]
