import argparse
import cv2
from dataset import DetectionImageDataset, EmbeddingDataset
from facenet_pytorch import MTCNN, InceptionResnetV1
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import pickle
import torch
from torch.utils.data import DataLoader
from typing import List


# TODO 웹툰 이미지에 해당하는 crop 구현
def collate_fn(x):
    return x[0]

def denormalize(img, mean=127.5, std=128):
    """
    denormalize an image array

    - Args
        img : a numpy array for an image
    """
    return img * std + mean

def save_image(img_list:List[np.array], file_names, names, save_dir)->None:
    """
    save image crops

    - Args
        img_list: list of image arrays
        file_names: original file_names
    """
    for i, output in enumerate(img_list):
        if file_names[i].split('.')[-1] not in ['JPG', 'jpeg', 'jpg', 'png']:
            continue
        save_img = (denormalize(output)).cpu().numpy().astype(np.Uint64).transpose((1,2,0))[...,[2,1,0]]
        folder_path = os.path.join(save_dir, names[i])
        img_path = os.path.join(folder_path, file_names[i])

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        cv2.imwrite(img_path, save_img)
    

def main(args):
    workers = 0 if os.name == 'nt' else 4
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_path = args.data_dir


    mtcnn = MTCNN(
        image_size=args.detect_size, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


    # 사진이 없는 경우 해당 폴더는 crop대상에서 제외
    actor_names = []
    remove_names = []
    for actor_name in os.listdir(data_path):
        file_path = os.path.join(data_path, actor_name)
        if os.path.isdir(file_path):
            if len(os.listdir(file_path)) > 0: # 해당 폴더에 사진이 있는 경우
                actor_names.append(actor_name)
            else:
                remove_names.append(actor_name)

    dataset = DetectionImageDataset(data_path, actor_names)
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers, batch_size=1)            


    print("Detecting Face from images...")

    aligned, names, paths = [], [], []
    
    for x, y, path in loader:
        x_aligned = mtcnn(x, return_prob=False) 
        
        if x_aligned is not None:
            aligned.append(x_aligned)
            names.append(dataset.idx_to_class[y])
            paths.append(path)
    print("Detecting Finished.", end = ' ')
    print(f"Totally {len(aligned)} faces detected.")

    # save the crops
    
    if args.save_crops:
        print("Saving Face Crops...")
        if os.path.exists(args.crop_save_dir):
            os.makedirs(args.crop_save_dir)
        save_image(aligned, paths, names, args.crop_save_dir)
        print("Saving Face Crops Done!")   

    # calculate embedding vectors
    print("Calculating Embedding...")
    aligned_dataset = EmbeddingDataset(aligned, names, paths)
    alinged_loader = DataLoader(aligned_dataset, num_workers=workers, batch_size=50)


    saving_embeddings, saving_names, saving_img_names = [], [], []
    for x, name, img_name in alinged_loader:
        y = facenet(x.to(device))
        saving_embeddings.extend(y.detach().cpu().tolist())
        saving_names.extend(name)
        saving_img_names.extend(img_name)

    saving_img_infos = list(map(list, zip(saving_names, saving_img_names))) 


    print(f'Embedding Done! Shape of Embedding matrices: {np.array(saving_embeddings).shape}')
    
    if os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if os.path.exists(args.img_info_dir):
        os.makedirs(args.img_info_dir)

    with open(args.save_dir, 'wb') as f1:
        pickle.dump(saving_embeddings, f1)
    
    with open(args.img_info_dir, 'wb') as f2:
        pickle.dump(saving_img_infos, f2)

    print("Saving Embedding Matrices Done!")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--detect_size', type=int, default=256)
    parser.add_argument('--data_dir', type=str, default='/opt/ml/facenet_pytorch/data/actor_data')
    parser.add_argument('--save_dir', type=str, default='./embedding_results/')
    parser.add_argument('--img_info_dir', type=str, default='./embedding_image_infos/')
    parser.add_argument('--save_crops', action='store_true', help='whether to save crops. If not to save cropped images, do not use this flag.')
    parser.add_argument('--crop_save_dir', type=str, default='./cropped_results/')
    args = parser.parse_args()

    main(args)