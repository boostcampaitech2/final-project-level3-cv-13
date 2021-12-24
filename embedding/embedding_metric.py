import torch
import numpy as np
import pickle
from tqdm import tqdm
import argparse


def return_nearest_image(input_embedding, embeddings, path_infos, similarity):
    repeat_embedding = input_embedding.unsqueeze(0).expand(embeddings.shape[0], -1)
    if similarity == "l2":
        dist = np.linalg.norm(repeat_embedding - embeddings, axis=-1, ord=2)
    elif similarity == "cosine":
        dist = -(repeat_embedding*embeddings).sum(axis=-1)
    idx_sort = dist.argsort()

    return path_infos[idx_sort[1]][0]


def open_pickle(embedding_path, embedding_info_path):
    # embeddings 로드
    with open(embedding_path, "rb") as f:
        embeddings = torch.tensor(pickle.load(f))
    # path_infos 로드
    with open(embedding_info_path, "rb") as f:
        path_infos = pickle.load(f)

    return embeddings, path_infos


def main(args):
    num_correct = 0
    num_incorrect = 0

    embeddings, path_infos = open_pickle(args.embedding_path, args.embedding_info_path)

    for i in tqdm(range(len(embeddings))):
        input_embedding = embeddings[i]

        min_info = return_nearest_image(input_embedding, embeddings, path_infos, args.similarity)
        if path_infos[i][0] == min_info:
            num_correct += 1
        else:
            num_incorrect += 1

    print(f"The number of closest pair vectors with the same classes: {num_correct}")  
    print(f"The number of closest pair vectors with the differenct classes: {num_incorrect}")
    print(f"Accuray: {num_correct / (num_correct + num_incorrect)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding_path",
        "-e",
        help="path of pickle file where embedding variable is saved",
        default="/opt/ml/facenet_pytorch/data/embedding_backup.data",
    )
    parser.add_argument(
        "--embedding_info_path",
        "-i",
        help="path of pickle file where path of images is saved",
        default="/opt/ml/facenet_pytorch/data/embedding_info_backup.data",
    )
    parser.add_argument(
        "--similarity",
        "-s",
        help="method of caculating similarity",
        default="l2",
    )
    args = parser.parse_args()
    main(args)
