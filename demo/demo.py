from face2celeb import get_embedding, get_nearest_image
from webtoon2face import get_generated_image
from torchvision import transforms
from copy import deepcopy
from PIL import Image
import streamlit as st
import io
import os
import torch
import time

img_root = "./data/actor_data"


def main():
    st.title("Webtoon to Face")

    # 이미지 업로드
    cols1 = st.columns(2)
    with cols1[0]:
        k = st.number_input("Top K", min_value=1, max_value=5)
    with cols1[1]:
        uploded_file = st.file_uploader(
            "webtoon image", type=["jpg", "jpeg", "png"])

    if uploded_file:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        start_time = time.time()

        # 업로드 된 이미지 보여주기
        img_byte = uploded_file.getvalue()
        img = Image.open(io.BytesIO(img_byte))
        st.image(img, caption="Uploaded Image", width=300)

        # 업로드 된 이미지를 입력으로 GAN을 통해 실사 이미지 생성
        st.title("캐릭터의 얼굴을 실사화 합니다...")
        a2b, b2a = deepcopy(get_generated_image(img, device))

        a2b = transforms.ToPILImage()(a2b.squeeze(0))
        b2a = transforms.ToPILImage()(b2a.squeeze(0))

        st.image(a2b, caption="실사화", width=256)
        
        half_time = time.time()
        st.write("time : " + str(half_time-start_time))

        # 생성된 이미지 임베딩 추출 및 유사한 연예인 임베딩 검색
        st.title("유사한 인물을 검색합니다...")
        emb = get_embedding(a2b, device)
        distances, infos = get_nearest_image(emb, k)

        # 검색된 k개의 연예인 임베딩의 원본 이미지 보여주기
        st.write("검색 결과...")
        cols3 = st.columns(k)
        for i, (d, info) in enumerate(zip(distances, infos)):
            img_path = os.path.join(img_root, info[0], info[1])
            img = Image.open(img_path)

            with cols3[i]:
                st.image(img, caption=str(d)+info[0], width=256)

        end_time = time.time()
        st.write("time : " + str(end_time-half_time))


if __name__ == "__main__":
    main()
