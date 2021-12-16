from PIL import Image
import streamlit as st
import io
import time
import requests


def main():
    st.title("Webtoon to Face")

    # 이미지 업로드
    cols1 = st.columns(2)
    with cols1[0]:
        k = st.number_input("Top K", min_value=1, max_value=5)
    with cols1[1]:
        uploaded_file = st.file_uploader(
            "webtoon image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        start_time = time.time()

        # 업로드 된 이미지 보여주기
        img_byte = uploaded_file.getvalue()
        img = Image.open(io.BytesIO(img_byte)).convert('RGB')
        st.image(img, caption=f"Uploaded Image {img.size}", width=300)

        # 업로드 된 이미지를 입력으로 GAN을 통해 실사 이미지 생성
        st.title("캐릭터의 얼굴을 실사화 합니다...")
        files = [
            ('files', (uploaded_file.name, img_byte,
                       uploaded_file.type))
        ]
        r = requests.post("http://localhost:8001/gan", files=files)

        a2b = Image.open(io.BytesIO(r.content)).convert('RGB')
        st.image(a2b, caption=f"gen Image {img.size}", width=256)

        half_time = time.time()
        st.write("time : " + str(half_time-start_time))

        # # 생성된 이미지 임베딩 추출 및 유사한 연예인 임베딩 검색
        st.title("유사한 인물을 검색합니다...")
        file = {"file": r.content}
        r = requests.post("http://localhost:8001/embed", files=file)
        info = r.json()['infos']
        dist = r.json()['dist']

        # # 검색된 k개의 연예인 임베딩의 원본 이미지 보여주기
        st.write("검색 결과...")
        cols3 = st.columns(k)
        for i in range(k):
            r = requests.post(
                "http://localhost:8001/get_img/"+info[i][0]+"/"+info[i][1])
            img = Image.open(io.BytesIO(r.content)).convert('RGB')

            with cols3[i]:
                st.image(img, caption=str(dist[i]) + info[i][0], width=256)

        end_time = time.time()
        st.write("time : " + str(end_time-half_time))


if __name__ == "__main__":
    main()
