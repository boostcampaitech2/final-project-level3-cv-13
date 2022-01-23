from PIL import Image
import streamlit as st
import io
import time
import requests


# 생성된 이미지 임베딩 추출 및 유사한 연예인 임베딩 검색
@st.cache(ttl=300)
def embed_phase(r):
  
    file = {"file": r.content}
    r = requests.post("http://localhost:8001/embed", files=file)
    info = r.json()['infos']
    dist = r.json()['dist']

    return info, dist


# 업로드 된 이미지를 입력으로 GAN을 통해 실사 이미지 생성
@st.cache(ttl=300)
def gan_phase(uploaded_file, img_byte, gan_ver):   
    files = [
        ('files', (uploaded_file.name, img_byte,
                    uploaded_file.type))
    ]
    r = requests.post(f"http://localhost:8001/{gan_ver}", files=files)
    a2b = Image.open(io.BytesIO(r.content)).convert('RGB')

    return r, a2b


def main():
    st.title("웹툰 가상 캐스팅")    
    st.write("웹툰 캐릭터의 얼굴 하나가 포함된 이미지를 업로드해주세요!")
    st.write("얼굴 위주의 이미지여야 합니다!")

    gan_ver = st.selectbox('생성 모델 선택',('StyleGAN','U-GAT-IT'))
    st.write('You selected:', gan_ver)
    
    # 이미지 업로드
    cols1 = st.columns(2)
    with cols1[0]:
        k = st.number_input("몇 명을 찾아볼까요? (최대 5명)", min_value=1, max_value=5)
    with cols1[1]:
        uploaded_file = st.file_uploader(
            "webtoon image", type=["jpg", "jpeg", "png"])

    
    # 업로드 된 이미지 보여주기
    if uploaded_file:
        img_byte = uploaded_file.getvalue()
        img = Image.open(io.BytesIO(img_byte)).convert('RGB')
        st.image(img, caption=f"Uploaded Image {img.size}", width=300)

        st.title("캐릭터의 얼굴을 실사화 합니다...")
        st.write("30초 정도 소요됩니다... 잠깐만 기다려주세요...")

        start_time = time.time()
        r, a2b = gan_phase(uploaded_file, img_byte, gan_ver)
        st.image(a2b, caption=f"generated image", width=256)
        end_time = time.time()
        st.write("time : " + str(end_time-start_time))

        st.title("유사한 인물을 검색합니다...")
        start_time = time.time()
        info, dist = embed_phase(r)
        st.write("검색 결과...")
            
        cols3 = st.columns(k)
        for i in range(k):
            r = requests.post(
                "http://localhost:8001/get_img/"+info[i][0]+"/"+info[i][1])
            img = Image.open(io.BytesIO(r.content)).convert('RGB')

            with cols3[i]:
                st.image(img, caption=str(dist[i]) + info[i][0], width=256)
        
        end_time = time.time()
        st.write("time : " + str(end_time-start_time))

if __name__ == "__main__":
    main()
