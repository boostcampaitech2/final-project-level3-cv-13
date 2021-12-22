# final-project-level3-cv-13

# CASToon

---

### 1. 프로젝트 개요

웹툰을 기반으로한 드라마 제작이 많아지면서 온라인 상에서 웹툰 캐릭터 가상 캐스팅의 콘텐츠가 인기를 끌고 있다. 이를 실제 웹툰 캐릭터의 '이미지'만을 기반으로 하여 배우 가상캐스팅을 할 수 있는 프로젝트를 수행하였다.

[ 파이프라인 이미지사진 첨부 ]

---

### 2.  디렉토리 구조

- **UGOTIT-pytorch** : U-GAT-IT 모델을 학습 및 모델 생성
- **UI2I_via_StyleGAN2** : StyleGAN2 모델을 학습 및 모델 생성
- **demo** : 데모 페이지 실행
- **embedding** : 사람 및 웹툰 얼굴 embedding 생성 및 시각화

---

### 3. 실행 환경

- Python 3.7.11
- `pip install -r requirements.txt`

```
anime_face_detector==0.0.6
beautifulsoup4==4.10.0
facenet_pytorch==2.5.2
fastapi==0.70.1
ipython==7.30.1
lmdb==1.2.1
numpy==1.21.4
opencv_python==4.5.4.60
Pillow==8.4.0
requests==2.26.0
scikit_image==0.19.1
scipy==1.7.3
skimage==0.0
streamlit==1.3.0
torch==1.10.0
torchsummary==1.5.1
torchvision==0.11.1
tqdm==4.62.3
uvicorn==0.16.0
wandb==0.12.9
```

---

### 4-1. Training 단계

- Face Detector (RetinaFace)

  **Step 1)** 웹툰 이미지의 face crop 얻기

  ```bash
  cd webtoon_det
  python align_crop.py --image_name IMAGE_NAME
  										 --model MODEL
  										 --face_thres FACE_THRES
  										 --landmark_thres LANDMARK_THRES
  ```

  **Step 2)** 얼굴 이미지의 face crop과 embedding 얻기

  ```python
  cd ../embedding
  python get_face_embedding.py --detect_size DETECT_SIZE
  														 --data_dir DATA_DIR
  														 --save_dir SAVE_DIR
  														 --img_info_dir IMAGE_INFO_DIR
  														 --save_crops SAVE_CROPS
  														 --crop_save_dir CROP_SAVE_DIR
  ```

- Face Generator (UI2I)

  UI2I 학습에 앞서 align과 crop이 완료된 이미지로 구성이 된 웹툰 데이터셋과 연예인 데이터셋을 준비

  **Step 1)**  각 도메인 데이터셋을 LMDB 데이터 format으로 변환 

  ```bash
  cd UI2I_via_StyleGAN2
  python prepare_data.py --out WEBTOON_LMDB_PATH --n_worker N_WORKER --size SIZE1,SIZE2,SIZE3,... WEBTOON_DATASET_PATH
  python prepare_data.py --out PHOTO_LMDB_PATH --n_worker N_WORKER --size SIZE1,SIZE2,SIZE3,... PHOTO_DATASET_PATH
  ```

  **Step 2)** 각 Domain 의 Generator Fine-Tuning

  도메인별로 저장된 LMDB 데이터셋과 pre-trained weight를 이용하여 fine-tuning을 진행합니다. 이 과정에서 pre-trained weight 은 [이곳](https://drive.google.com/file/d/1PQutd-JboOCOZqmd95XWxWrO8gGEvRcO/view)을 사용했습니다.

  ```bash
  python -m torch.distributed.launch --nproc_per_node=N_GPU --master_port=PORT train.py --batch BATCH_SIZE WEBTOON_LMDB_PATH --ckpt your_base_model_path
  python -m torch.distributed.launch --nproc_per_node=N_GPU --master_port=PORT train.py --batch BATCH_SIZE PHOTO_LMDB_PATH --ckpt your_base_model_path
  ```

  **Step 3)** Model weight를 이용하여 latent Code를 얻기 위해 사용되는 Factor계산

  ```bash
  python3 closed_form_factorization.py --ckpt webtoon_stylegan_model_path --out webtoon_stylegan_model_factor_path
  ```

### 4-2. Inference 단계

**Step 1)** 입력 웹툰 이미지 align

```python
cd webtoon_det
python align_crop.py --image_name IMAGE_NAME
										 --model MODEL
										 --face_thres FACE_THRES
										 --landmark_thres LANDMARK_THRES
```

**Step 2)** 학습시킨 StyleGAN2 모델을 기반으로 Source Domain(웹툰)의 이미지를 Latent Code로 변환 후, Latent code를 이용해 다양한 style의 이미지를 생성

```python
cd UI2I_via_StyleGAN2
python projector_factor.py --ckpt webtoon_stylegan_model_path --fact webtoon_stylegan_model_factor_path IMAGE_FILE
python gen_multi_style.py --model1 webtoon_model_path --model2 photo_model_path --fact webtoon_inverse.pt --fact_base webtoon_stylegan_model_factor_path -o output_path --swap_layer 3 --stylenum 10
```

**Step 3)** 생성된 얼굴 Recognition (임베딩 계산)
**Step 4)** 얼굴 유사도 계산 후 k개의 가까운 이미지 출력

```python
cd ../demo
python face2celeb.py
```

---

### 5. 사용한 Datasets

- 네이버 웹툰 크롤링 이미지
- 배우 크롤링 이미지
- 인스타 크롤링 이미지

---

### 6. Demo

- 실행 결과 (사진 3~5장 정도?)


- demo 영상

  [https://youtu.be/_sUZ2_L7Owg](https://youtu.be/_sUZ2_L7Owg)

---

### 👋 7. 팀 소개 <a name = 'Team'></a>

- 조원 : 장동주, 최한준, 이유진, 차미경, 서동진, 오주영

|                                                         장동주                                                                                                                   |                                                            최한준                                                             |                                                          이유진                                                           |                                                            차미경                                                            |                                                            서동진                                                             |                                                         오주영                                                             |                                                            
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | 
| <a href='https://github.com/tygu1004'><img src='https://github.com/boostcampaitech2/final-project-level3-cv-13/blob/main/contributors/%EC%9E%A5%EB%8F%99%EC%A3%BC.png' width='200px'/></a> | <a href='https://github.com/gkswns3708'><img src='https://github.com/boostcampaitech2/final-project-level3-cv-13/blob/main/contributors/%EC%B5%9C%ED%95%9C%EC%A4%80.png' width='200px'/></a> | <a href='https://github.com/YoojLee'><img src='https://github.com/boostcampaitech2/final-project-level3-cv-13/blob/main/contributors/%EC%9D%B4%EC%9C%A0%EC%A7%84.png' width='200px'/></a> | <a href='https://github.com/note823'><img src='https://github.com/boostcampaitech2/final-project-level3-cv-13/blob/main/contributors/%EC%B0%A8%EB%AF%B8%EA%B2%BD.png' width='200px'/></a> | <a href='https://github.com/SEOzizou'><img src='https://github.com/boostcampaitech2/final-project-level3-cv-13/blob/main/contributors/%EC%84%9C%EB%8F%99%EC%A7%84.png' width='200px'/></a> | <a href='https://github.com/Jy0923'><img src='https://github.com/boostcampaitech2/final-project-level3-cv-13/blob/main/contributors/%EC%98%A4%EC%A3%BC%EC%98%81.png' width='200px'/></a> 

### 8. Reference

1. Image-translation
   - U-GAT-IT : [https://arxiv.org/abs/1907.10830](https://arxiv.org/abs/1907.10830)
   - UI2I via Pre-trained StyleGAN2 Network  : [https://arxiv.org/pdf/2010.05713](https://arxiv.org/pdf/2010.05713.pdf)
2. Face Detection & Embedding
   - FaceNet:  [https://arxiv.org/abs/1503.03832](https://arxiv.org/abs/1503.03832)
   - MTCNN: [https://arxiv.org/abs/1604.02878](https://arxiv.org/abs/1604.02878)
   - RetinaFace: [https://arxiv.org/abs/1905.00641](https://arxiv.org/abs/1905.00641)
   - Anime Face Detector : [https://github.com/hysts/anime-face-detector](https://github.com/hysts/anime-face-detector)
3. Data
   - NaverWebtoonData 크롤링 ([https://github.com/bryandlee/naver-webtoon-faces](https://github.com/bryandlee/naver-webtoon-faces))
   - VGGFace2 : [https://github.com/ox-vgg/vgg_face2](https://github.com/ox-vgg/vgg_face2)
   - iCartoonFace Dataset : [https://github.com/luxiangju-PersonAI/iCartoonFace](
