from fastapi import FastAPI, File, UploadFile
from fastapi import FastAPI, Response
import uvicorn
from typing import List
from PIL import Image
import io
import torch
from webtoon2face import get_generated_image
from face2celeb import get_embedding, get_nearest_image
import os
from torchvision import transforms

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"
img_root = "./data/actor_data"


@app.post("/gan", description="gan")
async def gan_img(files: List[UploadFile] = File(...)):
    for file in files:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        a2b = get_generated_image(img, device)

    a2b = transforms.ToPILImage()(a2b.squeeze(0))

    img_byte_arr = io.BytesIO()
    a2b.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    return Response(content=img_byte_arr, media_type="image/png")


@app.post("/embed")
async def embed_img(file: UploadFile = File(...)):
    max_k = 5

    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    emb = get_embedding(img, device)
    distances, infos = get_nearest_image(emb, max_k)
    distances = distances.tolist()

    return {"infos": infos,
            "dist": distances}


@app.post("/get_img/{celeb}/{id}")
async def embed_img(celeb, id):
    img = Image.open(os.path.join(img_root, celeb, id))

    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    return Response(content=img_byte_arr, media_type="image/png")


if __name__ == '__main__':
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)
