from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
import uuid
from queue_worker import task_queue
import cv2
from realcugan import upscale_image2x, upscale_image3x, upscale_image4x,upscale_image2xnoise, upscale_image3xnoise, upscale_image4xnoise,upscale_image2xsharp, upscale_image3xsharp, upscale_image4xsharp
from realcugan import upscale_video2x,upscale_video4x, upscale_video3x
import gc
import torch

app = FastAPI()
CHUNK = 1024*1024

BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")



FFMPEG_PATH = r"C:\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"

os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="../frontend/static"), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    with open("../frontend/templates/index.html", "r", encoding="utf-8") as f:
        return f.read()
    

@app.post("/api/upscale2x")
async def upscale(file: UploadFile = File(...)):
    uid = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{uid}_in.png")
    output_path = os.path.join(UPLOAD_DIR, f"{uid}_out.png")

    try:
        # lưu file
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        # xử lý AI
        sr = upscale_image2x(input_path)
        cv2.imwrite(output_path, sr)

        return {"result": f"/result/{uid}"}

    finally : 
        if os.path.exists(input_path):
           os.remove(input_path)
        if sr is not None:
           del sr

           
        gc.collect()
        if torch.cuda.is_available(): 
            torch.cuda.empty_cache()

@app.post("/api/upscale2xnoise")
async def upscale(file: UploadFile = File(...)):
    uid = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{uid}_in.png")
    output_path = os.path.join(UPLOAD_DIR, f"{uid}_out.png")

    try:
        # lưu file
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        # xử lý AI
        sr = upscale_image2xnoise(input_path)
        cv2.imwrite(output_path, sr)

        return {"result": f"/result/{uid}"}

    finally : 
        if os.path.exists(input_path):
           os.remove(input_path)
        if sr is not None:
           del sr

           
        gc.collect()
        if torch.cuda.is_available(): 
            torch.cuda.empty_cache()

@app.post("/api/upscale2xsharp")
async def upscale(file: UploadFile = File(...)):
    uid = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{uid}_in.png")
    output_path = os.path.join(UPLOAD_DIR, f"{uid}_out.png")

    try:
        # lưu file
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        # xử lý AI
        sr = upscale_image2xsharp(input_path)
        cv2.imwrite(output_path, sr)

        return {"result": f"/result/{uid}"}

    finally : 
        if os.path.exists(input_path):
           os.remove(input_path)
        if sr is not None:
           del sr

           
        gc.collect()
        if torch.cuda.is_available(): 
            torch.cuda.empty_cache()


@app.post("/api/upscale3x")
async def upscale(file: UploadFile = File(...)):
    uid = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{uid}_in.png")
    output_path = os.path.join(UPLOAD_DIR, f"{uid}_out.png")

    try:
        # lưu file
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # xử lý AI
        sr = upscale_image3x(input_path)
        cv2.imwrite(output_path, sr)
        return {"result": f"/result/{uid}"}
       

    finally : 
        if os.path.exists(input_path):
           os.remove(input_path)
        
        del sr
        gc.collect()
        if torch.cuda.is_available(): 
            torch.cuda.empty_cache()

@app.post("/api/upscale3xnoise")
async def upscale(file: UploadFile = File(...)):
    uid = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{uid}_in.png")
    output_path = os.path.join(UPLOAD_DIR, f"{uid}_out.png")

    try:
        # lưu file
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # xử lý AI
        sr = upscale_image3xnoise(input_path)
        cv2.imwrite(output_path, sr)
        return {"result": f"/result/{uid}"}
       

    finally : 
        if os.path.exists(input_path):
           os.remove(input_path)
        
        del sr
        gc.collect()
        if torch.cuda.is_available(): 
            torch.cuda.empty_cache()

@app.post("/api/upscale3xsharp")
async def upscale(file: UploadFile = File(...)):
    uid = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{uid}_in.png")
    output_path = os.path.join(UPLOAD_DIR, f"{uid}_out.png")

    try:
        # lưu file
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        # xử lý AI
        sr = upscale_image3xsharp(input_path)
        cv2.imwrite(output_path, sr)

        return {"result": f"/result/{uid}"}

    finally : 
        if os.path.exists(input_path):
           os.remove(input_path)
        if sr is not None:
           del sr

           
        gc.collect()
        if torch.cuda.is_available(): 
            torch.cuda.empty_cache()



@app.post("/api/upscale4x")
async def upscale(file: UploadFile = File(...)):
    uid = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{uid}_in.png")
    output_path = os.path.join(UPLOAD_DIR, f"{uid}_out.png")

    try:
        # lưu file
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # xử lý AI
        sr = upscale_image4x(input_path)
        cv2.imwrite(output_path, sr)
        return {"result": f"/result/{uid}"}
       

    finally : 
        if os.path.exists(input_path):
           os.remove(input_path)
        
        del sr
        gc.collect()
        if torch.cuda.is_available(): 
            torch.cuda.empty_cache()

@app.post("/api/upscale4xnoise")
async def upscale(file: UploadFile = File(...)):
    uid = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{uid}_in.png")
    output_path = os.path.join(UPLOAD_DIR, f"{uid}_out.png")

    try:
        # lưu file
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # xử lý AI
        sr = upscale_image4xnoise(input_path)
        cv2.imwrite(output_path, sr)
        return {"result": f"/result/{uid}"}
       

    finally : 
        if os.path.exists(input_path):
           os.remove(input_path)
        
        del sr
        gc.collect()
        if torch.cuda.is_available(): 
            torch.cuda.empty_cache()

@app.post("/api/upscale4xsharp")
async def upscale(file: UploadFile = File(...)):
    uid = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{uid}_in.png")
    output_path = os.path.join(UPLOAD_DIR, f"{uid}_out.png")

    try:
        # lưu file
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        # xử lý AI
        sr = upscale_image4xsharp(input_path)
        cv2.imwrite(output_path, sr)

        return {"result": f"/result/{uid}"}

    finally : 
        if os.path.exists(input_path):
           os.remove(input_path)
        if sr is not None:
           del sr

           
        gc.collect()
        if torch.cuda.is_available(): 
            torch.cuda.empty_cache()


@app.get("/result/{uid}")
def get_result(uid: str):
    path = f"{UPLOAD_DIR}/{uid}_out.png"
    if os.path.exists(path):
       return FileResponse(path, media_type="image/png")

    
BASE_DIR_2 = os.path.dirname(__file__)
UPLOAD_DIR_2 = os.path.join(BASE_DIR_2, "uploads")
os.makedirs(UPLOAD_DIR_2, exist_ok=True)

app.mount("/static", StaticFiles(directory="../frontend/static"), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    with open("../frontend/templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

#2xVideo 
@app.post("/api/upscaleVideo2x")
async def upscaleVideo(file: UploadFile = File(...)):
    uid = str(uuid.uuid4())

    input_path  = os.path.join(UPLOAD_DIR_2, f"{uid}_in.mp4")
    output_path = os.path.join(UPLOAD_DIR_2, f"{uid}_out.mp4")

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f, 1024*1024)

    # enqueue full pipeline
    task_queue.put((
        upscale_video2x,
        (input_path,
        output_path)
    ))

    return {"task_id": uid, "status": "queued"}

#3xVideo
@app.post("/api/upscaleVideo3x")
async def upscaleVideo(file: UploadFile = File(...)):
    uid = str(uuid.uuid4())

    input_path  = os.path.join(UPLOAD_DIR_2, f"{uid}_in.mp4")
    output_path = os.path.join(UPLOAD_DIR_2, f"{uid}_out.mp4")

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f, 1024*1024)

    # enqueue full pipeline
    task_queue.put((
        upscale_video3x,
        (input_path,
        output_path)
    ))

    return {"task_id": uid, "status": "queued"}


@app.post("/api/upscaleVideo4x")
async def upscaleVideo(file: UploadFile = File(...)):
    uid = str(uuid.uuid4())

    input_path  = os.path.join(UPLOAD_DIR_2, f"{uid}_in.mp4")
    output_path = os.path.join(UPLOAD_DIR_2, f"{uid}_out.mp4")

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f, 1024*1024)

    # enqueue full pipeline
    task_queue.put((
        upscale_video4x,
        (input_path,
        output_path)
    ))

    return {"task_id": uid, "status": "queued"}

    
@app.get("/api/status/{task_id}") 
def check_status(task_id: str): 
        path = f"{UPLOAD_DIR_2}/{task_id}_out.mp4" 
        if os.path.exists(path + ".done"): 
            return { "status": "done", "url": f"/api/download/{task_id}" } 
        return {"status": "processing"} 

@app.get("/api/download/{task_id}") 
def download(task_id: str): 
    return FileResponse( f"{UPLOAD_DIR_2}/{task_id}_out.mp4", media_type="video/mp4" )
