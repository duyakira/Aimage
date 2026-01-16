import subprocess
import numpy as np
import gc
from realcugan_ncnn_py import Realcugan
import cv2
import torch

def sharpen_unsharp(img, strength=1.0):
    blur = cv2.GaussianBlur(img, (0,0), 1.0)
    sharpened = cv2.addWeighted(img, 1 + strength, blur, -strength, 0)
    return sharpened

# ========== CONFIG ==========
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)


GPU_ID = 1

_model_cache = {}

def get_model(scale,tilescale,noise):
    if scale not in _model_cache:
        _model_cache[scale] = Realcugan(
            gpuid=1,
            num_threads= 6,
            tilesize=tilescale,
            syncgap=3,
            scale=scale,
            noise=noise,
            model="models-se"
        )
    return _model_cache[scale]

#2x
def upscale_image2x(img_path, scale = 2,noise = 0, tilescale = 256):
    
    model_get = get_model(scale,tilescale,noise)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR

    
    sr = model_get.process_cv2(img)

    del img
    gc.collect()
    if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return sr

def upscale_image2xnoise(img_path, scale = 2,noise = 3,tilescale = 256):
    
    model_get = get_model(scale,tilescale,noise)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR

    
    sr = model_get.process_cv2(img)

    del img
    gc.collect()
    if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return sr

def upscale_image2xsharp(img_path, scale = 2,noise = 0, tilescale = 256):
    
    model_get = get_model(scale,tilescale,noise)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR

    
    sr = model_get.process_cv2(img)
    img = sharpen_unsharp(img,1)
    del img
    gc.collect()
    if torch.cuda.is_available():
          torch.cuda.empty_cache()
    return sr

#3x
def upscale_image3x(img_path, scale = 3,tilescale = 256,noise = 0):
    
    model_get = get_model(scale,tilescale,noise)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR
    sr = model_get.process_cv2(img)

    del img
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return sr

def upscale_image3xnoise(img_path, scale = 3,tilescale = 256,noise = 3):
    
    model_get = get_model(scale,tilescale,noise)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR
    sr = model_get.process_cv2(img)

    del img
    gc.collect()
    if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return sr

def upscale_image3xsharp(img_path, scale = 3,noise = 0, tilescale = 256):
    
    model_get = get_model(scale,tilescale,noise)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR

    
    sr = model_get.process_cv2(img)
    img = sharpen_unsharp(img,1)
    del img
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return sr


def upscale_image4x(img_path, scale = 4,tilescale = 256,noise = 0):
    
    model_get = get_model(scale,tilescale,noise)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR
    sr = model_get.process_cv2(img)

    del img
    gc.collect()
    if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return sr

def upscale_image4xnoise(img_path, scale = 4,tilescale = 256,noise = 3):
    
    model_get = get_model(scale,tilescale,noise)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR
    sr = model_get.process_cv2(img)

    del img
    gc.collect()
    if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return sr
def upscale_image4xsharp(img_path, scale = 4,noise = 0, tilescale = 256):
    
    model_get = get_model(scale,tilescale,noise)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR

    
    sr = model_get.process_cv2(img)
    img = sharpen_unsharp(img,1)
    del img
    gc.collect()
    if torch.cuda.is_available():
             torch.cuda.empty_cache()
    return sr

# ========== VIDEO INFO ==========
def get_video_info(path):
    import json
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "json", path
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(r.stdout)["streams"][0]
    return int(info["width"]), int(info["height"]), info["r_frame_rate"]

def upscale_video2x(input_path,output_path,scale = 2, tilescale = 512,noise =0) :
    width, height, fps = get_video_info(input_path) 
    out_w, out_h = width * scale, height * scale
    decode_cmd = [
    "ffmpeg", "-loglevel", "error",
    "-hwaccel", "none",
    "-i", input_path,
    "-f", "rawvideo",
    "-pix_fmt", "rgb24",
    "-"
]
    

    encode_cmd = [
    "ffmpeg", "-y", "-loglevel", "error",

    "-f", "rawvideo",
    "-pix_fmt", "rgb24",
    "-s", f"{out_w}x{out_h}",
    "-r", fps,
    "-i", "-",

    "-i", input_path,
    "-map", "0:v:0",
    "-map", "1:a?",

    "-c:v", "h264_qsv",
    "-profile:v", "main",
    "-preset", "veryfast",

    "-global_quality", "23",   # thay cho CRF

    "-pix_fmt", "yuv420p",
    "-c:a", "copy",

    output_path
]



    decoder = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE)
    encoder = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE)

    frame_size = width * height * 3
    frame_count = 0

    while True:
        model = get_model(scale,tilescale,noise)
        raw = decoder.stdout.read(frame_size)
        if not raw:
            break

        frame = np.frombuffer(raw, np.uint8).reshape((height, width, 3))
        sr = model.process_cv2(frame)

        encoder.stdin.write(sr.tobytes())
        print("Frame:", frame.shape, "SR:", sr.shape)
        del frame, sr
        frame_count += 1
        print(f"Processed {frame_count} frames")

    decoder.stdout.close()
    encoder.stdin.close()
    decoder.wait()
    encoder.wait()

if __name__ == "__main__":
    upscale_video2x("input.mp4", "output_upscaled.mp4")  

def upscale_video3x(input_path,output_path,scale = 3, tilescale = 256,noise = 0) :
    width, height, fps = get_video_info(input_path) 
    out_w, out_h = width * scale, height * scale
    decode_cmd = [
    "ffmpeg", "-loglevel", "error",
    "-hwaccel", "none",
    "-i", input_path,
    "-f", "rawvideo",
    "-pix_fmt", "rgb24",
    "-"
]
    

    encode_cmd = [
    "ffmpeg", "-y", "-loglevel", "error",

    "-f", "rawvideo",
    "-pix_fmt", "rgb24",
    "-s", f"{out_w}x{out_h}",
    "-r", fps,
    "-i", "-",

    "-i", input_path,
    "-map", "0:v:0",
    "-map", "1:a?",

    "-c:v", "h264_qsv",
    "-profile:v", "main",
    "-preset", "veryfast",

    "-global_quality", "23",   # thay cho CRF

    "-pix_fmt", "yuv420p",
    "-c:a", "copy",

    output_path
]



    decoder = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE)
    encoder = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE)

    frame_size = width * height * 3
    frame_count = 0

    while True:
        model = get_model(scale,tilescale,noise)
        raw = decoder.stdout.read(frame_size)
        if not raw:
            break

        frame = np.frombuffer(raw, np.uint8).reshape((height, width, 3))
        sr = model.process_cv2(frame)

        encoder.stdin.write(sr.tobytes())
        print("Frame:", frame.shape, "SR:", sr.shape)
        del frame, sr
        frame_count += 1
        print(f"Processed {frame_count} frames")

    decoder.stdout.close()
    encoder.stdin.close()
    decoder.wait()
    encoder.wait()

if __name__ == "__main__":
    upscale_video3x("input.mp4", "output_upscaled.mp4")  

def upscale_video4x(input_path,output_path,scale = 4, tilescale = 256, noise = 0) :
    width, height, fps = get_video_info(input_path) 
    out_w, out_h = width * scale, height * scale
    decode_cmd = [
    "ffmpeg", "-loglevel", "error",
    "-hwaccel", "none",
    "-i", input_path,
    "-f", "rawvideo",
    "-pix_fmt", "rgb24",
    "-"
]
    

    encode_cmd = [
    "ffmpeg", "-y", "-loglevel", "error",

    "-f", "rawvideo",
    "-pix_fmt", "rgb24",
    "-s", f"{out_w}x{out_h}",
    "-r", fps,
    "-i", "-",

    "-i", input_path,
    "-map", "0:v:0",
    "-map", "1:a?",

    "-c:v", "h264_qsv",
    "-profile:v", "main",
    "-preset", "veryfast",

    "-global_quality", "23",   # thay cho CRF

    "-pix_fmt", "yuv420p",
    "-c:a", "copy",

    output_path
]



    decoder = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE)
    encoder = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE)

    frame_size = width * height * 3
    frame_count = 0

    while True:
        model = get_model(scale,tilescale,noise)
        raw = decoder.stdout.read(frame_size)
        if not raw:
            break

        frame = np.frombuffer(raw, np.uint8).reshape((height, width, 3))
        sr = model.process_cv2(frame)

        encoder.stdin.write(sr.tobytes())
        print("Frame:", frame.shape, "SR:", sr.shape)
        del frame, sr
        frame_count += 1
        print(f"Processed {frame_count} frames")

    decoder.stdout.close()
    encoder.stdin.close()
    decoder.wait()
    encoder.wait()
    

# ========== RUN ==========
if __name__ == "__main__":
    upscale_video4x("input.mp4", "output_upscaled.mp4") 



