import queue
import threading
import torch
import gc
import traceback
import subprocess
import os
task_queue = queue.Queue(maxsize=40)
FFMPEG_PATH = r"C:\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"
def merge_audio(video_no_audio, video_with_audio, output_path):
    cmd = [
        FFMPEG_PATH, "-y",
        "-loglevel", "error",
        "-stats",
        "-i", video_no_audio,
        "-i", video_with_audio,

        "-map", "0:v:0",
        "-map", "1:a:0?",     # dấu ? = không có audio cũng không crash

        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",

        "-movflags", "+faststart",
        output_path
    ]
    subprocess.run(cmd, check=True)
    return output_path

def convert_video_gpu(input_path, output_path):
    cmd = [
        FFMPEG_PATH, "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        output_path
    ]

    subprocess.run(cmd, check=True)
    return output_path

def worker():
    while True:
        func, args = task_queue.get()
        input_path,gpu_path, ai_path, output_path = args

        try:
            gpu_lock = threading.Lock()

            with gpu_lock:
                 convert_video_gpu(input_path, gpu_path)

                 func(gpu_path, ai_path)

                 merge_audio(ai_path, gpu_path, output_path)

                 open(output_path + ".done", "w").close()

        except Exception:
            print("❌ Worker error:")
            traceback.print_exc()

        finally:
            for p in (ai_path, gpu_path):
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except:
                        pass

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            task_queue.task_done()


for _ in range(40):
    threading.Thread(target=worker, daemon=True).start()



