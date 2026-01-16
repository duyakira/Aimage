import queue
import threading
import torch
import gc
import traceback
import os
task_queue = queue.Queue(maxsize=40)

FFMPEG_PATH = r"C:\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"

def worker():
    while True:
        func, args = task_queue.get()
        input_path, output_path = args

        try:
            gpu_lock = threading.Lock()

            with gpu_lock:
                 func(input_path, output_path)
                 open(output_path + ".done", "w").close()

        except Exception:
            print("❌ Worker error:")
            traceback.print_exc()

        finally:
            os.remove(input_path)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        task_queue.task_done()


for _ in range(40):
    threading.Thread(target=worker, daemon=True).start()




