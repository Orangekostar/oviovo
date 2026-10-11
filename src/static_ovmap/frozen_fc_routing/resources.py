"""One shared queued GPU; no changes to global CUDA/NVML installations."""
import contextlib
import fcntl

@contextlib.contextmanager
def queued_gpu_lock(path):
    with open(path,'a+') as stream:
        print('WAIT_GPU_LOCK',path,flush=True);fcntl.flock(stream,fcntl.LOCK_EX)
        try:
            print('GPU_LOCK_ACQUIRED',flush=True);yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)
