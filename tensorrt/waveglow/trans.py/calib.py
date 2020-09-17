# from tensorrt sample code-INT8 MNIST Caffe
import tensorrt as trt
import torch
import os

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import json

class WaveGlowEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, batch_data_dir, cache_file, C, W):
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = cache_file
        self.host_input = cuda.pagelocked_empty(C * W, np.float32)
        self.device_input = cuda.mem_alloc(C * W * trt.float32.itemsize)
        self.batch_files = [os.path.join(batch_data_dir, f) for f in os.listdir(batch_data_dir)]

        def load_batches(): # to make it iterable
            for f in self.batch_files:
                cond = torch.load(f).numpy()
                quite = np.zeros((C, W)); l = min(W, cond.shape[1])
                quite[:, :l] = cond[:, :l]; cond = quite.flatten()
                yield cond
        self.batches = load_batches()

    def get_batch_size(self):
        return 1 # fixed as no gain on batching temporally and latency is crucial

    def get_batch(self, names):
        try:
            data = next(self.batches)
            np.copyto(self.host_input, data)
            cuda.memcpy_htod(self.device_input, self.host_input)
            return [int(self.device_input)]
        except StopIteration:
            # When we're out of batches, we return either [] or None.
            # Signals TensorRT that there is no calibration data remaining.
            return None

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

