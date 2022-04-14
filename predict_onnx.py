import threading
import time
import tensorrt as trt
import inference
import pycuda.driver as cuda
import pycuda.autoinit

BATCH_SIZE = 1
DTYPE = trt.float32
ENGINE_FILENAME = "sonyc_engine_fp16.trt"
TAXONOMY_FILENAME = "./dcase-ust-taxonomy.yaml"

#%%
# =============================================================================
# Load  taxonomy
# =============================================================================

labels = inference.load_taxonomy(TAXONOMY_FILENAME)

#%%
# =============================================================================
# Audio preprocessing
# =============================================================================

audio_sample_wav = ["engine_truck_trimmed.wav", "00_010587.wav"]
log_mel_input = []
for e in audio_sample_wav:
    log_mel_input.append(inference.audio_preprocessing(e))


#%%
# =============================================================================
# Load ONNX model and run inference
# =============================================================================
