import threading
import time
import tensorrt as trt
import inference
import pycuda.driver as cuda
import pycuda.autoinit

BATCH_SIZE = 1
DTYPE = trt.float32
ENGINE_FILENAME = "sonyc_engine.trt"
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

audio_sample_wav = "00_010587.wav"
log_mel_input = inference.audio_preprocessing(audio_sample_wav)

#%%
# =============================================================================
# Load TensorRT engine and run inference
# =============================================================================

# m = threading.Thread(target=inference.do_inference(engine, log_mel_input, h_input, d_input, h_output, d_output, stream, BATCH_SIZE))
# m.start()

start_time = time.time()
with inference.load_engine(ENGINE_FILENAME) as engine:
    h_input, d_input, h_output, d_output, stream = \
    inference.allocate_buffers(engine, BATCH_SIZE, DTYPE)
    out = inference.do_inference(engine, log_mel_input, h_input, \
    d_input, h_output, d_output, stream, BATCH_SIZE, profile=False)
print("Total time elapsed with engine loading: %s s" % (time.time() - start_time))

#%%
# =============================================================================
# Inference results
# =============================================================================

for ind, label in enumerate(labels):
    print(label + ": {0:.1%}".format(out[0][ind]))