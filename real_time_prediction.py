import threading
import time
import sys
import tensorrt as trt
import inference
import vggish_params
import mel_features
import pycuda.driver as cuda
import pycuda.autoinit
import sounddevice as sd
import soundfile as sf
import numpy as np

INPUT_FILENAME = "street_sound.wav"
DURATION = 2 # seconds
BATCH_SIZE = 1
DTYPE = trt.float32
ENGINE_FILENAME = "sonyc_engine.trt"
TAXONOMY_FILENAME = "./dcase-ust-taxonomy.yaml"
SR = vggish_params.SAMPLE_RATE

#%%
# =============================================================================
# Load  taxonomy
# =============================================================================

labels = inference.load_taxonomy(TAXONOMY_FILENAME)

#%%
# =============================================================================
# Real Time audio preprocessing & inference
# =============================================================================

arr = np.zeros((1,2))
event = threading.Event()

# load pre-recorded input file
data, sr = sf.read(INPUT_FILENAME)

def callback(outdata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    global current_frame
    global start_time
    global arr
    if status:
        print(status, file=sys.stderr)
    if start_time == 0:
        start_time = time.currentTime
    chunksize = min(len(data) - current_frame, frames)
    outdata[:chunksize] = data[current_frame:current_frame + chunksize]
    if chunksize < frames:
        outdata[chunksize:] = 0
        raise sd.CallbackStop()
    current_frame += chunksize
    arr = np.concatenate((arr, outdata))
    if time.currentTime - start_time > DURATION:
        start_time = time.currentTime
        # convert to mono
        arr = arr.mean(axis=1)
        with inference.load_engine(ENGINE_FILENAME) as engine:
            log_mel_spec = mel_features.waveform_to_log_mel_spectrogram(y, SR)
            h_input, d_input, h_output, d_output, stream = \
            inference.allocate_buffers(engine, BATCH_SIZE, DTYPE)
            out = inference.do_inference(engine, log_mel_spec[np.newaxis,:,:,np.newaxis], \
            h_input, d_input, h_output, d_output, stream, BATCH_SIZE, profile=False)
        #print("Output matrix size", arr.shape)
        arr = np.zeros((1,2))

    # print("Current Time: {}s".format(time.currentTime))

current_frame = 0
start_time = 0

stream = sd.OutputStream(samplerate=sr, channels=data.shape[1], callback=callback, finished_callback=event.set)

with stream:
    event.wait()  # Wait until playback is finished