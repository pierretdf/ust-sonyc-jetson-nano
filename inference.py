import os
import oyaml as yaml
import tensorrt as trt
import vggish_params
import mel_features
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from scipy.io import wavfile
import scipy.signal as sps
import soundfile as sf
import sounddevice as sd

CLASSES = 8
SR = vggish_params.SAMPLE_RATE

def allocate_buffers(engine, batch_size, data_type):
    """
    This is the function to allocate buffers for input and output in the device
    Args:
        engine : The path to the TensorRT engine.
        batch_size : The batch size for execution time.
        data_type: The type of the data for input and output, for example trt.float32.

    Output:
        h_input_1: Input in the host.
        d_input_1: Input in the device.
        h_output_1: Output in the host.
        d_output_1: Output in the device.
        stream: CUDA stream.
    """

    # Determine dimensions and create page-locked memory buffers (which won't be swapped to disk) to hold host inputs/outputs.
    h_input_1 = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(data_type))
    h_output = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(data_type))
    # Allocate device memory for inputs and outputs.
    d_input_1 = cuda.mem_alloc(h_input_1.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()

    return h_input_1, d_input_1, h_output, d_output, stream

def load_audio_to_buffer(log_mel, pagelocked_buffer):
   preprocessed = np.asarray(log_mel).ravel()
   np.copyto(pagelocked_buffer, preprocessed)

def do_inference(engine, wav_1, h_input_1, d_input_1, h_output, d_output, stream, batch_size=1, profile=True):
    """
    This is the function to run the inference
    Args:
        engine : Path to the TensorRT engine
        wav_1 : Input audio sample to the model.
        h_input_1: Input in the host
        d_input_1: Input in the device
        h_output_1: Output in the host
        d_output_1: Output in the device
        stream: CUDA stream
        batch_size : Batch size for execution time

    Output:
        The list of class predictions
    """
    start = cuda.Event()
    end = cuda.Event()

    load_audio_to_buffer(wav_1, h_input_1)
    with engine.create_execution_context() as context:
        # Start profiling
        start.record()

        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(d_input_1, h_input_1, stream)

        # Allow verbose
        if profile:
            context.profiler = trt.Profiler()
        # Run inference.
        context.execute_async_v2(bindings=[int(d_input_1), int(d_output)], stream_handle=stream.handle)

        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(h_output, d_output, stream)

        # Synchronize the stream
        stream.synchronize()

        # End profiling
        end.record()
        end.synchronize()
        secs = start.time_till(end)
        print("Inference time elapsed: {0:.3f}ms \n".format(secs))

        # Return the host output.
        out = h_output.reshape((batch_size, CLASSES))
        return out

def load_engine(engine_file_path):
    # Deserialize the TensorRT engine from specified plan file.
    trt.init_libnvinfer_plugins(None, "")
    assert os.path.exists(engine_file_path)
    print("Reading engine from {} \n ".format(engine_file_path))
    with open(engine_file_path, 'rb') as f, trt.Runtime(trt.Logger()) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def audio_preprocessing(audio_file):
    # Compute log-Mel spectrogram from input audio file
    print("Reading input audio from file {}".format(audio_file))
    old_sr, y = wavfile.read(audio_file)
    # convert to mono if input audio file is stereo
    if len(y.shape) > 1:
        if y.shape[1] == 2:
            y = y.mean(axis=1)
    print("DEBUG: input shape", y.shape)
    # resampling
    if SR != old_sr:
        number_of_samples = round(len(y) * float(SR) / old_sr)
        y = sps.resample(y, number_of_samples)

    print("DEBUG: input shape after resampling", y.shape)
    log_mel_spec = mel_features.waveform_to_log_mel_spectrogram(y, SR)
    return log_mel_spec[np.newaxis,:,:,np.newaxis]

def load_taxonomy(taxonomy_file):
    with open(taxonomy_file, 'r') as f:
        taxonomy = yaml.load(f, Loader=yaml.Loader)

    # Get list of coarse labels from taxonomy
    return [v for k,v in taxonomy['coarse'].items()]
