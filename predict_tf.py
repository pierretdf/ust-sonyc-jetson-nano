"""
RUNNING INFERENCE WITH KERAS API DOESN'T WORKS
"""
import os
from tensorflow import keras
import inference
import numpy as np

model_architecture_file = 'model_architecture.json'
model_weights_file = 'best_model_weights.h5'

#%%
# =============================================================================
# Load  taxonomy
# =============================================================================

taxonomy_file =  './dcase-ust-taxonomy.yaml'
labels = inference.load_taxonomy(taxonomy_file)

#%%
# =============================================================================
# Audio preprocessing
# =============================================================================

audio_sample_wav = "00_010587.wav"
log_mel_input = inference.audio_preprocessing(audio_sample_wav)

#%%
# =============================================================================
# Load model
# =============================================================================

# Model reconstruction from JSON file
with open(model_architecture_file, 'r') as f:
    model = keras.models.model_from_json(f.read())

# Load weights into the new model
model.load_weights(model_weights_file)


#%%
# =============================================================================
# Inference pipeline
# =============================================================================
y_pred = np.squeeze(model.predict(log_mel_input, batch_size=1, verbose=1))

for ind, label in enumerate(labels):
    print(label + ": {0:.0%}".format(y_pred[ind]))