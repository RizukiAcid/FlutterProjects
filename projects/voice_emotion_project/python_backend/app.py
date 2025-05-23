import os
import flask
from flask import Flask, request, jsonify
from flask_cors import CORS # For handling Cross-Origin Resource Sharing
import librosa
import numpy as np
import joblib
import soundfile as sf # To explicitly handle audio loading if librosa needs it

app = Flask(__name__)
CORS(app) # This will allow your Flutter web app to call the backend

# --- COPIED FROM IndoWaveSentiment_DataPrep.ipynb ---
def extract_features(audio_data, sample_rate):
    # Zero Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio_data) # Ensure 'y=' for clarity

    # Energy (RMS)
    energy = librosa.feature.rms(y=audio_data)

    # Spectral Centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]

    # Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]

    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)

    # Mel Spectrogram (We just need its mean for a feature if included, or skip if not one of the 45)
    # For this example, assuming the 45 features are calculated as in the notebook.
    # The notebook does extract 'mel1', 'mel2', 'mel3' as np.mean(mel[0..2])
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)


    # Contrast
    # S = np.abs(librosa.stft(audio_data)) # Use 'y=' for stft as well
    # Re-calculate S for spectral_contrast
    S_contrast = np.abs(librosa.stft(y=audio_data))
    contrast = librosa.feature.spectral_contrast(S=S_contrast, sr=sample_rate)


    # Tonnetz
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio_data), sr=sample_rate)

    features = {
        'zero_crossing_rate': np.mean(zero_crossing_rate),
        'energy': np.mean(energy),
        'spectral_centroid': np.mean(spectral_centroids),
        'spectral_rolloff': np.mean(spectral_rolloff),
        'mfcc1': np.mean(mfccs[0]), 'mfcc2': np.mean(mfccs[1]), 'mfcc3': np.mean(mfccs[2]),
        'mfcc4': np.mean(mfccs[3]), 'mfcc5': np.mean(mfccs[4]), 'mfcc6': np.mean(mfccs[5]),
        'mfcc7': np.mean(mfccs[6]), 'mfcc8': np.mean(mfccs[7]), 'mfcc9': np.mean(mfccs[8]),
        'mfcc10': np.mean(mfccs[9]), 'mfcc11': np.mean(mfccs[10]), 'mfcc12': np.mean(mfccs[11]),
        'mfcc13': np.mean(mfccs[12]),
        'chroma1': np.mean(chroma[0]), 'chroma2': np.mean(chroma[1]), 'chroma3': np.mean(chroma[2]),
        'chroma4': np.mean(chroma[3]), 'chroma5': np.mean(chroma[4]), 'chroma6': np.mean(chroma[5]),
        'chroma7': np.mean(chroma[6]), 'chroma8': np.mean(chroma[7]), 'chroma9': np.mean(chroma[8]),
        'chroma10': np.mean(chroma[9]), 'chroma11': np.mean(chroma[10]), 'chroma12': np.mean(chroma[11]),
        'mel1': np.mean(mel_spectrogram[0]), 'mel2': np.mean(mel_spectrogram[1]), 'mel3': np.mean(mel_spectrogram[2]), # As per notebook's features_df
        'contrast1': np.mean(contrast[0]), 'contrast2': np.mean(contrast[1]), 'contrast3': np.mean(contrast[2]),
        'contrast4': np.mean(contrast[3]), 'contrast5': np.mean(contrast[4]), 'contrast6': np.mean(contrast[5]),
        'contrast7': np.mean(contrast[6]),
        'tonnetz1': np.mean(tonnetz[0]), 'tonnetz2': np.mean(tonnetz[1]), 'tonnetz3': np.mean(tonnetz[2]),
        'tonnetz4': np.mean(tonnetz[3]), 'tonnetz5': np.mean(tonnetz[4]), 'tonnetz6': np.mean(tonnetz[5]),
    }
    # Return as a list in the same order as your training data columns (X_train.columns)
    # This order is crucial and must match the columns in ExtractedData.csv, excluding 'emotion'
    # Refer to the columns in your df from IndoWaveSentiment_MachineLearning.ipynb
    # Or better: load X_train.columns from a saved list or ensure this order.
    # For now, let's assume this is the correct order based on common feature extraction.
    # You MUST verify this order against your `X_train.columns` from the ML notebook.
    feature_vector = [
        features['zero_crossing_rate'], features['energy'], features['spectral_centroid'], features['spectral_rolloff'],
        features['mfcc1'], features['mfcc2'], features['mfcc3'], features['mfcc4'], features['mfcc5'],
        features['mfcc6'], features['mfcc7'], features['mfcc8'], features['mfcc9'], features['mfcc10'],
        features['mfcc11'], features['mfcc12'], features['mfcc13'],
        features['chroma1'], features['chroma2'], features['chroma3'], features['chroma4'], features['chroma5'],
        features['chroma6'], features['chroma7'], features['chroma8'], features['chroma9'], features['chroma10'],
        features['chroma11'], features['chroma12'],
        features['mel1'], features['mel2'], features['mel3'], # These were added in IndoWaveSentiment_DataPrep.ipynb
        features['contrast1'], features['contrast2'], features['contrast3'], features['contrast4'],
        features['contrast5'], features['contrast6'], features['contrast7'],
        features['tonnetz1'], features['tonnetz2'], features['tonnetz3'], features['tonnetz4'],
        features['tonnetz5'], features['tonnetz6']
    ]
    return feature_vector
# --- END OF COPIED FUNCTION ---

# Load the trained model and label encoder
# Use 'r' before the string to make it a raw string
model = joblib.load(r'D:\Repositories\FlutterProjects\projects\voice_emotion_project\python_backend\random_forest_emotion_model.pkl')
label_encoder = joblib.load(r'D:\Repositories\FlutterProjects\projects\voice_emotion_project\python_backend\label_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['file']

    # Save the file temporarily (optional, could process in memory)
    # temp_audio_path = "temp_audio.wav" # You might need a proper temp file handling
    # audio_file.save(temp_audio_path)

    try:
        # Load audio data using soundfile (more robust for web uploads)
        # Or librosa directly if it works with the BytesIO stream
        audio_data, sample_rate = sf.read(audio_file) # sf.read returns data and samplerate
        # If stereo, convert to mono:
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Extract features
        # Librosa expects a NumPy array, ensure audio_data is in that format.
        # sf.read already returns a NumPy array.
        features_vector = extract_features(audio_data, sample_rate)
        features_2d = np.array(features_vector).reshape(1, -1) # Reshape for single prediction

        # Make prediction
        prediction_encoded = model.predict(features_2d)
        predicted_emotion = label_encoder.inverse_transform(prediction_encoded)[0] # Get the string label

        return jsonify({'emotion': predicted_emotion})

    except Exception as e:
        app.logger.error(f"Error processing file: {e}") # Log the error server-side
        # Add traceback for more details:
        import traceback
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    # finally:
        # if os.path.exists(temp_audio_path):
        #     os.remove(temp_audio_path) # Clean up temp file

if __name__ == '__main__':
    app.run(debug=True, port=5000) # Runs on http://127.0.0.1:5000