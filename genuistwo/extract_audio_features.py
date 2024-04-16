import os
import time
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
from genuistwo.utils import load_audio_file

# import dask
# from dask.distributed import LocalCluster

# cluster = LocalCluster()
# client = cluster.get_client()

itunes_file = "Library.xml"


CHUNK_SIZE_SECONDS = 30


def extract_features(data, sr):
  """
  Extracts temporal and spectral features from audio data.

  Args:
      data (np.array): Audio data as a NumPy array.
      sr (int): Sampling rate of the audio data.

  Returns:
      list: A list of extracted features.
  """

  features = {}

  # --- Temporal Features ---

  # 1. Root Mean Square Energy: Indicates the average energy/loudness of the signal.
  rmse = librosa.feature.rms(y=data)[0]
  features['rmse_mean'] = np.mean(rmse)
  features['rmse_std'] = np.std(rmse)

  # 2. Zero Crossing Rate: Indicates the number of times the signal crosses the zero axis.
  zcr = librosa.feature.zero_crossing_rate(data)[0]
  features['zcr_mean'] = np.mean(zcr)
  features['zcr_std'] = np.std(zcr)

  # 3. Tempo: Estimate the tempo of the track
  tempo, _ = librosa.beat.beat_track(y=data, sr=sr)
  features['tempo'] = tempo

  # --- Spectral Features ---

  # 4. Spectral Centroid: Represents the center of mass of the spectrum.
  spectral_centroid = librosa.feature.spectral_centroid(y=data, sr=sr)[0]
  features['spectral_centroid_mean'] = np.mean(spectral_centroid)
  features['spectral_centroid_std'] = np.std(spectral_centroid)

  # 5. Spectral Bandwidth: Measures the spread of the spectrum around the centroid.
  spectral_bandwidth = librosa.feature.spectral_bandwidth(y=data, sr=sr)[0]
  features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
  features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)

  # 6. Spectral Rolloff: Frequency below which a specified percentage of the total spectral energy lies.
  spectral_rolloff = librosa.feature.spectral_rolloff(y=data, sr=sr)[0]
  features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
  features['spectral_rolloff_std'] = np.std(spectral_rolloff)

  # 7. Mel-Frequency Cepstral Coefficients (MFCCs): Represents the short-term power spectrum of a sound.
  mfccs = librosa.feature.mfcc(y=data, sr=sr)
  for i in range(mfccs.shape[0]):
      features[f'mfcc_{i}'] = np.mean(mfccs[i])

  # 8. Chroma Features: Capture harmonic and melodic characteristics.
  chroma_stft = librosa.feature.chroma_stft(y=data, sr=sr)
  for i in range(chroma_stft.shape[0]):
      features[f'chroma_stft_{i}'] = np.mean(chroma_stft[i])

  return features

def process_audio_file(file_path, song_id, chunk=False):
    '''
    Process an audio file by extracting features from 30-second chunks in parallel.
    '''
    data, sr = load_audio_file(file_path)

    if data is None or sr is None:
        return []

    if chunk:
        # Efficient Chunking with NumPy array slicing
        chunk_size_samples = CHUNK_SIZE_SECONDS * sr
        chunks = [data[i: i + chunk_size_samples] for i in range(0, len(data), chunk_size_samples)]
        features = [extract_features_with_metadata(chunk, sr, i, song_id) for i, chunk in enumerate(chunks)]

    else:
        features = [extract_features_with_metadata(data, sr, 0, song_id)]

    return features

def extract_features_with_metadata(chunk, sr, chunk_index, song_id):
    feature = extract_features(chunk, sr)
    feature["chunk_index"] = chunk_index
    feature["song_id"] = song_id
    feature["start_time"] = (chunk_index * CHUNK_SIZE_SECONDS) 
    feature["end_time"] = ((chunk_index + 1) * CHUNK_SIZE_SECONDS)
    return feature 

def main():
    start_time = time.time()

    # Load the iTunes library
    df = pd.read_csv("Library.csv")

    print('There are {} songs in the library.'.format(len(df)))

    features_list = []

    for row in tqdm(df.itertuples(), total=len(df)):
       features_list.append(process_audio_file(row.Location, row.id, chunk=False))

    # Parallel Processing with multiprocessing Pool
    # features_list = dask.compute(*[
    #     process_audio_file(row.Location, row.id, chunk=True) for _, row in df.itertuples()
    # ])

    # features_list = []
    # for _, row in df.itertuples():
    #     features = client.submit(process_audio_file, row.Location, row.id, chunk=False)
    #     features_list.append(features)

    # features_list = client.gather(features_list)

    # Flatten and save features
    features = [item for sublist in features_list for item in sublist]
    features_df = pd.DataFrame.from_records(features)
    features_df.set_index("song_id", inplace=True)
    features_df.to_csv("audio_features.csv")

    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")


if __name__ == "__main__":

    main()

    