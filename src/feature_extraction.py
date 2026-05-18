import numpy as np
import librosa

def compute_statistics(representation):
    mean_ = np.mean(representation, axis=1)   # średnia amplituda dla każdej częstotliwości
    std_ = np.std(representation, axis=1)     # odchylenie standardowe
    maximum_ = np.max(representation, axis=1) # maksima
    minimum_ = np.min(representation, axis=1) # minima
    median_ = np.median(representation, axis=1) # mediana
    
    statistics = np.concatenate([mean_, std_, maximum_, minimum_, median_])
    return statistics

def feature_extractor(file_path, n_mfcc=20, n_fft=2048, win_length=None, window='hann'):
    mfcc_features_list = []
    spectrogram_features_list = []
    
    for file in file_path:
        audio, sample_rate = librosa.load(file, sr=None)

        # MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft)
        
        # delty
        mfcc_delta = librosa.feature.delta(mfcc) # delta mfcc
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2) # delta-delta mfcc
        mfcc_combined = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0) 

        mfcc_stats = compute_statistics(mfcc_combined)
        mfcc_features_list.append(mfcc_stats)

        # Spektrogram w dB
        stft_ = librosa.stft(audio, n_fft=n_fft, win_length=win_length, window=window)
        spectrogram = librosa.amplitude_to_db(np.abs(stft_), ref=np.max)
        spectrogram_stats = compute_statistics(spectrogram)
        spectrogram_features_list.append(spectrogram_stats)

    # Zamiana list na macierze NumPy
    mfcc_features_array = np.array(mfcc_features_list)
    spectrogram_features_array = np.array(spectrogram_features_list)

    return mfcc_features_array, spectrogram_features_array