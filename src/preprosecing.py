import os
import numpy as np
import librosa
import soundfile as sf


import statistics as st
from joblib import load
from pydub import AudioSegment

CLASSES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

class Features:
    def __init__(self, y, sr, hop_length=5000):
        """
        Initialize the class with audio signal, sr and hop_length
        :param y: audio signal
        :param sr: sample rate of audio signal
        :param hop_length: hop_length  parameter used while calculating the chroma_stft feature
        """
        self.y = np.split(y, 10)
        self.sr = sr
        self.hop_length = hop_length

    def get_mean_var(self, y):
        """
        Helper function to get mean and variance of feature
        :param y: audio feature
        :return: mean, variance
        """
        mean = y.mean()
        var = y.var()
        return mean, var

    def zero_crossing_rate(self, y):
        """
        Returns the zero-crossing rate of the audio signal
        :return: mean and variance of zero-crossing rate
        """
        values = librosa.feature.zero_crossing_rate(y)
        return self.get_mean_var(values)

    def harmonic_and_per(self, y):
        """
        separates the harmonic and percussive components of the audio signal
        :return: harmonic and percussive components' mean and variance
        """
        y_harm, y_perc  = librosa.effects.hpss(y)
        harm = self.get_mean_var(y_harm)
        perc = self.get_mean_var(y_perc)
        return harm, perc
    
  
    def tempo(self, y):
            """
            Extracts the tempo (beats per minute) of an audio signal.
            
            Parameters:
                y (ndarray): The audio signal represented as an numpy array.
            
            Returns:
                float: The tempo of the audio signal in beats per minute.
            """
            tempo = librosa.beat.tempo(y, sr=self.sr)
            return tempo

    def centroid(self, y):
            """
            Extracts the spectral centroid of an audio signal.
            
            Parameters:
                y (ndarray): The audio signal represented as an numpy array.
            
            Returns:
                tuple: A tuple containing the mean and variance of the spectral centroid.
            """
            centroid = librosa.feature.spectral_centroid(y, sr=self.sr)
            return self.get_mean_var(centroid)
    

    
    def mfccs(self, y):
        """
        Extracts the Mel-Frequency Cepstral Coefficients (MFCCs) of an audio signal.
        
        Parameters:
            y (ndarray): The audio signal represented as an numpy array.
        
        Returns:
            ndarray: An array containing the mean and variance of the MFCCs.
        """
        mfccs = librosa.feature.mfcc(y, sr=self.sr)
        mean = mfccs.mean(axis=1)
        var = mfccs.var(axis=1)
        values = [[mean[i], var[i]] for i in range(mean.shape[0])]
        return np.array(values).reshape(-1)
    
    def chroma_stft(self, y):
        """
        Extracts the chroma feature of an audio signal.
        
        Parameters:
            y (ndarray): The audio signal represented as an numpy array.
        
        Returns:
            tuple: A tuple containing the mean and variance of the chroma feature.
        """
        chroma = librosa.feature.chroma_stft(y, sr=self.sr, hop_length=self.hop_length)
        return self.get_mean_var(chroma)
    
    def spectral_bandwidth(self, y):
        """
        Extracts the spectral bandwidth of an audio signal.
        
        Parameters:
            y (ndarray): The audio signal represented as an numpy array.
        
        Returns:
            tuple: A tuple containing the mean and variance of the spectral bandwidth.
        """
        spd = librosa.feature.spectral_bandwidth(y,sr=self.sr )
        return self.get_mean_var(spd)
    
    def rollof(self, y):
        """
        Extracts the spectral rolloff of an audio signal.
        
        Parameters:
            y (ndarray): The audio signal represented as an numpy array.
        
        Returns:
            tuple: A tuple containing the mean and variance of the spectral rolloff.
        """
        rollof = librosa.feature.spectral_rolloff(y=y, sr=self.sr)[0]
        return self.get_mean_var(rollof)
    
    def rms(self, y):
        """
        Extracts the root mean square (RMS) of an audio signal.
        
        Parameters:
            y (ndarray): The audio signal represented as an numpy array.
        
        Returns:
            tuple: A tuple containing the mean and variance of the RMS.
        """
        rms = librosa.feature.rms(y=y)
        return self.get_mean_var(rms)
    
    def features(self,y):
        """
        Extracts various audio features from an audio signal.
        
        Parameters:
            y (ndarray): The audio signal represented as an numpy array.
        
        Returns:
            ndarray: An array containing the extracted audio features.
        """
        tempo = self.tempo(y)
        centroid_mean, centroid_var = self.centroid(y)
        chroma_mean, chroma_var = self.chroma_stft(y)
        zcr_mean, zcr_var = self.zero_crossing_rate(y)
        spd_mean, spd_var = self.spectral_bandwidth(y)
        rollof_mean, rollof_var = self.rollof(y)
        rsm_mean, rsm_var = self.rms(y)
        harm, perc = self.harmonic_and_per(y)
        harm_mean, harm_var = harm
        perc_mean, perc_var = perc
        mfccs = self.mfccs(y)
        
        features = np.array([y.shape[0],
                            chroma_mean, chroma_var,
                            rsm_mean, rsm_var,
                            centroid_mean, centroid_var ,
                            spd_mean, spd_var,
                            rollof_mean, rollof_var,
                            zcr_mean, zcr_var,
                            harm_mean, harm_var,
                            perc_mean, perc_var ,
                            tempo,
                           ],
                 dtype=np.float32)
        features = np.concatenate([features, mfccs])
        return features
    
    def splits_3sec(self):
        """
        Splits an audio signal into 3-second sub-sequences and extracts audio features from each sub-sequence.
        
        Returns:
            ndarray: An array containing the extracted audio features for each 3-second sub-sequence.
        """
        features_split = []
        for sub_sequence in self.y:
            feature = self.features(sub_sequence)
            features_split.append(feature)
        
        features_np = np.array(features_split)
        return features_np


def load_model():
    path =  os.path.dirname(__file__)
    path_model = os.path.join(path, 'models', "model.pkl")
    model = load(path_model)
    return model

def predict(features):
    model = load_model()
    prediction = model.predict(features)
    mode = st.mode(prediction)
    return CLASSES[mode], prediction

def cuts_silence(audio):
    audio_file, _ = librosa.effects.trim(audio)
    return audio_file

def convert_mp3_to_wav(music_file):  
    name_file = "music_file.wav"
    sound = AudioSegment.from_mp3(music_file)
    sound.export(name_file,format="wav")
    return name_file


def preprosecing(uploaded_file):
    name_file = convert_mp3_to_wav(uploaded_file)
    y, sr = librosa.load(name_file)
    audio_file = cuts_silence(y)
    audio_file = audio_file[:sr*30]
    sf.write(file=name_file, data=audio_file, samplerate=sr)
    file = open(name_file, 'rb')
    features = Features(audio_file, sr).splits_3sec()
    prediction = predict(features)
    return file, prediction