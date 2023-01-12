import librosa
import numpy as np
from pydub import AudioSegment
import soundfile as sf
import sklearn

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
        tempo = librosa.beat.tempo(y, sr=self.sr)
        return tempo
    
    def centroid(self, y):
        centroid = librosa.feature.spectral_centroid(y, sr=self.sr)
        return self.get_mean_var(centroid)
    
    
    def spectral_contrast(self,y):
        contrast = librosa.feature.spectral_contrast(y, sr=self.sr)
        return self.get_mean_var(contrast)
    
    def mfccs(self, y):
        mfccs = librosa.feature.mfcc(y, sr=self.sr)
        #mfccs = sklearn.preprocessing.scale(mfccs, axis = 1)
        # Get mean and variance for 2oth values
        mean = mfccs.mean(axis=1)
        var = mfccs.var(axis=1)
        return np.array([mean, var])
    
    def chroma_stft(self, y):
        chroma = librosa.feature.chroma_stft(y, sr=self.sr, hop_length=self.hop_length)
        return self.get_mean_var(chroma)
    
    def spectral_bandwidth(self, y):
        spd = librosa.feature.spectral_bandwidth(y,sr=self.sr )
        return self.get_mean_var(spd)
    
    def rollof(self, y):
        rollof = librosa.feature.spectral_rolloff(y=y, sr=self.sr)[0]
        return self.get_mean_var(rollof)
    
    def features(self,y):
        tempo = self.tempo(y)
        centroid_mean, centroid_var = self.centroid(y)
        contrast_mean, contrast_var = self.spectral_contrast(y)
        mfccs_mean, mfccs_var = self.mfccs(y)
        chroma_mean, chroma_var = self.chroma_stft(y)
        zcr_mean, zcr_var = self.zero_crossing_rate(y)
        spd_mean, spd_var = self.spectral_bandwidth(y)
        rollof_mean, rollof_var = self.rollof(y)
        harm, perc = self.harmonic_and_per(y)
        harm_mean, harm_var = harm
        perc_mean, perc_var = perc
        features = np.array([y.shape[0],rollof_mean, rollof_var, spd_mean, spd_var, tempo[0], centroid_mean, centroid_var, contrast_mean, contrast_var, chroma_mean, chroma_var, zcr_mean, zcr_var, harm_mean, harm_var, perc_mean, perc_var],
                 dtype=np.float32)
        features = np.concatenate([features, mfccs_mean, mfccs_var])
        return features
    
    def splits_3sec(self):
        features_split = []
        for sub_sequence in self.y:
            feature = self.features(sub_sequence)
            features_split.append(feature)
        
        features_np = np.array(features_split)
        return features_np
    
    


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
    return file, features