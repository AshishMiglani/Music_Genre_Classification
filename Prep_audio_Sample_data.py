import librosa,librosa.display
import matplotlib.pyplot as plt
import numpy as np


file = "blues.00000.wav"



#waveform
signal,sr= librosa.load(file , sr =22050) #sr*T -> 22050 * 30
# librosa.display.waveshow(signal,sr = sr)
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.show()

# fast fourier transform ->spectrum
fft = np.fft.fft(signal)
magnitude = np.abs(fft)
frequency = np.linspace(0,sr,len(magnitude))

left_frequency=frequency[:int(len(frequency)/2)]
left_magnitude=magnitude[:int(len(magnitude)/2)]
# plt.plot(left_frequency, left_magnitude)
# plt.xlabel("Frequency")
# plt.ylabel("Magnitude")
# plt.show()


#stft -> spectrogram

#number of samples for stft
n_fft = 2048


#the length we are shifting to the right in the interval
hop_length = 512

stft = librosa.core.stft(signal,hop_length = hop_length,n_fft = n_fft)
spectrogram = np.abs(stft)

log_spectrogram = librosa.amplitude_to_db(spectrogram)


# librosa.display.specshow(log_spectrogram,sr = sr,hop_length=hop_length)
# plt.xlabel("Time")
# plt.ylabel("Frequency")
# plt.colorbar()
# plt.show()


MFCCs=librosa.feature.mfcc(y = signal,n_mfcc= 13,hop_length = hop_length,n_fft = n_fft)
librosa.display.specshow(MFCCs,sr = sr,hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC")
plt.colorbar()
plt.show()

