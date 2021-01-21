from gtg import gammatonegram
import numpy as np
from scipy.stats import norm as ssn
import scipy.io.wavfile as wf
import matplotlib.pyplot as plt

"""
Example of using python gammatonegram code
mcusi@mit.edu, 2018 Sept 24
"""

def gtg_in_dB(sound, sampling_rate, log_constant=1e-80, dB_threshold=-50.0, fmin=20):
    """ Convert sound into gammatonegram, with amplitude in decibels"""
    sxx, center_frequencies = gammatonegram(sound, sr=sampling_rate, fmin=fmin, fmax=int(sampling_rate/2.))
    print("Gammatones", np.min(sxx), np.max(sxx))
    sxx[sxx == 0] = log_constant
    sxx = 20.*np.log10(sxx) #convert to dB
    print("--> Log_spectrogram", np.min(sxx), np.max(sxx))
    return sxx, center_frequencies

"""
def loglikelihood(sxx_observation, sxx_hypothesis):
    likelihood_weighting = 5.0 #free parameter!
    loglikelihood = likelihood_weighting*np.sum(ssn.logpdf(sxx_observation,
                                                           loc=sxx_hypothesis, scale=1))
    return loglikelihood
"""

def gtgplot(sxx, center_frequencies, sample_duration, sampling_rate,
            dB_threshold=-50.0, dB_max=10.0, t_space=50, f_space=10):
    """Plot gammatonegram"""

    fig, ax = plt.subplots(1,1)

    time_per_pixel = sample_duration/(1.*sampling_rate*sxx.shape[1])
    t = time_per_pixel * np.arange(sxx.shape[1])

    print(sxx.shape)


    plt.pcolormesh(sxx,vmin=dB_threshold, vmax=dB_max, cmap='Blues')
    ax.set_ylabel('Frequency (Hz)', fontsize=16)
    ax.set_xlabel('Time (s)',fontsize=16)
    ax.xaxis.set_ticks(range(sxx.shape[1])[::t_space])
    ax.xaxis.set_ticklabels((t[::t_space]*100.).astype(int)/100.,fontsize=16)
    ax.set_xbound(0,sxx.shape[1])
    ax.yaxis.set_ticks(range(len(center_frequencies))[::f_space])
    ax.yaxis.set_ticklabels(center_frequencies.astype(int)[::f_space],fontsize=16)
    ax.set_ybound(0,len(center_frequencies))
    cbar = plt.colorbar(pad=0.01)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.set_ylabel('Amplitude (dB)', rotation=270, labelpad=15,fontsize=16)
    plt.show()

if __name__ == '__main__':
    import librosa
    import librosa.display

    input = "1-0-smartlab_chen-1.wav"

    sampling_rate , sound = wf.read(input)
    print(np.max(sound))
    #sound, sampling_rate = librosa.load('1-0-smartlab_chen-1.wav', sr=None)
    # print(sampling_rate)
    # print(len(sound))

    if len(sound.shape)>1:
        sound = sound.mean(1)
    else:
        sound = sound

    print(sound.dtype)
    sxx, center_frequencies = gtg_in_dB(sound.astype(np.int16), sampling_rate)
    #gtgplot(sxx, center_frequencies, len(sound), sampling_rate)


    plt.figure(figsize=(10, 6))
    audio_tensor, sr = librosa.load(input, sr=None)

    print(len(audio_tensor))
    # generate power spectrogram
    mels = librosa.feature.melspectrogram(audio_tensor, sr=sr, n_fft=1024, hop_length=160, center=False, fmin=20)
    # convert power spectrogram to dB
    print("Mels", np.min(mels), np.max(mels))
    log_spectrogram = librosa.power_to_db(mels) #convert to dB
    print("--> Log_spectrogram", np.min(log_spectrogram), np.max(log_spectrogram))
    print(log_spectrogram.shape)
    plt.subplot(1,2,1)
    librosa.display.specshow(log_spectrogram, cmap='jet',
                         sr = sr, hop_length = 160, fmin = 20,
                         y_axis='mel', fmax=sampling_rate/2,
                         x_axis='time', vmax=0, vmin=-60)
    plt.ylabel('Frequency (Hz)', fontsize=10)
    # plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.subplot(1,2,2)

    dB_threshold=-50.0
    dB_max=0
    t_space=100
    f_space=20

    time_per_pixel = len(sound)/(1.*sampling_rate*sxx.shape[1])
    t = time_per_pixel * np.arange(sxx.shape[1])

    import png
    arr = sxx+np.abs(np.min(sxx))
    img = (arr * 255 / np.max(arr)).astype('uint8')
    img = np.flip(img, axis=0) # put low frequencies at the bottom in image
    try:
        png.from_array(img, mode="L").save("res.png")
    except Exception as e:
        print("Skipping - {}".format(e))

    print(sxx.shape)


    plt.pcolormesh(sxx, vmax=0, vmin=-60, cmap='jet')

    plt.xlabel('Time (s)',fontsize=10)
    plt.xticks(range(sxx.shape[1])[::t_space], labels=(t[::t_space]*100.).astype(int)/100.)
    plt.xlim(0,sxx.shape[1])
    plt.yticks(range(len(center_frequencies))[::f_space], labels=center_frequencies.astype(int)[::f_space])
    plt.ylim(0,len(center_frequencies))
    plt.colorbar(format='%+2.0f dB')
    plt.title('Gammatonegram')
    plt.tight_layout()
    plt.show()
