import numpy as np
from scipy.stats import norm as ssn
import matplotlib.pyplot as plt
import png

"""
Created on Sat May 27 15:37:50 2017
Python version of:
D. P. W. Ellis (2009). "Gammatone-like spectrograms", web resource. http://www.ee.columbia.edu/~dpwe/resources/matlab/gammatonegram/
On the corresponding webpage, Dan notes that he would be grateful if you cited him if you use his work (as above).
This python code does not contain all features present in MATLAB code.
Sat May 27 15:37:50 2017 Maddie Cusimano, mcusi@mit.edu 27 May 2017
"""

from __future__ import division
import numpy as np
import scipy.signal as sps
import scipy.io.wavfile as wf

def fft2gammatonemx(nfft, sr=20000, nfilts=64, width=1.0, minfreq=100,
                    maxfreq=10000, maxlen=1024):
    """
    # Ellis' description in MATLAB:
    # [wts,cfreqa] = fft2gammatonemx(nfft, sr, nfilts, width, minfreq, maxfreq, maxlen)
    #      Generate a matrix of weights to combine FFT bins into
    #      Gammatone bins.  nfft defines the source FFT size at
    #      sampling rate sr.  Optional nfilts specifies the number of
    #      output bands required (default 64), and width is the
    #      constant width of each band in Bark (default 1).
    #      minfreq, maxfreq specify range covered in Hz (100, sr/2).
    #      While wts has nfft columns, the second half are all zero.
    #      Hence, aud spectrum is
    #      fft2gammatonemx(nfft,sr)*abs(fft(xincols,nfft));
    #      maxlen truncates the rows to this many bins.
    #      cfreqs returns the actual center frequencies of each
    #      gammatone band in Hz.
    #
    # 2009/02/22 02:29:25 Dan Ellis dpwe@ee.columbia.edu  based on rastamat/audspec.m
    # Sat May 27 15:37:50 2017 Maddie Cusimano, mcusi@mit.edu 27 May 2017: convert to python
    """

    wts = np.zeros([nfilts,nfft])

    #after Slaney's MakeERBFilters
    EarQ = 9.26449; minBW = 24.7; order = 1;

    nFr = np.array(range(nfilts)) + 1
    em = EarQ*minBW
    cfreqs = (maxfreq+em)*np.exp(nFr*(-np.log(maxfreq + em)+np.log(minfreq + em))/nfilts)-em
    cfreqs = cfreqs[::-1]

    GTord = 4
    ucircArray = np.array(range(int(nfft/2 + 1)))
    ucirc = np.exp(1j*2*np.pi*ucircArray/nfft);
    #justpoles = 0 :taking out the 'if' corresponding to this.

    ERB = width*np.power(np.power(cfreqs/EarQ,order) + np.power(minBW,order),1/order);
    B = 1.019 * 2 * np.pi * ERB;
    r = np.exp(-B/sr)
    theta = 2*np.pi*cfreqs/sr
    pole = r*np.exp(1j*theta)
    T = 1/sr
    ebt = np.exp(B*T); cpt = 2*cfreqs*np.pi*T;
    ccpt = 2*T*np.cos(cpt); scpt = 2*T*np.sin(cpt);
    A11 = -np.divide(np.divide(ccpt,ebt) + np.divide(np.sqrt(3+2**1.5)*scpt,ebt),2);
    A12 = -np.divide(np.divide(ccpt,ebt) - np.divide(np.sqrt(3+2**1.5)*scpt,ebt),2);
    A13 = -np.divide(np.divide(ccpt,ebt) + np.divide(np.sqrt(3-2**1.5)*scpt,ebt),2);
    A14 = -np.divide(np.divide(ccpt,ebt) - np.divide(np.sqrt(3-2**1.5)*scpt,ebt),2);
    zros = -np.array([A11, A12, A13, A14])/T;
    wIdx = range(int(nfft/2 + 1))
    gain = np.abs((-2*np.exp(4*1j*cfreqs*np.pi*T)*T + 2*np.exp(-(B*T) + 2*1j*cfreqs*np.pi*T)*T* (np.cos(2*cfreqs*np.pi*T) - np.sqrt(3 - 2**(3/2))*  np.sin(2*cfreqs*np.pi*T))) *(-2*np.exp(4*1j*cfreqs*np.pi*T)*T + 2*np.exp(-(B*T) + 2*1j*cfreqs*np.pi*T)*T* (np.cos(2*cfreqs*np.pi*T) + np.sqrt(3 - 2**(3/2)) *  np.sin(2*cfreqs*np.pi*T)))*(-2*np.exp(4*1j*cfreqs*np.pi*T)*T + 2*np.exp(-(B*T) + 2*1j*cfreqs*np.pi*T)*T* (np.cos(2*cfreqs*np.pi*T) -  np.sqrt(3 + 2**(3/2))*np.sin(2*cfreqs*np.pi*T))) *(-2*np.exp(4*1j*cfreqs*np.pi*T)*T + 2*np.exp(-(B*T) + 2*1j*cfreqs*np.pi*T)*T* (np.cos(2*cfreqs*np.pi*T) + np.sqrt(3 + 2**(3/2))*np.sin(2*cfreqs*np.pi*T))) /(-2 / np.exp(2*B*T) - 2*np.exp(4*1j*cfreqs*np.pi*T) +  2*(1 + np.exp(4*1j*cfreqs*np.pi*T))/np.exp(B*T))**4);
    #in MATLAB, there used to be 64 where here it says nfilts:
    wts[:, wIdx] =  ((T**4)/np.reshape(gain,(nfilts,1))) * np.abs(ucirc-np.reshape(zros[0],(nfilts,1)))*np.abs(ucirc-np.reshape(zros[1],(nfilts,1)))*np.abs(ucirc-np.reshape(zros[2],(nfilts,1)))*np.abs(ucirc-np.reshape(zros[3],(nfilts,1)))*(np.abs(np.power(np.multiply(np.reshape(pole,(nfilts,1))-ucirc,np.conj(np.reshape(pole,(nfilts,1)))-ucirc),-GTord)));
    wts = wts[:,range(maxlen)];

    return wts, cfreqs

def gammatonegram(spectrogram,sr=20000,nfft=1024,nhop=160,N=128,
                  fmin=50,fmax=10000,width=1.0):
    """
    # Ellis' description in MATLAB:
    # [Y,F] = gammatonegram(X,SR,N,TWIN,THOP,FMIN,FMAX,USEFFT,WIDTH)
    # Calculate a spectrogram-like time frequency magnitude array
    # based on Gammatone subband filters.  Waveform X (at sample
    # rate SR) is passed through an N (default 64) channel gammatone
    # auditory model filterbank, with lowest frequency FMIN (50)
    # and highest frequency FMAX (SR/2).  The outputs of each band
    # then have their energy integrated over windows of TWIN secs
    # (0.025), advancing by THOP secs (0.010) for successive
    # columns.  These magnitudes are returned as an N-row
    # nonnegative real matrix, Y.
    # WIDTH (default 1.0) is how to scale bandwidth of filters
    # relative to ERB default (for fast method only).
    # F returns the center frequencies in Hz of each row of Y
    # (uniformly spaced on a Bark scale).

    # 2009/02/23 DAn Ellis dpwe@ee.columbia.edu
    # Sat May 27 15:37:50 2017 Maddie Cusimano mcusi@mit.edu, converted to python
    """

    #Entirely skipping Malcolm's function, because would require
    #altering ERBFilterBank code as well.
    #i.e., in Ellis' code: usefft = 1
    # assert(x.dtype == 'int16')

    # How long a window to use relative to the integration window requested

    nwin = nfft
    [gtm,f] = fft2gammatonemx(nfft, sr, N, width, fmin, fmax, int(nfft/2+1))
    # perform FFT and weighting in amplitude domain
    # note: in MATLAB, abs(spectrogram(X, hanning(nwin), nwin-nhop, nfft, SR))
    #                  = abs(specgram(X,nfft,SR,nwin,nwin-nhop))
    # in python approx = sps.spectrogram(x, fs=sr, window='hann', nperseg=nwin,
    #                    noverlap=nwin-nhop, nfft=nfft, detrend=False,
    #                    scaling='density', mode='magnitude')

    Sxx = spectrogram
    print("Spectrogram", np.min(Sxx), np.max(Sxx))
    y = (1/nfft)*np.dot(gtm,Sxx)

    return y, f


"""
Example of using python gammatonegram code
mcusi@mit.edu, 2018 Sept 24
"""

def gtg_in_dB(sound, sampling_rate, log_constant=1e-80, dB_threshold=-50.0, fmin=20):
    """ Convert sound into gammatonegram, with amplitude in decibels"""
    print("Sound", np.min(sound), np.max(sound))
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


def spectrogram_to_img(sxx,filename="res.png"):
    arr = sxx+np.abs(np.min(sxx))
    img = (arr * 255 / np.max(arr)).astype('uint8')
    img = np.flip(img, axis=0) # put low frequencies at the bottom in image
    try:
        png.from_array(img, mode="L").save(filename)
    except Exception as e:
        print("Skipping - {}".format(e))

    print(sxx.shape)


def create_spectrogram(type="mel", audio_path, image_path, check=False, novoice_path=None, hop_length=160, n_fft=1024, fmin=20):
    try:
        # sampling the audio with original sample rate - sr = none, return the audio value in ndarray
        audio_tensor, sr = librosa.load(audio_path, sr=None)
        plotF, plotT, Sxx = sps.spectrogram(audio_tensor, fs=sr, window='hann', nperseg=nfft,
                                    noverlap=nfft-nhop, nfft=nfft, detrend=False,
                                    scaling='spectrum', mode='magnitude')

        if type == "mel":
            spectrogram = librosa.feature.melspectrogram(S=Sxx, sr=sr, n_fft=nfft, hop_length=nhop, center=False, fmin=fmin)
        elif type == "gammatone":
            spectrogram, center_frequencies = gammatonegram(Sxx,sr=sr,nfft=nfft,nhop=nhop,N=128,fmin=fmin,fmax=int(sr/2),width=1.0)

        # convert power spectrogram to dB
        spectrogram_to_img(spectrogram,filename=image_path)

        return spectrogram, image

    except ValueError as e:
        logger.warning(f"Failed to generate spectrogram: {e}")


if __name__ == '__main__':
    import librosa
    import librosa.display

    input = "1605001210.0062447-0.wav"

    # sampling_rate , sound = wf.read(input)

    sound, sampling_rate = librosa.load(input, sr=16000)
    print(np.max(sound))
    # print(sampling_rate)
    # print(len(sound))

    if len(sound.shape)>1:
        sound = sound.mean(1)
    else:
        sound = sound

    print(sound.dtype)

    #gtgplot(sxx, center_frequencies, len(sound), sampling_rate)


    plt.figure(figsize=(10, 5))
    audio_tensor, sr = librosa.load(input, sr=None)

    print(len(audio_tensor))
    # generate power spectrogram

    nfft = 1024
    print(nfft)
    nhop = 160
    fmin=20
    plotF, plotT, Sxx = sps.spectrogram(audio_tensor, fs=sr, window='hann', nperseg=nfft,
                                noverlap=nfft-nhop, nfft=nfft, detrend=False,
                                scaling='spectrum', mode='magnitude')

    sxx, center_frequencies = gammatonegram(Sxx,sr=sr,nfft=nfft,nhop=nhop,N=128,fmin=fmin,fmax=int(sr/2),width=1.0)

    mels = librosa.feature.melspectrogram(S=Sxx, sr=sr, n_fft=nfft, hop_length=nhop, center=False, fmin=fmin)
    # convert power spectrogram to dB
    print("Mels", np.min(mels), np.max(mels))

    # mels = librosa.power_to_db(mels, ref=np.max) #convert to dB
    # sxx = librosa.power_to_db(sxx, ref=np.max)

    # log_constant=1e-80
    sxx = 20.*np.log10(sxx) #convert to dB
    mels = 20.*np.log10(mels) #convert to dB

    print("--> Log_spectrogram", np.min(mels), np.max(mels))
    print(mels.shape)
    plt.subplot(1,2,1)

    t_space=20
    f_space=20
    time_per_pixel = 0.01
    t = time_per_pixel * np.arange(sxx.shape[1])

    vmax = None
    vmin = None

    librosa.display.specshow(mels, cmap='jet',
                         sr = sr, hop_length = nhop, fmin = fmin,
                         y_axis='mel', fmax=sampling_rate/2,
                         x_axis='time', x_coords = np.linspace(0,3690,num=94), vmax = vmax, vmin = vmin)
    plt.xticks([i*39 for i in range(sxx.shape[1])[::t_space]], labels=(t[::t_space]*100.).astype(int)/100., fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlim(0,sxx.shape[1])
    plt.xlabel('Time (s)',fontsize=10)
    plt.ylabel('Frequency (Hz)', fontsize=10)
    plt.rc('xtick',labelsize=8)
    plt.rc('ytick',labelsize=8)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.axis('equal')
    plt.box(on=None)
    plt.subplot(1,2,2)


    print(time_per_pixel)


    spectrogram_to_img(sxx,filename="gamma-res.png")
    spectrogram_to_img(mels,filename="mel-res.png")


    plt.pcolormesh(sxx, cmap='jet',  vmax = vmax, vmin = vmin)

    plt.xlabel('Time (s)',fontsize=10)
    print()
    print(range(sxx.shape[1])[::t_space])
    plt.xticks(range(sxx.shape[1])[::t_space], labels=(t[::t_space]*100.).astype(int)/100., fontsize=8)
    plt.xlim(0,sxx.shape[1])
    plt.yticks(range(len(center_frequencies))[::f_space], labels=center_frequencies.astype(int)[::f_space], fontsize=8)
    plt.ylim(0,len(center_frequencies))
    plt.rc('xtick',labelsize=8)
    plt.rc('ytick',labelsize=8)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Gammatonegram')
    plt.tight_layout()
    plt.axis('equal')
    plt.box(on=None)
    plt.show()
