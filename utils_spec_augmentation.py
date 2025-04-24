import tensorflow as tf 
import tensorflow_io as tfio



# Partly taken from :

#https://github.com/DemisEom/SpecAugment/blob/master/SpecAugment/spec_augment_tensorflow.py

# Since in tf 2.19.0, tf-addons are not supported anymore, we do not use time warping


def frequency_masking(mel_spectrogram, v, frequency_masking_para=27, frequency_mask_num=2):
    """Spec augmentation Calculation Function.

    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.

    # Arguments:
      mel_spectrogram(numpy array): audio file path of you want to warping and masking.
      frequency_masking_para(float): Augmentation parameter, "frequency mask parameter F"
        If none, default = 100 for LibriSpeech.
      frequency_mask_num(float): number of frequency masking lines, "m_F".
        If none, default = 1 for LibriSpeech.

    # Returns
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """
    # Step 2 : Frequency masking
    fbank_size = tf.shape(mel_spectrogram)
    n, v = fbank_size[1], fbank_size[2]

    for i in range(frequency_mask_num):
        f = tf.random.uniform([], minval=0, maxval=frequency_masking_para, dtype=tf.int32)
        v = tf.cast(v, dtype=tf.int32)
        f0 = tf.random.uniform([], minval=0, maxval=v-f, dtype=tf.int32)

        # warped_mel_spectrogram[f0:f0 + f, :] = 0
        mask = tf.concat((tf.ones(shape=(1, n, v - f0 - f, 1)),
                          tf.zeros(shape=(1, n, f, 1)),
                          tf.ones(shape=(1, n, f0, 1)),
                          ), 2)
        mel_spectrogram = mel_spectrogram * mask
    return tf.cast(mel_spectrogram, dtype=tf.float32)



def time_masking(mel_spectrogram, tau, time_masking_para=100, time_mask_num=2):
    """Spec augmentation Calculation Function.

    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.

    # Arguments:
      mel_spectrogram(numpy array): audio file path of you want to warping and masking.
      time_masking_para(float): Augmentation parameter, "time mask parameter T"
        If none, default = 27 for LibriSpeech.
      time_mask_num(float): number of time masking lines, "m_T".
        If none, default = 1 for LibriSpeech.

    # Returns
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """
    fbank_size = tf.shape(mel_spectrogram)
    n, v = fbank_size[1], fbank_size[2]

    # Step 3 : Time masking
    for i in range(time_mask_num):
        t = tf.random.uniform([], minval=0, maxval=time_masking_para, dtype=tf.int32)
        t0 = tf.random.uniform([], minval=0, maxval=tau-t, dtype=tf.int32)

        # mel_spectrogram[:, t0:t0 + t] = 0
        mask = tf.concat((tf.ones(shape=(1, n-t0-t, v, 1)),
                          tf.zeros(shape=(1, t, v, 1)),
                          tf.ones(shape=(1, t0, v, 1)),
                          ), 1)
        mel_spectrogram = mel_spectrogram * mask
    return tf.cast(mel_spectrogram, dtype=tf.float32)


def spec_augment(mel_spectrogram):

    v = mel_spectrogram.shape[0]
    tau = mel_spectrogram.shape[1]


    frequency_spectrogram = frequency_masking(mel_spectrogram, v=v)

    frequency_time_sepctrogram = time_masking(frequency_spectrogram, tau=tau)

    return frequency_time_sepctrogram





## Easy implementation, found directly here : https://www.tensorflow.org/io/tutorials/audio




import numpy as np 
def spec_augment_easy(spectrogram, freq_param = 10, time_param = 10, mode = 'all'):
    
    
    
    # TODO : add random number to be generated for freq and time param (more diversity)

    if mode == 'all':
        
        

        spectrogram = tfio.audio.freq_mask(spectrogram, param = freq_param)
        spectrogram = tfio.audio.time_mask(spectrogram, param = time_param)

    if mode == 'freq':
        spectrogram = tfio.audio.freq_mask(spectrogram, param = freq_param)
    
    if mode == 'time':
        spectrogram = tfio.audio.time_mask(spectrogram, param = freq_param)

    return spectrogram


