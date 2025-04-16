

import tensorflow as tf 
import os 
import pathlib 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import random 
import platform
from pathlib import Path
import sounddevice as sd



system = platform.system()



### HELPER FUNCTIONS

def idx_to_label_conversion(idx, class_to_index):
    """
    Convert index to label using the class_to_index mapping.
    
    Args:
        idx: Index of the class
        class_to_index: Mapping of class names to indices
    Returns:
        label: Corresponding class name
    """ 
    # Create mapping from index to class name for lookup of class name
    index_to_class = {v: k for k, v in class_to_index.items()}
    
    # Get the label using the index
    label = index_to_class[idx]
    
    return label


def label_to_idx_conversion(label, class_to_index):
    """
    Convert label to index using the class_to_index mapping.
    
    Args:
        label: Class name
        class_to_index: Mapping of class names to indices
    Returns:
        idx: Corresponding index of the class
    """ 
    # Get the index using the label
    idx = class_to_index[label]
    
    return idx


def read_path_to_wav(file_path):
    """
    Read the audio file and return its contents.
    
    Args:
        file_path: Path to the audio file
    Returns:
        wav: Waveform of the audio file
        sample_rate: Sample rate of the audio file
    """

    # Load audio file
    file_contents = tf.io.read_file(file_path)

    wav, sample_rate = tf.audio.decode_wav(file_contents)
    
    return wav, sample_rate

def listen_audio(wav, sample_rate=16000):
    """
    Listen to the audio waveform in a Python script.
    
    Args:
        wav: Audio waveform tensor or numpy array
        sample_rate: Sample rate of the audio file
    """
    # Convert to numpy if it's a tensor
    if hasattr(wav, 'numpy'):
        wav = wav.numpy()
    
    # Ensure the audio is normalized
    if np.abs(wav).max() > 1.0:
        wav = wav / np.abs(wav).max()
    
    # Play the audio
    sd.play(wav, sample_rate)
    sd.wait()  # Wait until audio finishes playing


### MANIPULATION FUNCTIONS 

# THESE TWO FOR SINGLE WAV FILES

def add_noise(wav, noise_dir='speech_commands_v0.02/_background_noise_', 
              noise_type='random', min_snr_db=-5, max_snr_db=10):
    """
    Add noise to an audio waveform at a random SNR between min_snr_db and max_snr_db.
    Using : https://github.com/hrtlacek/SNR/blob/main/SNR.ipynb

    Approach for random sampling is inspired by Andrade2018 paper.
    
    Args:
        wav: Audio waveform
        noise_dir: Directory containing noise files
        noise_type: Type of noise ('random', 'white', etc.)
        min_snr_db: Minimum SNR in dB
        max_snr_db: Maximum SNR in dB
    Returns:
        wavs_noisy: waveform with added noise at specified SNR
    """
    # Load the noise files
    noise_files = list(pathlib.Path(noise_dir).glob('*.wav'))
    noise_files = [str(noise_file) for noise_file in noise_files]
    noise_dir = Path(noise_dir + '/')

    if noise_type == 'random':
        # Randomly select a noise file
        noise_file = random.choice(noise_files)
    else:
        noise_file = noise_dir / (noise_type + '.wav')
        noise_file = str(noise_file)
        
    # Read the noise file
    noise_wav, _ = read_path_to_wav(noise_file)

    # Get a random segment of noise
    noise_length = tf.shape(noise_wav)[0]
    start_index = random.randint(0, noise_length - 16000)
    noise_segment = noise_wav[start_index:start_index + 16000]
    
    # Ensure the noise segment has the same shape as the wav
    if len(noise_segment.shape) > 1:
        noise_segment = tf.squeeze(noise_segment, axis=-1)
    
    # Calculate signal power
    signal_power = tf.reduce_mean(tf.square(wav))
    noise_power = tf.reduce_mean(tf.square(noise_segment))
    
    # Generate random SNR in the specified range
    target_snr_db = random.uniform(min_snr_db, max_snr_db)
    
    # Calculate the scaling factor for the noise
    target_snr_linear = 10 ** (target_snr_db / 10)
    scaling_factor = tf.sqrt(signal_power / (noise_power * target_snr_linear))
    
    # Scale the noise to achieve the target SNR
    scaled_noise = noise_segment * scaling_factor
    
    # Add the scaled noise to the signal
    wav_noisy = wav + scaled_noise
    
    return wav_noisy


def add_padding_or_trimming(wav, target_length = 16000, padding_mode = 'realistic'):
  
    """
    Add padding or trimming to the audio waveform to make it a fixed length.


    
    Args:
        wav: Audio waveform
        target_length: Target length in samples
        padding_mode: Padding mode ('zeros' or 'realistic')
    Returns:
        wav: Padded or trimmed waveform
    """
    # Get the current length of the waveform
    current_length = tf.shape(wav)[0]

    # Check if we need to pad or trim
    if current_length == target_length:
        print("Already at target length; no padding or trimming needed.")
        return wav
    
    # Handle different lengths
   
    else:
        # First, unsqueeze in the last axis (needed for padding)
        wav = tf.expand_dims(wav, axis=-1)
        if current_length > target_length:
            # Trim to target length
            wav = wav[:target_length]
        else:
            # TODO : unrealiable, maybe try something better here in the end (instead of zero padding)
            if padding_mode == 'realistic':
                # Take the first n_pad samples of the signal and repeat them as padding
                n_pad = 100
                wav_first_100 = wav[:n_pad]
                # Repeat the first 100 samples to fill the gap (along the time axis)

                # Calculate the number of repetitions needed
                n_repititions = (target_length - current_length) // n_pad
                wav_padding = tf.tile(wav_first_100, [n_repititions + 1, 1])
                # Concatenate the original wav with the padding
                wav = tf.concat([wav, wav_padding[:target_length - current_length]], axis=0)

                
            else:
                # Pad with zeros to reach target length
                paddings = [[0, target_length - current_length], [0, 0]]
                wav = tf.pad(wav, paddings)
        
    # Finally, squeeze the wav (i.e. remove the channel dimension (we have one channel))
    wav = tf.squeeze(wav, axis=-1)
    
    return wav

# THIS ONE FOR DATASET

def preprocess_audio(file_path, label, sample_rate, frame_length, frame_step, noise = False, noise_type = 'random', min_snr_db = -5, max_snr_db = 10):
    """
    Preprocess the audio file by loading, trimming/padding, and normalizing.
    
    Args:
        
        file_path: Path to the audio file
        label: Label of the audio file
        noise: Boolean indicating whether to add noise or not

    Returns:
        wav: Preprocessed waveform
        label: Label of the audio file"""
    # Load audio file
    file_contents = tf.io.read_file(file_path)
    # Decode wav (returns waveform and sample rate)
    wav, _ = tf.audio.decode_wav(file_contents)
    

    # Since not all audio samples have the same length, we need to
    # ensure they have the same length (for batching) by padding shorter/
    # trimming longer audio files
    # Standardize length to 16000 samples (1 second at 16kHz)


    target_length = 16000 # think can be changed to sample_rate
    
    # Get current length
    current_length = tf.shape(wav)[0]
    
    # Handle different lengths
    if current_length > target_length:
        # Trim to target length
        wav = wav[:target_length]
    else:
        # Pad with zeros to reach target length
        paddings = [[0, target_length - current_length], [0, 0]]
        wav = tf.pad(wav, paddings)


    # Squeeze the wav (i.e. remove the channel dimension (we have one channel))
    wav = tf.squeeze(wav, axis = -1)

    if noise == True:
        # Load the noise files
        noise_files = list(pathlib.Path('speech_commands_v0.02/_background_noise_').glob('*.wav'))
        noise_files = [str(noise_file) for noise_file in noise_files]
        noise_dir = Path('speech_commands_v0.02/_background_noise_' + '/')

        if noise_type == 'random':
            # Randomly select a noise file
            noise_file = random.choice(noise_files)
        else:
            noise_file = noise_dir / (noise_type + '.wav')
            noise_file = str(noise_file)
            
       
        noise_wav, _ = read_path_to_wav(noise_file)

        # Get a random segment of noise
        noise_length = tf.shape(noise_wav)[0]

        start_index = tf.random.uniform(
            shape=[], 
            minval=0, 
            maxval=noise_length - 16000,
            dtype=tf.int32
                                        )
      #  start_index = random.randint(0, int(noise_length) - 16000)


        noise_segment = noise_wav[start_index:start_index + 16000]
        
        # Ensure the noise segment has the same shape as the wav
        if len(noise_segment.shape) > 1:
            noise_segment = tf.squeeze(noise_segment, axis=-1)
        
        # Calculate signal power
        signal_power = tf.reduce_mean(tf.square(wav))
        noise_power = tf.reduce_mean(tf.square(noise_segment))
        
        # Generate random SNR in the specified range
        target_snr_db = 5 #random.uniform(min_snr_db, max_snr_db)
        
        # Calculate the scaling factor for the noise
        target_snr_linear = 10 ** (target_snr_db / 10)
        scaling_factor = tf.sqrt(signal_power / (noise_power * target_snr_linear))
        
        # Scale the noise to achieve the target SNR
        scaled_noise = noise_segment * scaling_factor
        
        # Add the scaled noise to the signal
        wav = wav + scaled_noise

    # Remove noise in the frequency domain
    wav = noise_reduction(wav, noise_threshold=0.1, frame_length=frame_length, frame_step=frame_step)

    # Next, get the spectrogram of the audio file
    spectrogram, frame_step = get_spectrogram(wav)
    # Apply mel filterbanks

    log_mel_spectrogram = apply_mel_filterbanks(spectrogram, sample_rate)
    # Get the MFCC
    mfcc = get_mfccs(log_mel_spectrogram, wav, frame_length=frame_length, frame_step=frame_step, M=2)
    
    return mfcc,wav, label


### MAIN FUNCTIONS 

def load_audio_dataset(data_dir, validation_file, test_file, batch_size=32):
    """
    Load audio datasets with predefined splits from text files.
    
    Args:
        data_dir: Directory containing audio files organized in class subfolders
        validation_file: Path to validation_list.txt
        test_file: Path to testing_list.txt
        batch_size: Batch size for the dataset
        sample_rate: Sample rate for the audio files
        duration: Duration in seconds to crop/pad audio files
    
    Returns:
        train_ds, val_ds, test_ds: TensorFlow datasets for each split
    """
    data_dir = pathlib.Path(data_dir)
    
    # Read validation and test file lists
    with open(validation_file, 'r') as f:
        val_files = set([line.strip() for line in f.readlines()])
    
    with open(test_file, 'r') as f:
        test_files = set([line.strip() for line in f.readlines()])
    
    # Get all class folders (excluding _background_noise_, since not part of train, val or test)
    class_names = sorted([item.name for item in data_dir.glob('*/') 
                          if item.name != '_background_noise_'])
    
    # Create class to index mapping
    class_to_index = {cls: i for i, cls in enumerate(class_names)}
    
    # Create file lists for each split
    train_files, train_labels = [], []
    val_files_list, val_labels = [], []
    test_files_list, test_labels = [], []
    
    # Iterate through all audio files
    for class_dir in data_dir.glob('*/'):
        if class_dir.name == '_background_noise_':
            continue
        
        class_idx = class_to_index[class_dir.name]
        
        # Go through all the audio files of the current folder
        for audio_file in class_dir.glob('*.wav'):
            # Get relative path for matching with validation/test lists (i.e. bed/0a7_nohash_0.wav)
            rel_path = os.path.join(class_dir.name, audio_file.name)
            # Needed for windows compatibility
            if system == "Windows":
                rel_path = rel_path.split("\\")
                rel_path = "/".join(rel_path)
            if rel_path in test_files:
                test_files_list.append(str(audio_file))
                test_labels.append(class_idx)
            
            elif rel_path in val_files:
                val_files_list.append(str(audio_file))
                val_labels.append(class_idx)
    
            else:
                train_files.append(str(audio_file))
                train_labels.append(class_idx)
             



    return train_files, train_labels, val_files_list, val_labels, test_files_list, test_labels, class_to_index

    
def create_tf_dataset(path_files, labels, sample_rate, frame_length, frame_step, mode = 'train', noise = False, noise_type = 'random', min_snr_db = -5, max_snr_db = 10):
    """
    Create a TensorFlow dataset from the audio files and labels.
    Args:
        path_files: List of audio file paths
        labels: List of corresponding labels
        mode: Mode of the dataset ('train', 'val', or 'test')
    Returns:
        dataset: TensorFlow dataset"""

    # Create datasets
    ds = tf.data.Dataset.from_tensor_slices((path_files, labels))
    # Shuffle if train
    if mode == 'train':
        ds = ds.shuffle(buffer_size=len(ds))
 
    ds = ds.map(
    lambda file_path, label: preprocess_audio(
        file_path, 
        label, 
        sample_rate=sample_rate,
        frame_length=frame_length,
        frame_step=frame_step,
        noise=noise, 
        noise_type=noise_type, 
        min_snr_db=min_snr_db, 
        max_snr_db=max_snr_db
    ),
    num_parallel_calls=tf.data.AUTOTUNE
                )  
     
    
    return ds



def noise_reduction(wav, noise_threshold=0.1, frame_length = 400, frame_step = 160):

    """
    Reduce noise in frequency domain before Mel filterbank application.
    
    Parameters:
    -----------
    audio_signal : tf.Tensor
        Input audio signal
    noise_threshold : float, optional
        Threshold for noise reduction (default 0.1)
    frame_length : int, optional
        FFT window size
    frame_step : int, optional
        Step size for FFT
    
    Returns:
    --------
    tf.Tensor
        Noise-reduced audio signal
    """
    # Compute Short-Time Fourier Transform (STFT) (Time -> Frequency domain)
    stft = tf.signal.stft(wav, frame_length=frame_length, frame_step=frame_step)
    
    # Compute magnitude and phase
    magnitude = tf.abs(stft)
    phase = tf.math.angle(stft)
    
    # Compute noise threshold
    noise_floor = tf.reduce_mean(magnitude, axis=0)
    
    # Create a noise reduction mask
    noise_mask = magnitude < (noise_floor * noise_threshold)
    
    # Filter out noise in magnitude spectrum (set it to zero)
    cleaned_magnitude = tf.where(noise_mask, tf.zeros_like(magnitude), magnitude)
    
    # Reconstruct complex spectrum
    cleaned_stft = tf.complex(cleaned_magnitude * tf.cos(phase), 
                           cleaned_magnitude * tf.sin(phase))
    
    # Compute the Inverse STFT (Frequency -> Time domain)
    cleaned_wav = tf.signal.inverse_stft(cleaned_stft, frame_length=frame_length, frame_step=frame_step, window_fn=tf.signal.hamming_window)
    
    return cleaned_wav




def get_spectrogram(wav, sample_rate = 16000):
  # Taken partly from : https://www.tensorflow.org/tutorials/audio/simple_audio


    # Convert the waveform to a spectrogram via a STFT.

    # fft_length : defines that x-point STFT. Higher values give finer frequencies,
    # but more calculations
    # frame_length : define the length of the frame  ; we'll use 25 ms due to it
    # being the standard value , therefore, since the sample rate is 16000kHz, we
    # have 16.000 samples/second * 0.025s = 400 samples.
    # frame_step : defines the overlap of frames that we have. We take the standard
    # value of 10 ms --> 16.000 samples/second * 0.010s = 160 samples.

    frame_length = int(sample_rate * 0.025)  # 25 ms # like in lecture
    frame_step = int(sample_rate * 0.010)  # 10 ms # like in lecture

    spectrogram = tf.signal.stft(wav, frame_length= frame_length, frame_step= frame_step, fft_length= frame_length,
                        window_fn= tf.signal.hamming_window) # using Hamming Window like in Lecture (TODO: Eventually we can try different types of windows (e.g. Gaussian etc))
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)

    # Until now, we computed the power spectrogram
    # We have fft_lengh= 400, so we represent fft_lengh/2= 200 frequency bins


    return spectrogram, frame_step


def apply_mel_filterbanks(spectrogram, sample_rate = 16000):
    # Taken partly from https://www.tensorflow.org/api_docs/python/tf/signal/mfccs_from_log_mel_spectrograms

    # Obtain the number of frequency bins of our spectrogram.
    num_spectrogram_bins = tf.shape(spectrogram)[-1]

    # Define the frequency band we are intereted into:
    min_frequency = 100  # to filter out some background noise, we look at frequencies from 80 ...
    max_frequency = float(sample_rate/2)    # ... up to Nyquist frequency (8000 Hz in our case)

    # And the number of filters
    num_mel_filters = 26

    # Create transformation matrix that maps from linear frequency scale to mel frequency scale
    mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_filters, num_spectrogram_bins, sample_rate, min_frequency, max_frequency)

    # Apply the transformation
    mel_spectrogram = tf.tensordot(spectrogram, mel_weight_matrix, 1)

    # Set output shape 
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrogram
    log_mel_spectrogram = tf.math.log(mel_spectrogram + np.finfo(float).eps)


    return log_mel_spectrogram

# TODO: Try to implement Gammatone filterbanks instead of Mel-Filterbanks, as they work better in handling noise


def get_mfccs(log_mel_spectrogram, wav, frame_length, frame_step, M = 2):
    
    # 1. Compute the DCT and selects the coefficients 2, ... 13 from the log-mel spectrogram
    mfccs_0 = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., 1:13]

    # 2. Define the function to compute the delta coefficients:

    def compute_delta(mfccs, M):
        # Get the number of frames (needed, bc tensorflow has None dynamically for the first dimension and we need it in calculations)
        frame_count = tf.shape(mfccs)[0]
    
        # Pad the mfccs at the beginning and at the end to handle boundary frames
        padded_mfccs = tf.pad(mfccs, [[M, M], [0, 0]], mode='SYMMETRIC')    # This pads [M,M] in time (frames) dimension and pads [0,0] in the frequency dimension
    
        # Prepare the denominator: 2 * sum(m²)
        denominator = 2 * sum([m**2 for m in range(1, M+1)])
    
        # Initialize the deltas
        deltas = tf.zeros_like(mfccs)
    
        # Iterate through each m value
        for m in range(1, M+1):
        
            # Get frames at n+m
            next_frames = padded_mfccs[M+m:M+m+frame_count]
            # The indexes are shifted by M with respect to the original mfccs because of the padding
            
            # Get frames at n-m
            prev_frames = padded_mfccs[M-m:M-m+frame_count]
        
            # Add weighted difference to the delta coefficients
            deltas += m * (next_frames - prev_frames) / denominator
    
        return deltas
    
    #TODO: check this website https://desh2608.github.io/2019-07-26-delta-feats/
    

    # 3. Compute the first derivative of the MFCCs
    mfccs_delta_1 = compute_delta(mfccs_0, M)


    # 4. Compute the second derivative of the MFCCs
    mfccs_delta_2 = compute_delta(mfccs_delta_1, M)


    # 5. Compute the energies of the signal:

    # First energy
    # Divide the raw audio into frames
    framed_wav = tf.signal.frame(wav, frame_length= frame_length, frame_step= frame_step)
    # Compute the energy of each frame
    frame_energy = tf.reduce_sum(framed_wav**2, axis=-1)
    # Take the logarithm
    log_frame_energy = tf.math.log(frame_energy + np.finfo(float).eps)/tf.math.log(10.0)
    # Add a dimension to match the shape of mfccs_0
    log_frame_energy = tf.expand_dims(log_frame_energy, axis=-1)

    # Second energy
    energy_delta_1 = compute_delta(log_frame_energy, M)

    # Third energy
    energy_delta_2 = compute_delta(energy_delta_1, M)

    # 6. Finally, concatenate the MFCCs, delta coefficients and energy coefficients:
    mfccs = tf.concat([mfccs_0, mfccs_delta_1, mfccs_delta_2, log_frame_energy, energy_delta_1, energy_delta_2], axis=-1)

    return mfccs








### VISUALIZATION FUNCTIONS



def visualize_single_waveform(wav, label):
    """
    Visualize a single waveform.

    Args:
        wav : waveform file
        label : waveform label
    
    """
    plt.figure(figsize=(20, 4))
    
    # Extract and plot audio data
    audio_signal = wav.numpy().flatten()  # Convert to numpy and flatten if needed
    time_axis = np.arange(len(audio_signal))
    plt.plot(time_axis, audio_signal, linewidth=1)
    
    # Set class name as title
    class_name = label
    plt.title(class_name, fontsize=10)
    
    # Improve y-axis
    plt.yticks(np.arange(-1.0, 1.1, 0.5), fontsize=8)
    plt.ylim([-1.1, 1.1])
    
    # Improve x-axis
    plt.xticks(fontsize=8)
    
    # Remove top and right spines for cleaner look
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Add very light grid for better readability
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.show()


def visualize_waveforms(wavs, labels):
    """
    Visualize some waveforms.

    Args:
        wavs : waveform files
        labels : waveform labels
    
    """
    plt.figure(figsize=(20, 16)) 
    rows = 4
    cols = 4
    n = rows * cols
    
    
    for i in range(n):

        ax = plt.subplot(rows, cols, i+1)
        
        # Extract and plot audio data
        audio_signal = wavs[i].numpy().flatten()  # Convert to numpy and flatten if needed
        time_axis = np.arange(len(audio_signal))
        plt.plot(time_axis, audio_signal, linewidth=1)
        
        # Set class name as title
        class_name = labels[i]
        plt.title(class_name, fontsize=10)
        
        # Improve y-axis
        plt.yticks(np.arange(-1.0, 1.1, 0.5), fontsize=8)
        plt.ylim([-1.1, 1.1])
        
        # Improve x-axis
        plt.xticks(fontsize=8)
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add very light grid for better readability
        plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout(pad=2.0)  # Add padding between subplots
    plt.subplots_adjust(hspace=0.4)  # Add more height space between rows
    plt.show()


def visualize_data_distribution(dataframe):
    """
    Visualize the distribution of classes in the dataset.
    Important for understanding if the model might perform
    less well on some specific words.
    
    Args:
        dataframe: Pandas DataFrame containing the dataset
    """
    plt.figure(figsize=(12, 6))
    class_counts = dataframe['label'].value_counts()
    class_counts.plot(kind='bar')
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()


def visualize_wavs_by_class(wavs, labels, class_name):

    
    """
    Show the amplitude of the waveforms by class.
    
    Args:
        wavs: Waveform files
        labels: Waveform labels
    """


    
    curr_class_wavs = []
    # Iterate through the waveforms and labels
    for wav, label in zip(wavs, labels):
        if label == class_name:
            curr_class_wavs.append(wav.numpy().flatten())
    
    # Plot the waveforms of the class 
    plt.figure(figsize=(20, 16))
    # Define the number of rows and columns for subplots based on the number of waveforms
    n = len(curr_class_wavs)
    # If there are more than 16 waveforms, limit to 16
    n = min(n, 16)
    rows = 4
    cols = 4
    for i in range(n):
        ax = plt.subplot(rows, cols, i+1)
        
        # Extract and plot audio data
        audio_signal = curr_class_wavs[i]
        time_axis = np.arange(len(audio_signal))
        plt.plot(time_axis, audio_signal, linewidth=1)
        plt.title(class_name, fontsize=10)
        plt.yticks(np.arange(-1.0, 1.1, 0.5), fontsize=8)
        plt.ylim([-1.1, 1.1])
        plt.xticks(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(hspace=0.4)
    plt.show()


def visualize_single_spectrogram(spectrogram, frame_step, sample_rate=16000):
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)
    
    # Convert the frequencies to log scale and transpose, so that the time is
    # represented on the x-axis (columns) (thats what the transposing step does).
    # Add an epsilon to avoid taking a log of zero.
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    
    # Calculate the x-axis values to match the waveform samples (up to 16000)
    # Each spectrogram frame corresponds to frame_step samples in the original waveform
    x_values = np.linspace(0, frame_step * width, width)
    
    plt.figure(figsize=(12, 4))
    plt.pcolormesh(x_values, range(height), log_spec)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Frequency bin')
    
    # Set x-axis limit to match the waveform's sample count (16000)
    if frame_step * width > sample_rate:
        plt.xlim([0, sample_rate])
    
    plt.show()


def visualize_waveform_and_spectrogram(waveform, spectrogram, frame_step, label=None, sample_rate=16000):
    # Create a figure with two subplots, stacked vertically
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot waveform on top subplot
    axes[0].plot(waveform)
    axes[0].set_title('Waveform')
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_xlim([0, len(waveform)])
    
    # Prepare spectrogram for bottom subplot
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)
    
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    
    # Calculate the x-axis values to match the waveform samples
    x_values = np.linspace(0, frame_step * width, width)
    
    # Plot spectrogram on bottom subplot
    im = axes[1].pcolormesh(x_values, range(height), log_spec)
  #  fig.colorbar(im, ax=axes[1], format='%+2.0f dB')
    axes[1].set_title('Spectrogram')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Frequency bin')
    
    # Add a main title if label is provided
    if label:
        plt.suptitle(label)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()


def visualize_mfccs(mfccs, label):
    # If there's a batch dimension and batch_size=1, remove it
    if len(mfccs.shape) == 3 and mfccs.shape[0] == 1:
        mfccs = mfccs[0]
    
    # Transpose to get features in rows, frames in columns
    mfccs = tf.transpose(mfccs)
    
    plt.figure(figsize=(12, 6))
    
    # Display the MFCCs
    img = plt.imshow(mfccs.numpy(), aspect='auto', origin='lower', 
               interpolation='none', cmap='jet')
    
    # Add x and y labels
    plt.title(label)
    plt.xlabel('Time (frames)')
    plt.ylabel('MFCC Coefficients')
    
    # Add y-ticks to show the different feature groups
    feature_groups = [
        'MFCC', 'Delta', 'Delta-Delta', 
        'Energy', 'Delta Energy', 'Delta-Delta Energy'
    ]
    # Position the labels at the midpoint of each feature group
    y_positions = [6, 18, 30, 36.5, 37.5, 38.5]
    plt.yticks(y_positions, feature_groups)
    
    # Add gridlines to separate feature groups
    plt.axhline(y=12, color='white', linestyle='-', alpha=0.3)
    plt.axhline(y=24, color='white', linestyle='-', alpha=0.3)
    plt.axhline(y=36, color='white', linestyle='-', alpha=0.3)
    plt.axhline(y=37, color='white', linestyle='-', alpha=0.3)
    plt.axhline(y=38, color='white', linestyle='-', alpha=0.3)
    
    plt.colorbar(img, label='Coefficient Value')

    plt.tight_layout()
    plt.show()


 

# TODO : normalization of audio ?




if __name__ == '__main__':


    """

    SAMPLE_RATE = 16000 # TODO : given ? 
    FRAME_LENGTH = 400
    FRAME_STEP = 160 
    # TODO : calculate these with sample rate

    # Load data
    train_files, train_labels, val_files, val_labels, test_files, test_labels, class_to_index = load_audio_dataset(data_dir = 'speech_commands_v0.02', validation_file = 'speech_commands_v0.02/validation_list.txt', test_file = 'speech_commands_v0.02/testing_list.txt')

    # Check the data split (predefined in the text files)
    total = len(train_files) + len(val_files) + len(test_files)
    print(f"Percentage of training samples: {len(train_files)/total*100:.2f}%")
    print(f"Percentage of validation samples: {len(val_files)/total*100:.2f}%")
    print(f"Percentage of testing samples: {len(test_files)/total*100:.2f}%")

    # Load into pandas dataframes
    train_df = pd.DataFrame({'file_path': train_files, 'label': list(map(lambda x : idx_to_label_conversion(x,class_to_index), train_labels)), 'split': 'train'})
    val_df = pd.DataFrame({'file_path': val_files, 'label': list(map(lambda x : idx_to_label_conversion(x, class_to_index), val_labels)), 'split': 'val'})
    test_df = pd.DataFrame({'file_path': test_files, 'label': list(map(lambda x : idx_to_label_conversion(x,class_to_index), test_labels)), 'split': 'test'})

    # Concatenate the dataframes
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Save the dataframe to a CSV file
   # all_df.to_csv('speech_commands_dataset.csv', index=False)


   # visualize_data_distribution(all_df)

    # Next, we want to check the actual audio signals. Since loading all the samples into
    # memory is not possible, we will only load a few samples (100) for analysis

    # Set random seed for reproducibility
    random.seed(42)


    wavs = []
    wav_shapes = []
    names_labels = []
    sample_rates = []

    # Randomly sample 100 files from the training set
    train_examples = random.sample(list(zip(train_files, list(map(lambda x : idx_to_label_conversion(x, class_to_index), train_labels)))), 100)

    # Extract some information from the audio files
    for idx,elem in enumerate(train_examples):
        wav, sample_rate = read_path_to_wav(elem[0])
        wavs.append(tf.squeeze(wav, axis = -1))
        names_labels.append(elem[1])
        wav_shapes.append(tf.shape(tf.squeeze(wav, axis = -1)))
        sample_rates.append(sample_rate)

    # Assess whether all audio files have 16000 samples
   # assert all(shape == 16000 for shape in wav_shapes), "Not all audio files have the same shape!"

    # Visualize the waveforms (of the first 16 samples)
   # visualize_waveforms(wavs[:16], names_labels[:16])

    # Visualize waveforms of a specific class
  #  visualize_wavs_by_class(wavs, names_labels, class_name = 'eight')


    EXAMPLE = 4
    wav_padded = add_padding_or_trimming(wavs[EXAMPLE], target_length = 16000, padding_mode = 'zeros')
    wav_noisy = add_noise(wav_padded, noise_dir = 'speech_commands_v0.02/_background_noise_', noise_type = 'exercise_bike')
    

    # Listen to the audio
    listen_audio(wavs[EXAMPLE], sample_rate = sample_rates[EXAMPLE].numpy())
    # Listen to the noisy audio
    listen_audio(wav_noisy, sample_rate = sample_rates[EXAMPLE].numpy())
    # Listen to the padded audio
   # listen_audio(wav_padded, sample_rate = sample_rates[EXAMPLE].numpy())

    
  #  visualize_single_waveform(wavs[EXAMPLE], names_labels[EXAMPLE])
    # Plot noisy waveforms
    # visualize_single_waveform(wav_noisy, names_labels[EXAMPLE] + ' (noisy)')
    # Plot the padded waveform
  #  visualize_single_waveform(wav_padded, names_labels[EXAMPLE] + ' (padded)')

    # Visualize the spectrogram of a specific waveform
  #  spectrogram, frame_step = get_spectrogram(wav_noisy, sample_rates[EXAMPLE].numpy())

   # visualize_single_spectrogram(spectrogram.numpy(), frame_step)
   # visualize_single_waveform(wavs[1], names_labels[1])
  #  visualize_waveform_and_spectrogram(wavs[EXAMPLE], spectrogram.numpy(), frame_step, names_labels[EXAMPLE], sample_rate = sample_rates[EXAMPLE].numpy())



  #  log_mel_spectrogram = apply_mel_filterbanks(spectrogram)

  #  mfccs = get_mfccs(log_mel_spectrogram, wav = wav_noisy, frame_length = 400, frame_step = frame_step, M = 2)

  #  visualize_mfccs(mfccs)


    # We can see that our shapes are not all 16000 samples long ; therefore, for the creation of the dataset, we employ trimming & zero-padding (to 16000 samples)

    # Create datasets (which also processes paths into audio files & does trimming/padding)
    # TODO : look in Andrade2018 paper how they did noisy datasets, I think train has noise and val/test they tested both in noisy and non noisy conditions

    
    train_ds, val_ds, test_ds = create_tf_dataset(train_files, train_labels, batch_size = 32, mode = 'train', noise = True),\
                                create_tf_dataset(val_files, val_labels, batch_size = 32, mode = 'val'),\
                                create_tf_dataset(test_files, test_labels, batch_size = 32, mode = 'test')

    
    
    # Check the shape of the dataset
    for mfcc, label in train_ds.take(1):
        print(f"Shape of the mfcc: {mfcc.shape}")
        print(f"Shape of the label: {label.shape}")

    # Visualize some examples 
    for mfcc, label in train_ds.take(1):
        visualize_mfccs(mfcc[0], idx_to_label_conversion(tf.get_static_value(label[0]), class_to_index))

    """
