import tensorflow as tf 
import os 
import pathlib 
import numpy as np 
import matplotlib.pyplot as plt


def load_audio_dataset(data_dir, validation_file, test_file, batch_size=32, sample_rate=16000, duration=1.0):
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
            
            if rel_path in test_files:
                test_files_list.append(str(audio_file))
                test_labels.append(class_idx)
            elif rel_path in val_files:
                val_files_list.append(str(audio_file))
                val_labels.append(class_idx)
            else:
                train_files.append(str(audio_file))
                train_labels.append(class_idx)
    

    # Function to load and preprocess audio
    def preprocess_audio(file_path, label):
        # Load audio file
        file_contents = tf.io.read_file(file_path)
        # Decode wav (returns waveform and sample rate)
        wav, sample_rate = tf.audio.decode_wav(file_contents)
       
   
        # Since not all audio samples have the same length, we need to
        # ensure they have the same length (for batching) by padding shorter/
        # trimming longer audio files
        # Standardize length to 16000 samples (1 second at 16kHz)
        target_length = 16000
        
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


        # Finally, squeeze the wav (i.e. remove the channel dimension (we have one channel))

        wav = tf.squeeze(wav, axis = -1)
        
        return wav, label

    
    # Create datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    train_ds = train_ds.shuffle(buffer_size = len(train_ds))
    train_ds = train_ds.map(preprocess_audio, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((val_files_list, val_labels))
    val_ds = val_ds.map(preprocess_audio, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    test_ds = tf.data.Dataset.from_tensor_slices((test_files_list, test_labels))
    test_ds = test_ds.map(preprocess_audio, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


    
    return train_ds, val_ds, test_ds, class_to_index


def visualize_waveforms(wavs, labels, class_to_index):
    """
    Visualize some waveforms.

    Args:
        wavs : waveform files
        labels : waveform labels
        class_to_index : mapping of class (label) names to indices
    
    """
    plt.figure(figsize=(20, 16)) 
    rows = 4
    cols = 4
    n = rows * cols
    
    # Create mapping from index to class name for lookup of class name
    index_to_class = {v: k for k, v in class_to_index.items()}
    
    for i in range(n):

        ax = plt.subplot(rows, cols, i+1)
        
        # Extract and plot audio data
        audio_signal = wavs[i].numpy().flatten()  # Convert to numpy and flatten if needed
        time_axis = np.arange(len(audio_signal))
        plt.plot(time_axis, audio_signal, linewidth=1)
        
        # Set class name as title
        class_name = index_to_class[labels[i].numpy()] if hasattr(labels[i], 'numpy') else index_to_class[labels[i]]
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





# TODO : normalization of audio ?
if __name__ == '__main__':
    train_ds, val_ds, test_ds, class_to_index = load_audio_dataset(data_dir = 'speech_commands_v0.02', validation_file = 'speech_commands_v0.02/validation_list.txt', test_file = 'speech_commands_v0.02/testing_list.txt')

    for example_audio, example_labels in train_ds.take(1):

        visualize_waveforms(example_audio[0:16], example_labels[0:16], class_to_index)



