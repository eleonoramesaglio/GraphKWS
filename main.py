from utils import utils_data, utils_graph

import pandas as pd 
import tensorflow as tf 




def main():

    SAMPLE_RATE = 16000 # given in the dataset
    FRAME_LENGTH = int(SAMPLE_RATE * 0.025)  # 25 ms 
    FRAME_STEP = int(SAMPLE_RATE * 0.010)  # 10 ms 


    # Load data
    train_files, train_labels, val_files, val_labels, test_files, test_labels, class_to_index = utils_data.load_audio_dataset(data_dir = 'speech_commands_v0.02',
                                                                                                                    validation_file = 'speech_commands_v0.02/validation_list.txt',
                                                                                                                      test_file = 'speech_commands_v0.02/testing_list.txt')
    

    # Check the data split (predefined in the text files)
    total = len(train_files) + len(val_files) + len(test_files)
    print(f"Percentage of training samples: {len(train_files)/total*100:.2f}%")
    print(f"Percentage of validation samples: {len(val_files)/total*100:.2f}%")
    print(f"Percentage of testing samples: {len(test_files)/total*100:.2f}%")

    # Load into pandas dataframes
    train_df = pd.DataFrame({'file_path': train_files, 'label': list(map(lambda x : utils_data.idx_to_label_conversion(x,class_to_index), train_labels)), 'split': 'train'})
    val_df = pd.DataFrame({'file_path': val_files, 'label': list(map(lambda x : utils_data.idx_to_label_conversion(x, class_to_index), val_labels)), 'split': 'val'})
    test_df = pd.DataFrame({'file_path': test_files, 'label': list(map(lambda x : utils_data.idx_to_label_conversion(x,class_to_index), test_labels)), 'split': 'test'})
    # Concatenate the dataframes
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    #visualize_data_distribution(all_df)

    # See in utils data for examples on adding noise, padding etc. (which we can show in the presentation ; do a .ipynb)

    # noise = True to match 2015 google paper ; possibly do a comparison with noise = False
    train_ds, val_ds, test_ds = utils_data.create_tf_dataset(train_files, train_labels, sample_rate= SAMPLE_RATE, 
                                                             frame_length = FRAME_LENGTH, frame_step= FRAME_STEP,
                                                               batch_size = 32, mode = 'train', noise = True),\
                                utils_data.create_tf_dataset(val_files, val_labels,sample_rate= SAMPLE_RATE, 
                                                             frame_length = FRAME_LENGTH, frame_step= FRAME_STEP,
                                                               batch_size = 32, mode = 'val'),\
                                utils_data.create_tf_dataset(test_files, test_labels, sample_rate= SAMPLE_RATE, 
                                                             frame_length = FRAME_LENGTH, frame_step= FRAME_STEP,
                                                               batch_size = 32, mode = 'test')
    

    # Since all audios are padded to the same length, we can extract static sizes for the MFCCs

    for mfcc, _ in train_ds.take(1):
        N_FRAMES = tf.shape(mfcc)[1]
        N_MFCCS = tf.shape(mfcc)[2]


    print(f"Number of frames: {N_FRAMES}")
    print(f"Number of MFCCs: {N_MFCCS}")


    # Create static adjacency matrix

    adjacency_matrix = utils_graph.create_adjacency_matrix(num_frames=N_FRAMES, mode='window', window_size=5)
    
    # Visualize the adjacency matrix
    utils_graph.visualize_adjacency_matrix(adjacency_matrix, title="Adjacency Matrix")

if __name__ == '__main__':
    main()