import os
os.environ["TF_USE_LEGACY_KERAS"] = "1" # needed for tfgnn
from utils import utils_data, utils_graph
from models import base_gnn

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

    for mfcc, label in train_ds.take(1):
        example_mfcc = mfcc
        example_label = label
        N_FRAMES = tf.shape(mfcc)[0]
        N_MFCCS = tf.shape(mfcc)[1]


    print(f"Number of frames: {N_FRAMES}")
    print(f"Number of MFCCs: {N_MFCCS}")




    _, adjacency_matrix, _ = utils_graph.create_adjacency_matrix(mfcc = example_mfcc, num_frames=N_FRAMES, label = example_label, mode='window', window_size=5)


    
    # Visualize the adjacency matrix
   # utils_graph.visualize_adjacency_matrix(adjacency_matrix, title="Adjacency Matrix")


    # Since some of our adjacency matrix modes will have a different adjacency matrix for each MFCC,
    # we want to efficiently load them (i.e. not create them all at once). Therefore, we add them
    # to our dataset aswell. 

    # Create a new dataset with adjacency matrices

    # use lambda because we want to apply the function to each element of the dataset, which is a tuple (mfcc, label) and we have additional 
    # parameters in our create_adjacency_matrix function
    train_ds = train_ds.map(lambda mfcc, label: utils_graph.create_adjacency_matrix(mfcc, N_FRAMES, label, mode='window', window_size=5))
    val_ds = val_ds.map(lambda mfcc, label: utils_graph.create_adjacency_matrix(mfcc, N_FRAMES, label, mode='window', window_size=5))
    test_ds = test_ds.map(lambda mfcc, label: utils_graph.create_adjacency_matrix(mfcc, N_FRAMES, label, mode='window', window_size=5))

    # Check the shape of the dataset
    for mfcc, adjacency_matrix, label in train_ds.take(1):
        print(f"MFCC shape: {mfcc.shape}")
        print(f"Adjacency matrix shape: {adjacency_matrix.shape}")
        print(f"Label shape: {label.shape}")
        example_mfcc = mfcc
        example_adjacency_matrix = adjacency_matrix



    graph_example = base_gnn.mfccs_to_graph_tensors(example_mfcc, example_adjacency_matrix)
    #print(f"Graph example shape: {graph_example.shape}")
      
    print(f"Graph example: {graph_example}")
    print("Edges:", graph_example.edge_sets["connections"].adjacency)

    #utils_graph.visualize_graph(graph_example, title="Graph Example")




    
    # Finally, we create our final dataset, which puts mfcc's & adjacney matrices together into a graph
    train_ds = train_ds.map(lambda mfcc, adjacency_matrix, label: base_gnn.mfccs_to_graph_tensors_for_dataset(mfcc, adjacency_matrix, label))
    val_ds = val_ds.map(lambda mfcc, adjacency_matrix, label:  base_gnn.mfccs_to_graph_tensors_for_dataset(mfcc, adjacency_matrix, label))
    test_ds = test_ds.map(lambda mfcc, adjacency_matrix, label:  base_gnn.mfccs_to_graph_tensors_for_dataset(mfcc, adjacency_matrix, label))

    # Now batch

    BATCH_SIZE = 32
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # TODO : look for file input & parsing https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/input_pipeline.md
    # TODO : possibly check graph schema tutorial : https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/data_prep.md
    # TODO : scale MFCCs  !!!! for NNs to work correctly 
    # TODO : look at base_gnn code and understand some steps, implement GCN, GAT etc.
    # Check the shape of the dataset
    for graph, label in train_ds.take(1):
        print(f"Graph shape: {graph.shape}")
        print(f"Label shape: {label.shape}")
        print(label)
        
        # We need to get the graphs_spec for our model input
        graphs_spec = graph.spec 

    # Note that we actually have 35 classes !!! not like written in project B1
    base_model = base_gnn.base_gnn_model(graph_tensor_specification = graphs_spec)

    print(base_model.summary())

    history = base_gnn.train(model = base_model,
                             train_ds = train_ds,
                             val_ds = val_ds,
                             epochs = 5,
                             batch_size = BATCH_SIZE)


    



if __name__ == '__main__':
    main()