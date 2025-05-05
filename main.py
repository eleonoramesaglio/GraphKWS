import os
os.environ["TF_USE_LEGACY_KERAS"] = "1" # needed for tfgnn
from utils import utils_data, utils_graph, utils_metrics
from models import base_gnn
import matplotlib 

import pandas as pd 
import tensorflow as tf 
from tuning_gnn_models import * 
import random 
import numpy as np



# Get the tensorflow version
print(f"Tensorflow version: {tf.__version__}")






def main():


    # For reproducibility & testing across different models


    for i in range(17,20):
        tf.random.set_seed(32)



        SAMPLE_RATE = 16000 # given in the dataset
        FRAME_LENGTH = int(SAMPLE_RATE * 0.025)  # 25 ms 
        FRAME_STEP = int(SAMPLE_RATE * 0.010)  # 10 ms 

        N_DILATION_LAYERS = 0

        REDUCED_NODE_REP_BOOL = True
        REDUCED_NODE_REP_K = 2


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
                                                                mode = 'train', gammatone = True, noise = True, noise_prob = 0.8, spec_augmentation = False, spec_prob = 0.8),\
                                    utils_data.create_tf_dataset(val_files, val_labels,sample_rate= SAMPLE_RATE, 
                                                                frame_length = FRAME_LENGTH, frame_step= FRAME_STEP,
                                                                mode = 'val', gammatone = True, noise = False, spec_augmentation = False),\
                                    utils_data.create_tf_dataset(test_files, test_labels, sample_rate= SAMPLE_RATE, 
                                                                frame_length = FRAME_LENGTH, frame_step= FRAME_STEP,
                                                                mode = 'test', gammatone = True, noise = False, spec_augmentation = False)
        

        # Reduce the node representation of our graph by pooling over the 98 frames in groups of size k
        if REDUCED_NODE_REP_BOOL:
            train_ds = train_ds.map(lambda mfcc, wav, label : utils_graph.get_reduced_representation(mfcc,wav,label, k = REDUCED_NODE_REP_K, pooling_type = 'max'))
            val_ds = val_ds.map(lambda mfcc, wav, label : utils_graph.get_reduced_representation(mfcc,wav,label, k = REDUCED_NODE_REP_K, pooling_type = 'max'))
            test_ds = test_ds.map(lambda mfcc, wav, label : utils_graph.get_reduced_representation(mfcc,wav,label, k = REDUCED_NODE_REP_K, pooling_type = 'max'))





        # Since all audios are padded to the same length, we can extract static sizes for the MFCCs

        for mfcc, wav, label in train_ds.take(1):
            example_mfcc = mfcc
            example_wav = wav
            example_label = label
            N_FRAMES = tf.shape(mfcc)[0]
            N_MFCCS = tf.shape(mfcc)[1]


        print(f"Number of frames: {N_FRAMES}")
        print(f"Number of MFCCs: {N_MFCCS}")




        _, adjacency_matrix, _ = utils_graph.create_adjacency_matrix(mfcc = example_mfcc, num_frames=N_FRAMES, label = example_label, mode='similarity', threshold= 0, cosine_window_thresh = 0.3,window_size_cosine= 3, window_size=5)


    # utils_data.listen_audio(example_wav, sample_rate=SAMPLE_RATE)

    # utils_data.visualize_single_waveform(example_wav, label = 1)
        



        # Since some of our adjacency matrix modes will have a different adjacency matrix for each MFCC,
        # we want to efficiently load them (i.e. not create them all at once). Therefore, we add them
        # to our dataset aswell. 

        # Create a new dataset with adjacency matrices

        # use lambda because we want to apply the function to each element of the dataset, which is a tuple (mfcc, label) and we have additional 
        # parameters in our create_adjacency_matrix function


        

        train_ds = train_ds.map(lambda mfcc, wav, label: utils_graph.create_adjacency_matrix(mfcc, N_FRAMES, label, mode='cosine window',window_size_cosine = 25, n_dilation_layers= N_DILATION_LAYERS, window_size=5, threshold = 0.3))
        val_ds = val_ds.map(lambda mfcc, wav, label: utils_graph.create_adjacency_matrix(mfcc, N_FRAMES, label, mode='cosine window',window_size_cosine = 25, n_dilation_layers= N_DILATION_LAYERS, window_size=5, threshold = 0.3))
        test_ds = test_ds.map(lambda mfcc, wav, label: utils_graph.create_adjacency_matrix(mfcc, N_FRAMES, label,mode='cosine window',window_size_cosine = 25, n_dilation_layers= N_DILATION_LAYERS, window_size=5, threshold = 0.3))
        # Check the shape of the dataset
        for mfcc, adjacency_matrices, label in train_ds.take(1):
            print(f"MFCC shape: {mfcc.shape}")


            print(f"Label shape: {label.shape}")
            example_mfcc = mfcc
            example_adjacency_matrix = adjacency_matrices[0]

        
        # Adjacency test
        # utils_graph.visualize_adjacency_matrix(example_adjacency_matrix, title="Adjacency Matrix")

        spectrogram, _ = utils_data.get_spectrogram(wav, sample_rate = 16000) 
        gam_filters = utils_data.create_gammatone_filterbank(fft_size=FRAME_LENGTH)
    #  utils_data.visualize_filterbank(gam_filters, spectrogram = spectrogram, gammatone = True)
        _ , mel_filters = utils_data.apply_mel_filterbanks(spectrogram)
    #  utils_data.visualize_filterbank(mel_filters, spectrogram = spectrogram)

    #  utils_data.visualize_mfccs(example_mfcc, gammatone = True, label = 1)
    #  utils_data.visualize_filterbank(filters, sample_rate = 16000, num_spectrogram_bins = num_spectrogram_bins)

    # utils_data.visualize_mfccs(example_mfcc, label = 1)



        # Graph test
        graph_example = base_gnn.mfccs_to_graph_tensors(example_mfcc, example_adjacency_matrix)
        # print(f"Graph example shape: {graph_example.shape}")
        print(f"Graph example: {graph_example}")
        print("Edges:", graph_example.edge_sets["connections"].adjacency)
        networkx_graph = utils_graph.convert_tensor_to_networkx(graph_example)
        pos = utils_graph.node_layout(networkx_graph)
        # utils_graph.visualize_graph_with_heatmap(networkx_graph, pos = pos, title="Graph Example")



        # Dilated test
        # dilated = utils_graph.create_dilated_adjacency_matrix(adjacency_matrix, dilation_rate = 2)
        # dilated = utils_graph.dilated_adjacency_matrix_with_weights(dilated, example_mfcc, num_frames=N_FRAMES)
        # utils_graph.visualize_adjacency_matrix(dilated, title="Adjacency Matrix")
        



        # Finally, we create our final dataset, which puts mfcc's & adjacney matrices together into a graph

        train_ds = train_ds.map(lambda mfcc, adjacency_matrices, label: base_gnn.mfccs_to_graph_tensors_for_dataset(mfcc, adjacency_matrices, label, reduced_node_bool= REDUCED_NODE_REP_BOOL, reduced_node_k= REDUCED_NODE_REP_K))
        val_ds = val_ds.map(lambda mfcc, adjacency_matrices, label:  base_gnn.mfccs_to_graph_tensors_for_dataset(mfcc, adjacency_matrices, label, reduced_node_bool= REDUCED_NODE_REP_BOOL, reduced_node_k= REDUCED_NODE_REP_K))
        test_ds = test_ds.map(lambda mfcc, adjacency_matrices, label:  base_gnn.mfccs_to_graph_tensors_for_dataset(mfcc, adjacency_matrices, label, reduced_node_bool= REDUCED_NODE_REP_BOOL, reduced_node_k= REDUCED_NODE_REP_K))



        # TESTING MODE
        
    #  train_ds = train_ds.map(lambda mfcc, adjacency_matrices, label: base_gnn.mfccs_to_graph_tensors_multi_nodes_for_dataset(mfcc, adjacency_matrices, label))
    #  val_ds = val_ds.map(lambda mfcc, adjacency_matrices, label:  base_gnn.mfccs_to_graph_tensors_multi_nodes_for_dataset(mfcc, adjacency_matrices, label))
    #  test_ds = test_ds.map(lambda mfcc, adjacency_matrices, label:  base_gnn.mfccs_to_graph_tensors_multi_nodes_for_dataset(mfcc, adjacency_matrices, label))

        # Now batch

        BATCH_SIZE = 64
        train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)



        # Check the shape of the dataset
        for graph, label in train_ds.take(1):
            print(f"Graph shape: {graph.shape}")
            print(f"Label shape: {label.shape}")
            print(label)
        
            # We need to get the graphs_spec for our model input
            graphs_spec = graph.spec 

        # Note : GCN residual block we didn't implement the dilation mode


        # Reduce multiplies by : less number of edges

        # Tryout :

        # GAT GCN model / GCN residual using multiple message passing (like 5-6) with residual next state, but much less parameters (i.e. smaller layers etc.)
            # such that not too much parameters --> in general just more layers to see if that helps
        # Try out more with dilation 
        


        if i == 17:        
            # Note that we actually have 35 classes !!! not like written in project B1
            base_model = base_gnn.base_gnn_model_using_gcn(graph_tensor_specification = graphs_spec,
                                                        n_message_passing_layers = 2,
                                                          use_residual_next_state = False,
                                                        initial_nodes_mfccs_layer_dims= 64,
                                                        dilation = False,
                                                        n_dilation_layers= N_DILATION_LAYERS,
                                                        l2_reg_factor= 1e-4,
                                                        )
                                                        #  skip_connection_type= 'sum')
        
        if i == 18:

            base_model = base_gnn.base_gnn_model_using_gcn(graph_tensor_specification = graphs_spec,
                                                        n_message_passing_layers = 4,
                                                          use_residual_next_state = False,
                                                        initial_nodes_mfccs_layer_dims= 64,
                                                        dilation = False,
                                                        n_dilation_layers= N_DILATION_LAYERS,
                                                        l2_reg_factor= 1e-4,
                                                        )
                                                        #  skip_connection_type= 'sum')

        if i == 19:


            base_model = base_gnn.GAT_GCN_model(graph_tensor_specification = graphs_spec,
                                                        n_message_passing_layers = 2,
                                                        #  use_residual_next_state = False,
                                                        initial_nodes_mfccs_layer_dims= 64,
                                                        dilation = False,
                                                        n_dilation_layers= N_DILATION_LAYERS,
                                                        l2_reg_factor= 1e-4,
                                                        )
                                                        #  skip_connection_type= 'sum')

        
        if i == 11:

            base_model = base_gnn.GAT_GCN_model_v2(graph_tensor_specification = graphs_spec,
                                                        n_message_passing_layers = 4,
                                                        #  use_residual_next_state = False,
                                                        initial_nodes_mfccs_layer_dims= 64,
                                                        dilation = True,
                                                        n_dilation_layers= N_DILATION_LAYERS,
                                                        l2_reg_factor= 1e-4,
                                                        )
                                                        #  skip_connection_type= 'sum')


        if i == 14:

           base_model = base_gnn.base_GATv2_model(graph_tensor_specification = graphs_spec,
                                                        n_message_passing_layers = 4,
                                                        #  use_residual_next_state = False,
                                                        initial_nodes_mfccs_layer_dims= 64,
                                                        dilation = True,
                                                        n_dilation_layers= N_DILATION_LAYERS,
                                                        l2_reg_factor= 1e-4,
                                                        )
                                                        #  skip_connection_type= 'sum')




        if i == 3:

            base_model = base_gnn.GAT_GCN_model_v2(graph_tensor_specification = graphs_spec,
                                                        n_message_passing_layers = 4,
                                                        #  use_residual_next_state = False,
                                                        initial_nodes_mfccs_layer_dims= 64,
                                                        dilation = False,
                                                        n_dilation_layers= N_DILATION_LAYERS,
                                                        l2_reg_factor= 1e-4,
                                                        )
                                                        #  skip_connection_type= 'sum')


        if i == 4:

            base_model = base_gnn.GAT_GCN_model(graph_tensor_specification = graphs_spec,
                                                        n_message_passing_layers = 2,
                                                        #  use_residual_next_state = False,
                                                        initial_nodes_mfccs_layer_dims= 64,
                                                        dilation = False,
                                                        n_dilation_layers= N_DILATION_LAYERS,
                                                        l2_reg_factor= 1e-4,
                                                        )
                                                        #  skip_connection_type= 'sum')




            





        # TODO : test that spec augmentatio nworks correctly
        # TODO : test diff batch sizes
        # TODO : try the other initialization of MFCCS (so with node init multiple dense layers)
        # TODO : check if also to test that noise in val/test ?

        # TODO : play with LR (cosine , etc. )

        print(base_model.summary())


        history = base_gnn.train(model = base_model,
                                train_ds = train_ds,
                                val_ds = val_ds,
                                test_ds = test_ds,
                                epochs = 30,
                                batch_size = BATCH_SIZE,
                                learning_rate = 0.001)
        



        

        utils_metrics.plot_history(history, columns=['loss', 'sparse_categorical_accuracy'], idx = i)



        # Confusion matrix visualization

        y_pred, y_true = utils_metrics.get_ys(test_ds, base_model)
        utils_metrics.visualize_confusion_matrix(y_pred, y_true, idx = i)

        # Precision, Recall, F1-score
        utils_metrics.metrics_evaluation(y_pred, y_true, model_name = f"Model {i}")
  
    
    
    
    # Run the Optuna study
  #  best_params, study = create_optuna_study(
  #      gnn_model = base_gnn.base_gnn_model_using_gcn,
  #      graph_tensor_specification= graphs_spec,
  #      train_ds=train_ds,
  #      val_ds=val_ds,
  #      num_classes=35,
  #      n_trials=20  # Start with a small number to verify functionality
  #  )

    # Visualize results
  #  visualize_optuna_results(study)

    # Build the model with the best parameters
  #  best_model = build_model_with_best_params(
  #      best_params=best_params,
  #      gnn_model = base_gnn.base_gnn_weighted_model,
  #      graph_tensor_specification=graphs_spec,
  #      num_classes=35
  #  )

    # Train the best model
  #  history = train_best_model(
  #      model=best_model,
  #      train_ds=train_ds,
  #      val_ds=val_ds,
  #      test_ds=test_ds,
  #      epochs=1
  #  )

    






if __name__ == '__main__':
    main()