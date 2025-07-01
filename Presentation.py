import os
os.environ["TF_USE_LEGACY_KERAS"] = "1" # needed for tfgnn
from utils import utils_data, utils_graph, utils_metrics, utils_spec_augmentation
from models import base_gnn, base_cnn
import matplotlib 
import pandas as pd 
import tensorflow as tf 
import random 
import numpy as np
import time 







def main():

    # For reproducibility & testing across different models (Note that we trained models for comparison on seed 30,31,32)
    tf.random.set_seed(32)

    # Get the tensorflow version
    print(f"Tensorflow version: {tf.__version__}")

    # Check GPU availability
    # Please Note that this code is made for CPU (since we initially tested on CPU). We also have a .ipynb that is GPU-agnostic on which we ran most tests in the end. (T4 GPU with extended RAM)
    print("Is GPU available:", tf.test.is_gpu_available())
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # Download the Dataset 
    data_dir = utils_data.download_and_prepare_dataset()

    for i in range(1):
        # Set to true if the model is to be trained
        TRAINING = False


        # Set the mode of the model (i.e. CNN or GNN)
        MODE_MODEL = 'CNN'


        SAMPLE_RATE = 16000 # given in the dataset
        FRAME_LENGTH = int(SAMPLE_RATE * 0.025)  # 25 ms
        FRAME_STEP = int(SAMPLE_RATE * 0.010)  # 10 ms
        
        BATCH_SIZE = 64

        # Set the number of dilation layers (i.e. k creates the undilated adjacency matrix and k-1 dilated adjacency matrices)
        N_DILATION_LAYERS = 5


        # Whether to reduce the node representation of the graph (i.e. pooling over the 98 frames in groups of size k)
        REDUCED_NODE_REP_BOOL = True
        REDUCED_NODE_REP_K = 2


        # set the mode of the adjacency matrix ('window', 'similarity', 'cosine window')
        ADJ_MODE = 'cosine window'
        # Whether to use gammatone filters or not (i.e. GFCCs)
        GAMMATONE = False
        # Whether to use SpecAugment or not
        SPEC_AUG = False
        # Wheter to use the Time Shift or not 
        TIME_SHIFT = True

        # Parameters for the adjacency matrix creation
        # Cosine sliding window size (for cosine 'window')
        WINDOW_SIZE_COSINE = 5
        # Simple sliding window size (for 'window')
        WINDOW_SIZE_SIMPLE = 5
        # Cosine window threshold (for 'cosine window')
        COSINE_WINDOW_THRESH = 0.3
        # Threshold for the similarity adjacency matrix (for 'similarity')
        SIMILARITY_THRESH = 0.3

        # Load data
        train_files, train_labels, val_files, val_labels, test_files, test_labels, class_to_index = utils_data.load_audio_dataset(
            data_dir=data_dir,
            validation_file=os.path.join(data_dir, "validation_list.txt"),
            test_file=os.path.join(data_dir, "testing_list.txt")
        )
        

        # Check the data split (predefined in the text files)
        total = len(train_files) + len(val_files) + len(test_files)
        print(f"Percentage of training samples: {len(train_files)/total*100:.2f}%")
        print(f"Percentage of validation samples: {len(val_files)/total*100:.2f}%")
        print(f"Percentage of testing samples: {len(test_files)/total*100:.2f}%")

        # Total number of samples 
        print("Total Number Samples :" , total)

        # Load into pandas dataframes
        train_df = pd.DataFrame({'file_path': train_files, 'label': list(map(lambda x : utils_data.idx_to_label_conversion(x,class_to_index), train_labels)), 'split': 'train'})
        val_df = pd.DataFrame({'file_path': val_files, 'label': list(map(lambda x : utils_data.idx_to_label_conversion(x, class_to_index), val_labels)), 'split': 'val'})
        test_df = pd.DataFrame({'file_path': test_files, 'label': list(map(lambda x : utils_data.idx_to_label_conversion(x,class_to_index), test_labels)), 'split': 'test'})
        # Concatenate the dataframes
        all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

        # To visualize the data distribution (but also already given in Warden Paper, so not necessary)
        #visualize_data_distribution(all_df)


        # Analysis on some random samples
        # Set random seed for reproducibility
        random.seed(42)


        wavs = []
        wav_shapes = []
        names_labels = []
        sample_rates = []

        # Randomly sample 100 files from the training set
        train_examples = random.sample(list(zip(train_files, list(map(lambda x : utils_data.idx_to_label_conversion(x, class_to_index), train_labels)))), 100)

        # Extract some information from the audio files
        for idx,elem in enumerate(train_examples):
            wav, sample_rate = utils_data.read_path_to_wav(elem[0])
            wavs.append(tf.squeeze(wav, axis = -1))
            names_labels.append(elem[1])
            wav_shapes.append(tf.shape(tf.squeeze(wav, axis = -1)))
            sample_rates.append(sample_rate)

        
        # Showing that we need padding/trimming
        if not all(shape == 16000 for shape in wav_shapes):
            print("Not all audio files have the same shape!")


        # Visualize the waveforms (of the first 16 samples)
     #   utils_data.visualize_waveforms(wavs[:16], names_labels[:16])


        # Visualize waveforms of a specific class
     #   utils_data.visualize_wavs_by_class(wavs, names_labels, class_name = 'eight')

        # Selecting a specific wav file (in this case, one with label 'left')
        EXAMPLE = 4

     #   utils_data.visualize_single_waveform(wavs[EXAMPLE], label = names_labels[EXAMPLE])

        wav_padded = utils_data.add_padding_or_trimming(wavs[EXAMPLE], target_length = 16000, padding_mode = 'zeros')


       # utils_data.visualize_single_waveform(wav = wav_padded, label = names_labels[EXAMPLE])


        # We listen to the maximum noise and minimum noise that is added to the samples
        wav_noisy_hard = utils_data.add_noise(wav_padded, noise_dir = 'speech_commands_v0.02/_background_noise_', noise_type = 'exercise_bike', specific_snr = -5)

        wav_noisy_little = utils_data.add_noise(wav_padded, noise_dir = 'speech_commands_v0.02/_background_noise_', noise_type = 'exercise_bike', specific_snr = 10)
        

        # Listen to the audio
    #    utils_data.listen_audio(wavs[EXAMPLE], sample_rate = sample_rates[EXAMPLE].numpy())
        # Listen to the really noisy audio
    #    utils_data.listen_audio(wav_noisy_hard, sample_rate = sample_rates[EXAMPLE].numpy())
        # Listen to the little noisy audio
     #   utils_data.listen_audio(wav_noisy_little, sample_rate = sample_rates[EXAMPLE].numpy())
        # Listen to the padded audio
       # utils_data.listen_audio(wav_padded, sample_rate = sample_rates[EXAMPLE].numpy())
        # Look at the very noisy waveform
    #    utils_data.visualize_single_waveform(wav_noisy_hard, names_labels[EXAMPLE])
        # Look at the little noisy waveform
    #    utils_data.visualize_single_waveform(wav_noisy_little, names_labels[EXAMPLE])


        ## Spectrogram
        spectrogram, _ = utils_data.get_spectrogram(wav_padded, sample_rates[EXAMPLE].numpy())
        spectrogram_noisy_hard, _ = utils_data.get_spectrogram(wav_noisy_hard, sample_rates[EXAMPLE].numpy())
        spectrogram_noisy_little, _ = utils_data.get_spectrogram(wav_noisy_little, sample_rates[EXAMPLE].numpy())

        # SpecAugment
        spectrogram_augmented = utils_spec_augmentation.spec_augment_easy(spectrogram = spectrogram)

       # utils_data.visualize_single_spectrogram(spectrogram_augmented.numpy(), frame_step = FRAME_STEP)


        ## Spectrogram vs Waveform 

    #    utils_data.visualize_waveform_and_spectrogram(wav_padded, spectrogram.numpy(), FRAME_STEP, names_labels[EXAMPLE], sample_rate = SAMPLE_RATE)
    #    utils_data.visualize_waveform_and_spectrogram(wav_noisy_hard, spectrogram_noisy_hard.numpy(), FRAME_STEP, names_labels[EXAMPLE], sample_rate = SAMPLE_RATE)
    #    utils_data.visualize_waveform_and_spectrogram(wav_noisy_little, spectrogram_noisy_little.numpy(), FRAME_STEP, names_labels[EXAMPLE], sample_rate = SAMPLE_RATE)


        # Create Filterbanks
        log_mel_spectrogram, mel_filters = utils_data.apply_mel_filterbanks(spectrogram)
        log_gammatone_spectrogram = utils_data.apply_gammatone_filterbanks(spectrogram)
        gam_filters = utils_data.create_gammatone_filterbank(fft_size = FRAME_LENGTH)


        # Mel Filterbank
   #     utils_data.visualize_filterbank(filters = mel_filters , spectrogram = spectrogram) 
        # Gammatone Filterbank
   #     utils_data.visualize_filterbank(filters = gam_filters, spectrogram = spectrogram, gammatone = True)
        


        mfccs = utils_data.get_mfccs(log_mel_spectrogram, wav = wav_padded, frame_length = FRAME_LENGTH, frame_step = FRAME_STEP)
        gfccs = utils_data.get_gnccs(log_gammatone_spectrogram = log_gammatone_spectrogram, wav = wav_padded,frame_length = FRAME_LENGTH, frame_step = FRAME_STEP)

        # MFCCs
        utils_data.visualize_mfccs(mfccs, label = 'MFCC')

        # GFCCs
        utils_data.visualize_mfccs(gfccs, label = 'GFCC', gammatone = True)






        # Creation of the basic datasets (this can be used for the CNN model, for the GNN model we need to create the adjacency matrices)
        train_ds_og, val_ds_og, test_ds_og = utils_data.create_tf_dataset(train_files, train_labels, sample_rate= SAMPLE_RATE, 
                                                                frame_length = FRAME_LENGTH, frame_step= FRAME_STEP,
                                                                mode = 'train', gammatone = GAMMATONE, noise = True, noise_prob = 0.8, spec_augmentation = SPEC_AUG, spec_prob = 0.8, time_shift = TIME_SHIFT),\
                                    utils_data.create_tf_dataset(val_files, val_labels,sample_rate= SAMPLE_RATE, 
                                                                frame_length = FRAME_LENGTH, frame_step= FRAME_STEP,
                                                                mode = 'val', gammatone = GAMMATONE, spec_augmentation = False),\
                                    utils_data.create_tf_dataset(test_files, test_labels, sample_rate= SAMPLE_RATE, 
                                                                frame_length = FRAME_LENGTH, frame_step= FRAME_STEP,
                                                                mode = 'test', gammatone = GAMMATONE, spec_augmentation = False)
        


        


        # This is to train the low-footprint CNN model as presented in the Tang et al. paper, (res8 narrow)
        if MODE_MODEL == 'CNN':


            if TRAINING:
                model, history = base_cnn.train_model(train_ds_og, val_ds_og, test_ds_og, input_shape=(98, 39, 1), num_classes=35, epochs=30, model_type='res8_narrow')

            else:
                model = base_cnn.create_res8_narrow_model(input_shape=(98, 39, 1), num_classes=35)

                model.load_weights('saved_models/cnn_res8narrow.h5')



            # Removing wav file 
            test_ds_og = base_cnn.preprocess_data(test_ds_og
                                    , is_training=False)

            y_pred, y_true = utils_metrics.get_ys(test_ds_og, model)


            utils_metrics.visualize_confusion_matrix(y_pred, y_true, idx = i)
            # Precision, Recall, F1-score
            start_time = time.time()
            utils_metrics.metrics_evaluation(y_pred, y_true, model_name = f"Model {i}") 
            end_time = time.time()

            test_time = end_time - start_time 
            print("Test Time : ", test_time)

            # res8-narrow : 0.008289813995361328




        else:

            # Reduce the node representation of our graph by pooling over the 98 frames in groups of size k
            if REDUCED_NODE_REP_BOOL:
                train_ds_og = train_ds_og.map(lambda mfcc, wav, label : utils_graph.get_reduced_representation(mfcc,wav,label, k = REDUCED_NODE_REP_K, pooling_type = 'mean'))
                val_ds_og = val_ds_og.map(lambda mfcc, wav, label : utils_graph.get_reduced_representation(mfcc,wav,label, k = REDUCED_NODE_REP_K, pooling_type = 'mean'))
                test_ds_og = test_ds_og.map(lambda mfcc, wav, label : utils_graph.get_reduced_representation(mfcc,wav,label, k = REDUCED_NODE_REP_K, pooling_type = 'mean'))



            for mfcc, wav, label in train_ds_og.take(1):
                example_mfcc = mfcc
                example_wav = wav
                example_label = label
                N_FRAMES = tf.shape(mfcc)[0] # Might be updated due to reduced node representation
                N_MFCCS = tf.shape(mfcc)[1]


            print(f"Number of frames: {N_FRAMES}")
            print(f"Number of MFCCs: {N_MFCCS}")


            
            # Creation of Dataset for the GNNs (i.e. with adjacency matrices)
            train_ds = train_ds_og.map(lambda mfcc, wav, label: utils_graph.create_adjacency_matrix(mfcc, N_FRAMES, label, mode= ADJ_MODE,window_size_cosine = WINDOW_SIZE_COSINE, cosine_window_thresh = COSINE_WINDOW_THRESH, n_dilation_layers= N_DILATION_LAYERS, window_size=  WINDOW_SIZE_SIMPLE, threshold = SIMILARITY_THRESH))
            val_ds = val_ds_og.map(lambda mfcc, wav, label: utils_graph.create_adjacency_matrix(mfcc, N_FRAMES, label, mode= ADJ_MODE,window_size_cosine =  WINDOW_SIZE_COSINE, cosine_window_thresh = COSINE_WINDOW_THRESH, n_dilation_layers= N_DILATION_LAYERS, window_size=  WINDOW_SIZE_SIMPLE, threshold = SIMILARITY_THRESH))
            test_ds = test_ds_og.map(lambda mfcc, wav, label: utils_graph.create_adjacency_matrix(mfcc, N_FRAMES, label,mode= ADJ_MODE,window_size_cosine =  WINDOW_SIZE_COSINE, cosine_window_thresh = COSINE_WINDOW_THRESH, n_dilation_layers= N_DILATION_LAYERS, window_size =  WINDOW_SIZE_SIMPLE, threshold = SIMILARITY_THRESH))
            
            # Check the shape of the dataset
            for mfcc, adjacency_matrices, label in train_ds.take(1):
                print(f"MFCC shape: {mfcc.shape}")
                print(f"Label shape: {label.shape}")
                example_mfcc = mfcc
                adjacency_matrix_undilated = adjacency_matrices[0]
                adjacency_matrix_dilated_2 = adjacency_matrices[1]
                adjacency_matrix_dilated_8 = adjacency_matrices[4]




           # utils_graph.visualize_adjacency_matrix(adjacency_matrix_undilated, title="Adjacency Matrix Undilated")
           # utils_graph.visualize_adjacency_matrix(adjacency_matrix_dilated_2, title="Adjacency Matrix Dilated (2)")
           # utils_graph.visualize_adjacency_matrix(adjacency_matrix_dilated_8, title="Adjacency Matrix Dilated (8)")

            





            ## Calculation of Multiplications for the GNN models

            # In the case of the fixed window approaches, we calculate an upper bound where all edges are present (independent of threshold) ;
            # Therefore, we note that the actual number of multiplications will be lower in general.
            if ADJ_MODE == 'cosine window':
                num_edges = N_FRAMES * WINDOW_SIZE_COSINE - N_FRAMES # for the window approaches, we calculate an upper bound where all edges are present (independent of threshold)
            elif ADJ_MODE == 'window':
                num_edges = N_FRAMES * WINDOW_SIZE_SIMPLE - N_FRAMES
            # Use this to estimate number of edges for usage with similarity adjacency matrix (as the exact number of edges is not known beforehand) ; in general,
            # using some average over multiple examples would be a better idea, but since we find in our analysis that the similarity approach is not very useful,
            # we also didn't spend too much time on this.
            # NOTE : this didn't make it in the report in the end due to already having 9 pages & not being that efficient
            elif ADJ_MODE == 'similarity':
                num_edges = utils_metrics.count_edges(adjacency_matrix_undilated)



            mults = utils_metrics.calculate_multiplications('base_gcn', feature_dim = 32, num_edges = num_edges, message_dim = 32, next_state_dim = 32,
                                                            message_layers = 5, reduced = False, k_reduced = 0,
                                                            num_heads = 0, per_head_channels=128, use_layer_normalization=True, init_node_enc = 'normal')
            print("Number of Multiplications :" , mults)


            
            # Finally, we create our final dataset, which puts MFCCs/GFCCs & adjacency matrices together in a graph (based on the GNN tensorflow API)
            train_ds = train_ds.map(lambda mfcc, adjacency_matrices, label: base_gnn.mfccs_to_graph_tensors_for_dataset(mfcc, adjacency_matrices, label, reduced_node_bool= REDUCED_NODE_REP_BOOL, reduced_node_k= REDUCED_NODE_REP_K))
            val_ds = val_ds.map(lambda mfcc, adjacency_matrices, label:  base_gnn.mfccs_to_graph_tensors_for_dataset(mfcc, adjacency_matrices, label, reduced_node_bool= REDUCED_NODE_REP_BOOL, reduced_node_k= REDUCED_NODE_REP_K))
            test_ds = test_ds.map(lambda mfcc, adjacency_matrices, label:  base_gnn.mfccs_to_graph_tensors_for_dataset(mfcc, adjacency_matrices, label, reduced_node_bool= REDUCED_NODE_REP_BOOL, reduced_node_k= REDUCED_NODE_REP_K))


            
            # prefetch the datasets
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

        
            # Set i == 0 for the model that should be trained/tested 
            if i == 1:
                base_model = base_gnn.base_gnn_model(graph_tensor_specification = graphs_spec,
                                            n_message_passing_layers = 5,
                                            use_residual_next_state = True,
                                            initial_nodes_mfccs_layer_dims= 32,
                                            next_state_dim = 32,
                                            message_dim = 32,
                                            initial_state_mfcc_mode = 'splitted',
                                            context_mode = 'mean',
                                            dropout_rate = 0.2,
                                            dilation = False,
                                            n_dilation_layers= 0,
                                            l2_reg_factor= 1e-4,
                                            )
                
            if i == 1:
                base_model = base_gnn.base_gnn_with_context_node_model(
                                            graph_tensor_specification = graphs_spec,
                                            initial_nodes_mfccs_layer_dims = 64,
                                            message_dim = 64,
                                            next_state_dim = 64,
                                            l2_reg_factor = 1e-4,
                                            dropout_rate = 0.2,
                                            use_layer_normalization = True,
                                            n_message_passing_layers = 5,
                                            dilation = False,
                                            n_dilation_layers = 0,
                                            context_mode = 'mean',
                                            initial_state_mfcc_mode = 'normal',
                                            )
            # SHOWCASE 0 (gnn_weighted_nospec), # SHOWCASE 1 (gnn_weighted_spec), # SHOWCASE 2 (gnn_weighted_nospec_reduced_2) # SHOWCASE 3 (gnn_weighted_nospec_reduced_4)
            if i == 1:
                base_model = base_gnn.base_gnn_weighted_model(
                                            graph_tensor_specification = graphs_spec,
                                            initial_nodes_mfccs_layer_dims = 64,
                                            message_dim = 64,
                                            next_state_dim = 64,
                                            l2_reg_factor = 1e-4,
                                            dropout_rate = 0.2,
                                            use_layer_normalization = True,
                                            n_message_passing_layers = 5,
                                            dilation = True,
                                            n_dilation_layers = N_DILATION_LAYERS,
                                            context_mode = 'mean',
                                            initial_state_mfcc_mode = 'splitted',
                                            use_residual_next_state= True,
                                            )

            if i == 1:
                base_model = base_gnn.base_gnn_weighted_with_context_model(
                                            graph_tensor_specification = graphs_spec,
                                            initial_nodes_mfccs_layer_dims = 64,
                                            message_dim = 64,
                                            next_state_dim = 64,
                                            l2_reg_factor = 1e-4,
                                            dropout_rate = 0.2,
                                            use_layer_normalization = True,
                                            n_message_passing_layers = 5,
                                            dilation = True,
                                            n_dilation_layers = N_DILATION_LAYERS,
                                            context_mode = 'mean',
                                            initial_state_mfcc_mode = 'normal',
                                            use_residual_next_state= False,
                                            )
            if i == 1:
                base_model = base_gnn.base_gnn_weighted_with_context_model_v2(
                                            graph_tensor_specification = graphs_spec,
                                            initial_nodes_mfccs_layer_dims = 64,
                                            message_dim = 64,
                                            next_state_dim = 64,
                                            l2_reg_factor = 1e-4,
                                            dropout_rate = 0.2,
                                            use_layer_normalization = True,
                                            n_message_passing_layers = 5,
                                            dilation = True,
                                            n_dilation_layers = N_DILATION_LAYERS,
                                            context_mode = 'mean',
                                            initial_state_mfcc_mode = 'splitted',
                                            use_residual_next_state= True,
                                            )


            # SHOWCASE 4 (gcn_nospec_normalnodeenc) # SHOWCASE 5 (gcn_spec_normalnodeenc) # SHOWCASE 6 (gcn_nospec_normalnodeenc_reduced_2) # SHOWCASE 7 (gcn_nospec_normalnodeenc_reduced_4)
            if i == 1:
                base_model = base_gnn.base_gnn_model_using_gcn(graph_tensor_specification = graphs_spec,
                                                                n_message_passing_layers = 5,
                                                                use_residual_next_state = True,
                                                                initial_nodes_mfccs_layer_dims= 32,
                                                                next_state_dim = 32,
                                                                message_dim = 32,
                                                                dropout_rate = 0.2,
                                                                initial_state_mfcc_mode = 'normal',
                                                                context_mode = 'mean',
                                                                dilation = True,
                                                                n_dilation_layers= N_DILATION_LAYERS,
                                                                l2_reg_factor= 1e-4,
                                                                )
            if i == 1:
                base_model = base_gnn.GCN_model(graph_tensor_specification = graphs_spec,
                                                                n_message_passing_layers = 3,
                                                                use_residual_next_state = True,
                                                                initial_nodes_mfccs_layer_dims= 32,
                                                                next_state_dim = 32,
                                                                message_dim = 32,
                                                                dropout_rate = 0.2,
                                                                initial_state_mfcc_mode = 'splitted',
                                                                context_mode = 'mean',
                                                                dilation = True,
                                                                n_dilation_layers= N_DILATION_LAYERS,
                                                                l2_reg_factor= 1e-4,
                                                                )
                
            # NOTE : We did not test this model in the end as we didn't have time.
            if i == 1:
                base_model = base_gnn.base_gnn_model_learning_edge_weights(graph_tensor_specification = graphs_spec,
                                            n_message_passing_layers = 3,
                                            use_residual_next_state = True,
                                            initial_nodes_mfccs_layer_dims= 64,
                                            next_state_dim = 64,
                                            message_dim = 64,
                                            initial_state_mfcc_mode = 'normal',
                                            context_mode = 'mean',
                                            dropout_rate = 0.2,
                                            dilation = False,
                                            n_dilation_layers= N_DILATION_LAYERS,
                                            l2_reg_factor= 1e-4,
                                            )
            # SHOWCASE 8 (gatgcnv2_nospec) # SHOWCASE 9 (gatgcnv2_spec) # SHOWCASE 10 (gatgcnv2_nospec_reduced_2)
            if i == 0:
                base_model = base_gnn.GAT_GCN_model_v2(graph_tensor_specification = graphs_spec,
                                                                n_message_passing_layers = 5,
                                                                use_residual_next_state = True,
                                                                initial_nodes_mfccs_layer_dims= 64,
                                                                next_state_dim = 64,
                                                                message_dim = 64,
                                                                initial_state_mfcc_mode = 'splitted',
                                                                dilation = True,
                                                                dropout_rate = 0.2,
                                                                n_dilation_layers= N_DILATION_LAYERS,
                                                                l2_reg_factor= 1e-4,
                                                                num_heads= 5,
                                                                per_head_channels= 128
                                                                )
            if i == 1:
                base_model = base_gnn.GAT_GCN_model(graph_tensor_specification = graphs_spec,
                                                                n_message_passing_layers = 3,
                                                                use_residual_next_state = True,
                                                                initial_nodes_mfccs_layer_dims= 64,
                                                                next_state_dim = 64,
                                                                message_dim = 64,
                                                                initial_state_mfcc_mode = 'normal',
                                                                dilation = True,
                                                                dropout_rate = 0.2,
                                                                n_dilation_layers= N_DILATION_LAYERS,
                                                                l2_reg_factor= 1e-4,
                                                                num_heads= 5,
                                                                per_head_channels= 64
                                                                )
                
            if i == 1:
                base_model = base_gnn.base_GATv2_model(graph_tensor_specification = graphs_spec,
                                                                n_message_passing_layers = 2,
                                                                use_residual_next_state = False,
                                                                initial_nodes_mfccs_layer_dims= 64,
                                                                next_state_dim = 32,
                                                                message_dim = 32,
                                                                initial_state_mfcc_mode = 'conv',
                                                                dilation = False,
                                                                dropout_rate = 0.2,
                                                                n_dilation_layers= N_DILATION_LAYERS,
                                                                l2_reg_factor= 1e-4,
                                                                num_heads= 3,
                                                                per_head_channels= 128
                                                                )
        
            # Model Summary
            print(base_model.summary())



            if TRAINING:

                start_time = time.time()

                # Train the model
                history = base_gnn.train(model = base_model,
                                        train_ds = train_ds,
                                        val_ds = val_ds,
                                        test_ds = test_ds,
                                        epochs = 30,
                                        batch_size = BATCH_SIZE,
                                        learning_rate = 0.001)


                end_time = time.time()
                training_time = end_time - start_time

                print(f"Training completed in {training_time/60:.2f} minutes")
            
            # Load the best weights of a model instead of training 
            else:
                # Load the best weights of the model (Note that the correct model architecture has to be set using i == 0)
                base_model.load_weights('saved_models/gatgcnv2_nospec_reduced_2.h5')
            

            # utils_metrics.plot_history(history, columns=['loss', 'sparse_categorical_accuracy'], idx = i)

            # Confusion matrix visualization
            y_pred, y_true = utils_metrics.get_ys(test_ds, base_model)
            utils_metrics.visualize_confusion_matrix(y_pred, y_true, idx = i)

            # Precision, Recall, F1-score
            start_time = time.time()
            utils_metrics.metrics_evaluation(y_pred, y_true, model_name = f"Model {i}")
            end_time = time.time()
            test_time = end_time - start_time

            print("Test Time : ", test_time)

            # gnn weighted nospec : 0.00815582275390625
            # gnn weighted spec :  0.007980823516845703
            # gnn weighted nospec reduced 2 : 0.008828878402709961
            # gnn weighted nospec reduced 4 : 0.00882101058959961
            # gcn nospec normalnodeenc : 0.00865316390991211
            # gcn spec normalnodeenc : 0.008502006530761719
            # gcn nospec normalnodeenc reduced 2 : 0.008662939071655273
            # gcn nospec normalnodeenc reduced 4 : 0.00885629653930664
            # gatgcnv2 nospec : 0.008797168731689453
            # gatgcnv2 spec : 0.008181095123291016
            # gatgcnv2 nospec reduced 2 : 0.007928609848022461


    
        


if __name__ == '__main__':
    main()