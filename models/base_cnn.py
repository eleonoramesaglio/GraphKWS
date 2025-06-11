import tensorflow as tf



"""
def mfccs_to_cnn_tensors_for_dataset(mfcc, label):
    '''
    Convert MFCC features to the appropriate format for CNN processing.
    
    Args:
        mfcc: MFCC features
        label: Class label
    
    Returns:
        A tuple (formatted_mfcc, label)
    '''
    # Ensure current shape of MFCC (98 frames, 39 MFCCs) and reshape to [height, width, channels] format for CNN
    mfcc_cnn = tf.reshape(mfcc, [39, 98, 1])  # [num_coefficients, time_frames, channels]
    
    # Normalization:
    mfcc_cnn = (mfcc_cnn - tf.reduce_mean(mfcc_cnn)) / tf.math.reduce_std(mfcc_cnn)
    
    return mfcc_cnn, label
"""



def residual_block_narrow(inputs, filters):
    """
    Create a residual block for res8-narrow.
    
    Args:
        inputs: Input tensor
        filters: Number of feature maps (n=19 for narrow variant)
    
    Returns:
        Output tensor after applying the residual block
    """
    # First convolutional layer with batch normalization and ReLU
    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        padding='same',
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(1e-5)
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # Second convolutional layer with batch normalization
    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        padding='same',
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(1e-5)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # Skip connection
    output = tf.keras.layers.Add()([inputs, x])
    
    return output




def residual_block(inputs, filters, dilation_width, dilation_height):
    """
    Create a residual block as described in the paper.
    
    Args:
        inputs: Input tensor
        filters: Number of feature maps (n=45 in the paper)
        dilation_width: Width dilation rate
        dilation_height: Height dilation rate
    
    Returns:
        Output tensor after applying the residual block
    """
    # First convolutional layer with batch normalization and ReLU
    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        dilation_rate=(dilation_width, dilation_height),
        padding='same',
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(1e-5)
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # Second convolutional layer with batch normalization
    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        padding='same',
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(1e-5)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # Skip connection
    output = tf.keras.layers.Add()([inputs, x])
    
    return output

def create_res8_narrow_model(input_shape, num_classes=12):
    """
    Create the res8-narrow model as described in the paper.
    
    Args:
        input_shape: Shape of input MFCC features (timesteps, mfcc_features, 1)
        num_classes: Number of output classes
    
    Returns:
        res8-narrow model
    """
    # Number of feature maps for narrow variant
    feature_maps = 19
    
    # Input layer
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Initial convolution layer
    x = tf.keras.layers.Conv2D(
        filters=feature_maps,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_regularizer=tf.keras.regularizers.l2(1e-5)
    )(inputs)
    
    # First average pooling layer (4x3 as specified in Table 2)
    x = tf.keras.layers.AveragePooling2D(pool_size=(4, 3))(x)
    
    # Chain of 3 residual blocks (res8 has 3 blocks instead of 6)
    for _ in range(3):
        x = residual_block_narrow(x, feature_maps)
    
    # Global average pooling and softmax
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model

def create_res15_model(input_shape, num_classes=12, feature_maps=45):
    """
    Create the res15 model as described in the paper.
    
    Args:
        input_shape: Shape of input MFCC features (timesteps, mfcc_features, 1)
        num_classes: Number of output classes
        feature_maps: Number of feature maps (n=45 in the paper)
    
    Returns:
        res15 model
    """
    # Input layer
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Initial convolution layer
    x = tf.keras.layers.Conv2D(
        filters=feature_maps,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_regularizer=tf.keras.regularizers.l2(1e-5)
    )(inputs)
    
    # Chain of 6 residual blocks with exponentially increasing dilation
    for i in range(6):
        dilation_rate =  2 ** (i // 3)
        x = residual_block(x, feature_maps, dilation_rate, dilation_rate)
    
    # Final convolution layer with batch normalization
    x = tf.keras.layers.Conv2D(
        filters=feature_maps,
        kernel_size=(3, 3),
        dilation_rate=(16, 16),  # As specified in the paper
        padding='same',
        use_bias=True,
        kernel_regularizer=tf.keras.regularizers.l2(1e-5)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Global average pooling and softmax
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model

# This function prepares the data for the model
def preprocess_data(dataset, batch_size=64, is_training=True):
    """
    Preprocess the dataset for training or evaluation.
    
    Args:
        dataset: TensorFlow dataset containing MFCC features and labels
        batch_size: Batch size for training/evaluation
        is_training: Whether this is for training (includes shuffling and augmentation)
    
    Returns:
        Processed dataset ready for the model
    """
    # Ignore wav files and only use MFCC features and labels
    dataset = dataset.map(lambda mfcc, wav, label: (mfcc, label))
    
    # Add channel dimension since Conv2D expects (batch, height, width, channels)
    dataset = dataset.map(lambda x, y: (tf.expand_dims(x, axis=-1), y))
    
    if is_training:
        # Shuffle with a buffer size
        dataset = dataset.shuffle(buffer_size=10000)
        

    
    # Batch the data
    dataset = dataset.batch(batch_size)
    
    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset



def train_model(train_ds, val_ds, test_ds, input_shape, num_classes=12, epochs=26, model_type ='res15'):
    """
    Train the res15 model.
    
    Args:
        train_ds: Training dataset
        val_ds: Validation dataset
        input_shape: Shape of input MFCC features
        num_classes: Number of output classes
        epochs: Number of training epochs
    
    Returns:
        Trained model
    """
    # Process datasets
    train_ds = preprocess_data(train_ds, is_training=True)
    val_ds = preprocess_data(val_ds, is_training=False)
    
    # Create model
    if model_type == 'res8_narrow':
        model = create_res8_narrow_model(input_shape, num_classes)
    elif model_type == 'res15':
        # Use the default feature maps for res15
        model = create_res15_model(input_shape, num_classes)
    else:
        raise ValueError("Invalid model type. Choose 'res8_narrow' or 'res15'.")


    # Print model summary
    print(model.summary())
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.1, momentum=0.9),
        # not from logits, since we use softmax in the last layer
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    )


    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=5,
                min_lr=0.00001
            ),
        

        # Tensorflow doesn't provide model saving for GNNs, so we save weights in a checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            filepath='best_model_weights_cnn.h5',  
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,  
            verbose=1)
    ]
    

    
    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

  #  test_measurements = model.evaluate(test_ds)


  #  print(f"Test Loss : {test_measurements[0]:.2f},\
  #        Test Sparse Categorical Accuracy : {test_measurements[1]:.2f}")
    
    return model, history