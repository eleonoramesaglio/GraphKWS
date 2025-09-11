import tensorflow as tf



def residual_block_narrow(inputs, filters):
    """
    Create a residual block for res8_narrow.
    
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



def create_res8_narrow_model(input_shape, num_classes=12):
    """
    Create the res8-narrow model as described in the Tang et al. paper.
    
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
    
    # First average pooling layer (4x3)
    x = tf.keras.layers.AveragePooling2D(pool_size=(4, 3))(x)
    
    # Chain of 3 residual blocks (res8_narrow has 3 blocks)
    for _ in range(3):
        x = residual_block_narrow(x, feature_maps)
    
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



def train_model(train_ds, val_ds, test_ds, input_shape, num_classes=12, epochs=26, model_type ='res8_narrow'):
    """
    Train the res8 narrow model
    
    """
    # Process datasets
    train_ds = preprocess_data(train_ds, is_training=True)
    val_ds = preprocess_data(val_ds, is_training=False)
    
    # Create model
    if model_type == 'res8_narrow':
        model = create_res8_narrow_model(input_shape, num_classes)

    # Print model summary
    print(model.summary())
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.1, momentum=0.9),
        # not from logits, since we use softmax in the last layer
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    # LR scheduler as defined in Tang et al. Paper
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=5,
                min_lr=0.00001
            ),
        

        # Save model weights
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


    
    return model, history