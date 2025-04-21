import tensorflow as tf
import keras_tuner as kt
import tensorflow_gnn as tfgnn
from models import base_gnn

# First, we'll need to modify the model-building function to accept a HyperParameters object
def build_tunable_gnn_model(hp, graph_tensor_specification, num_classes=35):
    """
    A tunable version of the base_gnn_weighted_model function that works with Keras Tuner.
    
    Args:
        hp: HyperParameters object from keras_tuner
        graph_tensor_specification: Specification for the input graph tensor
        num_classes: Number of output classes
        
    Returns:
        A compiled Keras model
    """
    # Define tunable hyperparameters with reasonable ranges
    initial_nodes_mfccs_layer_dims = hp.Int('initial_nodes_layer_dims', min_value=32, max_value=128, step=32)
    message_dim = hp.Int('message_dim', min_value=64, max_value=256, step=64)
    next_state_dim = hp.Int('next_state_dim', min_value=64, max_value=256, step=64)
    l2_reg_factor = hp.Float('l2_reg_factor', min_value=1e-6, max_value=1e-4, sampling='log')
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    n_message_passing_layers = hp.Int('n_message_passing_layers', min_value=2, max_value=6, step=1)
    use_layer_normalization = hp.Boolean('use_layer_normalization')
    use_residual_next_state = hp.Boolean('use_residual_next_state')
    

    
    # Now call your existing model building function with these hyperparameters
    model = base_gnn.base_gnn_weighted_model(
        graph_tensor_specification=graph_tensor_specification,
        initial_nodes_mfccs_layer_dims=initial_nodes_mfccs_layer_dims,
        message_dim=message_dim,
        next_state_dim=next_state_dim,
        num_classes=num_classes,
        l2_reg_factor=l2_reg_factor,
        dropout_rate=dropout_rate,
        use_layer_normalization=use_layer_normalization,
        n_message_passing_layers=n_message_passing_layers,
        dilation=False,
        n_dilation_layers=1,
        use_residual_next_state=use_residual_next_state
    )
    
    # Compile the model with tunable learning rate
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    
    return model

# Function to run hyperparameter tuning
def tune_gnn_model(graph_tensor_specification, train_ds, val_ds, num_classes=35, max_trials=10, epochs_per_trial=20):
    """
    Run hyperparameter tuning for the GNN model
    
    Args:
        graph_tensor_specification: Specification for the input graph tensor
        train_ds: Training dataset
        val_ds: Validation dataset
        num_classes: Number of output classes
        max_trials: Maximum number of hyperparameter combinations to try
        epochs_per_trial: Number of epochs to train each trial
        
    Returns:
        The best hyperparameters found
    """
    # Define the objective metric to optimize
    objective = kt.Objective('val_sparse_categorical_accuracy', direction='max')
    
    # Create a tuner
    tuner = kt.BayesianOptimization(
        lambda hp: build_tunable_gnn_model(hp, graph_tensor_specification, num_classes),
        objective=objective,
        max_trials=max_trials,
        directory='hyperparameter_tuning',
        project_name='gnn_model_tuning'
    )
    
    # Define early stopping callback for each trial
    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor='val_sparse_categorical_accuracy',
        patience=5,
        restore_best_weights=True
    )
    
    # Start the search
    tuner.search(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_per_trial,
        callbacks=[stop_early]
    )
    
    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    # Print the best hyperparameters
    print("Best hyperparameters found:")
    print(f"initial_nodes_layer_dims: {best_hps.get('initial_nodes_layer_dims')}")
    print(f"message_dim: {best_hps.get('message_dim')}")
    print(f"next_state_dim: {best_hps.get('next_state_dim')}")
    print(f"l2_reg_factor: {best_hps.get('l2_reg_factor')}")
    print(f"dropout_rate: {best_hps.get('dropout_rate')}")
    print(f"n_message_passing_layers: {best_hps.get('n_message_passing_layers')}")
    print(f"use_layer_normalization: {best_hps.get('use_layer_normalization')}")
    print(f"use_residual_next_state: {best_hps.get('use_residual_next_state')}")
    print(f"dilation: {best_hps.get('dilation')}")
    if best_hps.get('dilation'):
        print(f"n_dilation_layers: {best_hps.get('n_dilation_layers_value')}")
    print(f"learning_rate: {best_hps.get('learning_rate')}")
    
    return best_hps

# Function to build the final model with the best hyperparameters
def build_best_model(best_hps, graph_tensor_specification, num_classes=35):
    """
    Build the model with the best hyperparameters
    
    Args:
        best_hps: Best hyperparameters found by the tuner
        graph_tensor_specification: Specification for the input graph tensor
        num_classes: Number of output classes
        
    Returns:
        The model built with the best hyperparameters
    """
    # Get n_dilation_layers based on whether dilation is True or False
    n_dilation_layers = best_hps.get('n_dilation_layers_value', 1) if best_hps.get('dilation') else 1
    
    # Build the model with the best hyperparameters
    best_model = base_gnn.base_gnn_weighted_model(
        graph_tensor_specification=graph_tensor_specification,
        initial_nodes_mfccs_layer_dims=best_hps.get('initial_nodes_layer_dims'),
        message_dim=best_hps.get('message_dim'),
        next_state_dim=best_hps.get('next_state_dim'),
        num_classes=num_classes,
        l2_reg_factor=best_hps.get('l2_reg_factor'),
        dropout_rate=best_hps.get('dropout_rate'),
        use_layer_normalization=best_hps.get('use_layer_normalization'),
        n_message_passing_layers=best_hps.get('n_message_passing_layers'),
        dilation=best_hps.get('dilation'),
        n_dilation_layers=n_dilation_layers,
        use_residual_next_state=best_hps.get('use_residual_next_state')
    )
    
    # Compile the model
    best_model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=best_hps.get('learning_rate')),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    
    return best_model

# Modified train function that can be used with the best model
def train_best_model(model, train_ds, val_ds, test_ds, epochs=50, use_callbacks=True):
    """
    Train the model with the best hyperparameters
    
    Args:
        model: The model to train
        train_ds: Training dataset
        val_ds: Validation dataset
        test_ds: Test dataset
        epochs: Maximum number of epochs to train
        use_callbacks: Whether to use callbacks
        
    Returns:
        Training history
    """
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_sparse_categorical_accuracy',
            patience=10,
            restore_best_weights=True
        )
    ]
    
    # Train the model
    if use_callbacks:
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    else:
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    
    # Evaluate the model
    test_measurements = model.evaluate(test_ds)
    
    print(f"Test Loss: {test_measurements[0]:.2f}, "
          f"Test Sparse Categorical Accuracy: {test_measurements[1]:.2f}")
    
    return history

# Example usage:
