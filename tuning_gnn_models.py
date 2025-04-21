import tensorflow as tf
import tensorflow_gnn as tfgnn
import optuna
import numpy as np
import os

def create_optuna_study(gnn_model, graph_tensor_specification, train_ds, val_ds, 
                       num_classes=35, n_trials=10, timeout=None):
    """
    Create and run an Optuna study to find the best hyperparameters for the GNN model.
    
    Args:
        base_gnn_weighted_model: The base model-building function to optimize
        graph_tensor_specification: Specification for the input graph tensor
        train_ds: Training dataset
        val_ds: Validation dataset
        num_classes: Number of output classes
        n_trials: Number of hyperparameter combinations to try
        timeout: Timeout in seconds for the entire study (None means no timeout)
        
    Returns:
        The best parameters found and the Optuna study object
    """
    # Create a directory for saving study results
    os.makedirs("optuna_results", exist_ok=True)
    
    def objective(trial):
        """
        The objective function to minimize.
        Returns validation loss.
        """
        # Sample hyperparameters

        message_dim = trial.suggest_int('message_dim', 64, 256, step=64)
        initial_nodes_mfccs_layer_dims = trial.suggest_int('initial_nodes_layer_dims', 32, 128, step=32)
        next_state_dim = trial.suggest_int('next_state_dim', 64, 256, step=64)
        l2_reg_factor = trial.suggest_float('l2_reg_factor', 1e-6, 1e-4, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1)
        n_message_passing_layers = trial.suggest_int('n_message_passing_layers', 2, 6, step=1)
        use_layer_normalization = trial.suggest_categorical('use_layer_normalization', [True, False])
     #   use_residual_next_state = trial.suggest_categorical('use_residual_next_state', [True, False])
        dilation = trial.suggest_categorical('dilation', [True, False])
        n_dilation_layers = 1
        if dilation:
            n_dilation_layers = trial.suggest_int('n_dilation_layers', 2, 4, step=1)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        
        
        # Create and compile model with sampled parameters
        model = gnn_model(
            graph_tensor_specification=graph_tensor_specification,
            initial_nodes_mfccs_layer_dims=initial_nodes_mfccs_layer_dims,
            message_dim=message_dim,
            next_state_dim=next_state_dim,
            num_classes=num_classes,
            l2_reg_factor=l2_reg_factor,
            dropout_rate=dropout_rate,
            use_layer_normalization=use_layer_normalization,
            n_message_passing_layers=n_message_passing_layers,
            dilation=dilation,
            n_dilation_layers=n_dilation_layers,
            use_residual_next_state=False
        )
        
        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )
        
        # Define early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_sparse_categorical_accuracy',
            patience=2,
            restore_best_weights=True
        )

        # Print the current parameters :

        # Print the hyperparameters for this trial
        print("\nHyperparameters for this trial:")
        print(f"  message_dim: {message_dim}")
        print(f"  initial_nodes_mfccs_layer_dims: {initial_nodes_mfccs_layer_dims}")
        print(f"  next_state_dim: {next_state_dim}")
        print(f"  l2_reg_factor: {l2_reg_factor:.8f}")
        print(f"  dropout_rate: {dropout_rate:.2f}")
        print(f"  n_message_passing_layers: {n_message_passing_layers}")
        print(f"  use_layer_normalization: {use_layer_normalization}")
        print(f"  use_residual_next_state: False")
        print(f"  dilation: {dilation}")
        print(f"  n_dilation_layers: {n_dilation_layers}")
        print(f"  learning_rate: {learning_rate:.6f}")
        print(model.summary())

     

        
        # Train model for a few epochs 
        # Limiting to just 5 epochs for faster hyperparameter search
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=3,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Get the best validation accuracy
        best_val_acc = max(history.history['val_sparse_categorical_accuracy'])
        
        # Clean up to prevent memory leaks
        del model
        tf.keras.backend.clear_session()
        
        # Return negative accuracy because Optuna minimizes by default
        return -best_val_acc
    
    # Create a study that maximizes accuracy (by minimizing negative accuracy)
    study = optuna.create_study(
        direction='minimize',
        study_name='gnn_model_optimization',
        storage=f'sqlite:///optuna_results/gnn_study.db',
        load_if_exists=True
    )
    
    # Run the optimization
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    # Print optimization results
    print("Number of finished trials:", len(study.trials))
    print("Best trial:")
    best_trial = study.best_trial
    print("  Value:", -best_trial.value)  # Convert back to positive accuracy
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
        
    # Return the best parameters and the study
    return best_trial.params, study

def build_model_with_best_params(best_params, gnn_model, graph_tensor_specification, num_classes=35):
    """
    Build a model with the best parameters found by Optuna.
    
    Args:
        best_params: The best parameters found by Optuna
        base_gnn_weighted_model: The base model-building function
        graph_tensor_specification: Specification for the input graph tensor
        num_classes: Number of output classes
        
    Returns:
        A compiled model with the best parameters
    """
    # Extract parameters, using defaults for any that might be missing
    message_dim = best_params.get('message_dim', 128)
    
    # The following parameters will be used when you uncomment the extended parameter search
    initial_nodes_mfccs_layer_dims = best_params.get('initial_nodes_layer_dims', 64)
    next_state_dim = best_params.get('next_state_dim', 128)
    l2_reg_factor = best_params.get('l2_reg_factor', 6e-6)
    dropout_rate = best_params.get('dropout_rate', 0.2)
    use_layer_normalization = best_params.get('use_layer_normalization', True)
    n_message_passing_layers = best_params.get('n_message_passing_layers', 4)
    dilation = best_params.get('dilation', False)
    n_dilation_layers = best_params.get('n_dilation_layers', 2) if dilation else 2
    use_residual_next_state = best_params.get('use_residual_next_state', False)
    learning_rate = best_params.get('learning_rate', 0.001)
    
    # Build model with the best parameters
    model = gnn_model(
        graph_tensor_specification=graph_tensor_specification,
        initial_nodes_mfccs_layer_dims=initial_nodes_mfccs_layer_dims,
        message_dim=message_dim,
        next_state_dim=next_state_dim,
        num_classes=num_classes,
        l2_reg_factor=l2_reg_factor,
        dropout_rate=dropout_rate,
        use_layer_normalization=use_layer_normalization,
        n_message_passing_layers=n_message_passing_layers,
        dilation=dilation,
        n_dilation_layers=n_dilation_layers,
        use_residual_next_state=use_residual_next_state
    )
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    
    return model

def train_best_model(model, train_ds, val_ds, test_ds, epochs=50):
    """
    Train the model with the best hyperparameters
    
    Args:
        model: The model to train
        train_ds: Training dataset
        val_ds: Validation dataset
        test_ds: Test dataset
        epochs: Maximum number of epochs to train
        
    Returns:
        Training history
    """
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_sparse_categorical_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='best_model_weights.h5',
            monitor='val_sparse_categorical_accuracy',
            save_best_only=True,
            save_weights_only=True
        ),
        tf.keras.callbacks.CSVLogger('training_log.csv')
    ]
    
    # Train the model
    history = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=epochs, 
        callbacks=callbacks
    )
    
    # Evaluate the model
    test_measurements = model.evaluate(test_ds)
    
    print(f"Test Loss: {test_measurements[0]:.2f}, "
          f"Test Sparse Categorical Accuracy: {test_measurements[1]:.2f}")
    
    return history

def visualize_optuna_results(study):
    """
    Visualize the results of the Optuna study.
    
    Args:
        study: Optuna study object
    """
    try:
        # Import visualization modules
        from optuna.visualization import plot_optimization_history, plot_param_importances
        from optuna.visualization import plot_contour, plot_slice
        import matplotlib.pyplot as plt
        
        # Plot optimization history
        fig1 = plot_optimization_history(study)
        fig1.write_image("optuna_results/optimization_history.png")
        
        # Plot parameter importances
        fig2 = plot_param_importances(study)
        fig2.write_image("optuna_results/param_importances.png")
        
        # Plot contour of parameters (if there are at least 2 parameters)
        if len(study.best_trial.params) >= 2:
            fig3 = plot_contour(study)
            fig3.write_image("optuna_results/contour.png")
        
        # Plot slice of parameters
        fig4 = plot_slice(study)
        fig4.write_image("optuna_results/slice.png")
        
        print("Visualization images saved in 'optuna_results' directory")
        
    except ImportError:
        print("Could not import visualization modules. Install plotly and matplotlib for visualization.")
    except Exception as e:
        print(f"Visualization failed with error: {e}")

# Example usage:
