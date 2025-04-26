import tensorflow as tf



def mfccs_to_cnn_tensors_for_dataset(mfcc, label):
    """
    Convert MFCC features to the appropriate format for CNN processing.
    
    Args:
        mfcc: MFCC features
        label: Class label
    
    Returns:
        A tuple (formatted_mfcc, label)
    """
    # Ensure current shape of MFCC (98 frames, 39 MFCCs) and reshape to [height, width, channels] format for CNN
    mfcc_cnn = tf.reshape(mfcc, [39, 98, 1])  # [num_coefficients, time_frames, channels]
    
    # Normalization:
    mfcc_cnn = (mfcc_cnn - tf.reduce_mean(mfcc_cnn)) / tf.math.reduce_std(mfcc_cnn)
    
    return mfcc_cnn, label