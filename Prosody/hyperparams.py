class Hyperparams:
    '''Hyperparameters for TTS vs Real Voice Classification on ASVspoof 2019'''

    # Signal processing
    sr = 16000  # Sample rate: ASVspoof 2019 uses 16 kHz sample rate
    n_fft = 1024  # FFT points (samples): common choice for speech tasks
    frame_shift = 0.01  # 10 ms frame shift: typical for speech processing
    frame_length = 0.025  # 25 ms frame length: suitable for human speech
    hop_length = int(sr * frame_shift)  # Samples (hop length = frame shift * sr)
    win_length = int(sr * frame_length)  # Samples (frame length * sr)
    n_mels = 40  # Number of Mel filters: 40 Mel bins is standard for speech tasks
    power = 1.2  # Exponent for amplifying the magnitude (common for speech signals)
    n_iter = 50  # Number of inversion iterations for spectrogram-to-waveform inversion (if needed)
    preemphasis = 0.97  # Pre-emphasis factor to enhance high frequencies
    max_db = 100  # Maximum decibel value for normalization (scaling Mel spectrogram)
    ref_db = 20  # Reference decibel for normalization (scaling Mel spectrogram)
    
    # Model (For extracting prosody embeddings)
    embed_size = 128  # Dimensionality of the prosody embedding: 128 is a common choice for speech embeddings
    dropout_rate = 0.5  # Dropout rate: Helps prevent overfitting, 50% dropout is commonly used
    num_highwaynet_blocks = 4  # Number of HighwayNet blocks: commonly used for feature extraction in speech tasks
    encoder_num_banks = 16  # Number of encoder banks: parameter for controlling complexity of the encoder
    decoder_num_banks = 8  # Number of decoder banks: related to reconstruction of audio features

    # Classifier Model
    classifier_type = 'SVM'  # You can use an SVM or another classifier for TTS vs real classification
    classifier_kernel = 'rbf'  # Kernel for the SVM (Radial Basis Function)

    # Training parameters
    lr = 0.001  # Learning rate: Standard learning rate for training deep models
    batch_size = 32  # Batch size: This can be adjusted based on available computational resources
    num_iterations = 1000000  # Number of training iterations (you can adjust based on training time)
    
    # Training data details (for ASVspoof 2019 dataset)
    data_dir = "ASVspoof2019"  # Path to ASVspoof 2019 data directory
    train_data = "train.txt"  # Training set file containing the audio paths
    test_data = "test.txt"  # Testing set file containing the audio paths
    ref_audio = "ref_audio/*.wav"  # Reference audio (can be used for comparison in spoofing detection)

