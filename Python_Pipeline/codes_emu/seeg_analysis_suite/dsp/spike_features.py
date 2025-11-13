from sklearn.decomposition import PCA

def extract_features(spike_waveforms, n_components=3):
    """
    Extracts features from the spike waveforms.
    """
    pca = PCA(n_components=n_components)
    features = pca.fit_transform(spike_waveforms)
    return features.astype(float).tolist()