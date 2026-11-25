import numpy as np

def generate_synthetic_data(num_sequences=200, 
                            sequence_length=100, 
                            anomaly_fraction=0.1, 
                            block_length=20, 
                            mu=4.0, 
                            random_state=1405):
    """
    Generates synthetic sequence data for the One-Class SVM and HMAD project.
    
    Parameters:
    -----------
    num_sequences : int
        The total number of sequences to generate for this partition.
    sequence_length : int (T)
        The length of each individual sequence.
    anomaly_fraction : float (nu)
        The proportion of anomalous sequences in the generated dataset.
    block_length : int
        The length of the anomalous block inserted into anomalous sequences.
    mu : float
        The shifted mean of the anomalous block distribution N(mu, 1).
    random_state : int
        Seed for reproducibility.
        
    Returns:
    --------
    X : np.ndarray
        Array of shape (num_sequences, sequence_length) containing the data.
    y : np.ndarray
        Binary labels of shape (num_sequences,) where 0 = normal, 1 = anomaly.
    """
    np.random.seed(random_state)
    num_anomalies = int(np.round(num_sequences * anomaly_fraction))
    num_normals = num_sequences - num_anomalies
    
    X = np.zeros((num_sequences, sequence_length))
    y = np.zeros(num_sequences, dtype=int)
    
    X[:num_normals] = np.random.normal(0, 1, size=(num_normals, sequence_length))
    y[:num_normals] = 0
    
    if num_anomalies > 0:
        anomaly_base = np.random.normal(0, 1, size=(num_anomalies, sequence_length))
        
        for i in range(num_anomalies):
            max_start = sequence_length - block_length
            start_idx = np.random.randint(0, max_start + 1)
            end_idx = start_idx + block_length
            
            anomaly_base[i, start_idx:end_idx] = np.random.normal(mu, 1, size=block_length)
            
        X[num_normals:] = anomaly_base
        y[num_normals:] = 1
        
    shuffle_idx = np.random.permutation(num_sequences)
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return X, y

if __name__ == "__main__":
    X_train, y_train = generate_synthetic_data(num_sequences=200, anomaly_fraction=0.0)
    print(f"Training data shape: {X_train.shape} | Anomalies present: {np.sum(y_train)}")
    
    X_test, y_test = generate_synthetic_data(num_sequences=100, anomaly_fraction=0.1)
    print(f"Test data shape: {X_test.shape} | Anomalies present: {np.sum(y_test)}")