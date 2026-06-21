import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def get_data_xor(n_samples=200, seed=None):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n_samples, 2))
    y = np.sign(X[:, 0] * X[:, 1])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test