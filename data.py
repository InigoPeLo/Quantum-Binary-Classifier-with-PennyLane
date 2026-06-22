from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def get_data_moons(n_samples=200, noise=0.1, seed=None):
    
    x, y=make_moons(n_samples=n_samples, noise=noise, random_state=seed)

    x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=seed)

    scaler=MinMaxScaler(feature_range=(0, np.pi))

    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)

    
    y_test=(y_test*2)-1
    y_train=(y_train*2)-1

    return x_train, x_test, y_train, y_test

from sklearn.datasets import make_circles

def get_data_circ(n_samples=200, noise=0.1, seed=None):
    
    x, y=make_circles(n_samples=n_samples, noise=noise, random_state=seed, factor=0.3)

    x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=seed)

    scaler=MinMaxScaler(feature_range=(0, np.pi))

    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)

    
    y_test=(y_test*2)-1
    y_train=(y_train*2)-1

    return x_train, x_test, y_train, y_test

def get_data_xor(n_samples=200, seed=None):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n_samples, 2))
    y = np.sign(X[:, 0] * X[:, 1])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

from sklearn.datasets import load_iris

def get_data_iris(n_samples=None, seed=None):
    from sklearn.datasets import load_iris
    
    data = load_iris()
    X = data.data
    y = data.target
    
    # Versicolor vs Virginica, son más parecidos
    mask = y > 0
    X = X[mask]
    y = y[mask]
    
    # Convertir {1, 2} a {-1, +1}
    y = (y - 1) * 2 - 1
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


#Función que llamamos desde train

def get_data(dataset="moons", n_samples=200, seed=None):
    if dataset == "moons":
        return get_data_moons(n_samples=n_samples, seed=seed)
    elif dataset == "xor":
        return get_data_xor(n_samples=n_samples, seed=seed)
    elif dataset == "circle":
        return get_data_circ(n_samples=n_samples, seed=seed)
    elif dataset== "iris":
        return get_data_iris(n_samples=n_samples, seed=seed)