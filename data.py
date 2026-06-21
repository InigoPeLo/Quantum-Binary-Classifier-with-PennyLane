from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def get_data(n_samples=200, noise=0.1, seed=None):
    
    x, y=make_moons(n_samples=n_samples, noise=noise, random_state=seed)

    x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=seed)

    scaler=MinMaxScaler(feature_range=(0, np.pi))

    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)

    
    y_test=(y_test*2)-1
    y_train=(y_train*2)-1

    return x_train, x_test, y_train, y_test

