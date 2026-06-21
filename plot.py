import matplotlib.pyplot as plt
import numpy as np
from circuito import circuito

def plot_loss(loss_hist):
    plt.plot(loss_hist)
    plt.title("Evolución del loss en función de los epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("loss.png")
    plt.close()
    
def plot_limite(theta, X_test, y_test):
    x1=np.linspace(0, np.pi, 100)
    x2=np.linspace(0, np.pi, 100)
    xx1,xx2=np.meshgrid(x1, x2)
    #Aplanamos los puntos del grid

    puntos=np.c_[xx1.ravel(), xx2.ravel()]

    #Evaluamos el circuito
    Z=np.array([circuito(p, theta) for p in puntos])
    
    #Lo volvemos a colocar como un grid

    Z=Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k')
    plt.savefig("boundary.png")
    plt.close() 

  