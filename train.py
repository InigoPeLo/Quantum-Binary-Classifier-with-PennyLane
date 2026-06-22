
from circuito import circuito
import pennylane as qml
from pennylane import numpy as np
import argparse
from plot import plot_loss, plot_limite, plot_dataset
#Usamos una función loss, entropia cruzada no vale por usar valores entre -1 y +1

def loss(theta, X, y):
    predictions = np.array([circuito(x, theta) for x in X])
    #Calcula MSE entre predicción y etiqueta
    return np.mean((predictions - y) ** 2)


#Optimizador



def train(X_train, y_train, layers=2, epochs=50, lr=0.1):
    """    Entrena el clasificador cuántico optimizando los parámetros theta.
    Args:
        X_train: array (N, 2) con los datos escalados en el intervalo [0, pi]
        y_train: array (N,) con etiquetas en {-1, +1}
        layers: número de capas del ansatz
        epochs: número de iteraciones de optimización
    Returns:
        theta: array (layers, 2) con los parámetros optimizados del modelo"""
    theta = np.random.uniform(0, np.pi, size=(layers, 2), requires_grad=True)
    opt=qml.AdamOptimizer(stepsize=lr)
    
    def cost(theta):
        return loss(theta, X_train, y_train)
    
    loss_hist= []
    for i in range(epochs):
        theta, loss_val = opt.step_and_cost(cost, theta)
        loss_hist.append(float(loss_val))
        if i % 10 == 0:
            print(f'Epoch {i} | loss {loss_val:.4f}')
    
    return theta, loss_hist

def accuracy(theta, X, y):
    predictions = np.array([circuito(x, theta) for x in X])
    labels = np.sign(predictions)
    return np.mean(labels == y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento del clasificador cuántico")
    parser.add_argument("--epochs", type=int, default=50, help="Número de epochs")
    parser.add_argument("--layers", type=int, default=2, help="Número de capas del ansatz")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate del optimizador Adam")
    parser.add_argument("--dataset", type=str, default="moons", choices=["moons", "xor", "circle"], help="Dataset a usar", required=False)
    parser.add_argument("--seed", type=int, default=123, help="Semilla para la generación de datos", required=False)
    parser.add_argument("--plot", type=bool, default=False, help="Si True plotea el dataset", required=False)
    args = parser.parse_args()


    from data import get_data
    X_train, X_test, y_train, y_test = get_data(dataset=args.dataset, seed=args.seed)

    theta, loss_hist = train(X_train, y_train, layers=args.layers, epochs=args.epochs, lr=args.lr)
    acc = accuracy(theta, X_test, y_test)
    print(f"Accuracy en test: {acc:.2%}")
    plot_loss(loss_hist)
    plot_limite(theta, X_test, y_test)

    if args.plot:
        plot_dataset(X_train, y_train, title=args.dataset)

