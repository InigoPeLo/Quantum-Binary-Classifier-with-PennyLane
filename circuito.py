import pennylane as qml
from pennylane import numpy as np 

#Con esto podemos formar el circuito según las necisidades del problema
def make_circuit(n_qubits):
    dev=qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)

    def circuito(x, theta):
        l=len(theta)
        #Encoding
        for i in range(n_qubits):
            qml.RY(x[i], wires=i)
        #Ansazt
        for i in range(l):
            for j in range(n_qubits):
                qml.RY(theta[i][j], wires=j)
            for j in range (n_qubits-1):
                qml.CNOT(wires=[j,j+1]) 
        return qml.expval(qml.PauliZ(0))
    return circuito

