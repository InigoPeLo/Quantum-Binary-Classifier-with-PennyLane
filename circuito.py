import pennylane as qml
from pennylane import numpy as np 

dev=qml.device("default.qubit", wires=2)

@qml.qnode(dev)

def circuito(x, theta):
    l=len(theta)
    #Encoding
    qml.RY(x[0], wires=0)
    qml.RY(x[1], wires=1)
    #Ansazt
    for i in range(l):
        qml.RY(theta[i][0], wires=0)
        qml.RY(theta[i][1],wires=1)
        qml.CNOT(wires=[0,1])
    return qml.expval(qml.PauliZ(0))

