"""
# Classical search in a list
def classical_search(lst, target):
    for i, val in enumerate(lst):
        if val == target:
            return i  # Return the index of the target
    return -1  # Not found

lst = [1, 3, 5, 7, 9, 11, 13, 15]
target = 7
index = classical_search(lst, target)
print(f"Target found at index: {index}")
"""

# Simplified quantum equivalent using Grover's Algorithm:

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import GroverOperator
from qiskit.algorithms import AmplificationProblem
from qiskit.algorithms import Grover
from qiskit.providers.aer import AerSimulator

# Define the oracle for the target element
def create_oracle(target, num_qubits):
    oracle = QuantumCircuit(num_qubits)
    oracle.x(target)  # Apply X gate to mark the target
    return oracle.to_gate()

# Number of qubits needed
num_qubits = 3  # Enough for a list of size 8
target = 2  # Target element encoded as index 2
oracle = create_oracle(target, num_qubits)

# Create the Grover operator and problem
grover_op = GroverOperator(oracle)
problem = AmplificationProblem(oracle=oracle)
grover = Grover(quantum_instance=Aer.get_backend('qasm_simulator'))
result = grover.amplify(problem)
print(f"Measured result: {result}")

