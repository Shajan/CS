from dotenv import load_dotenv
import os

# Retrieve the API token from the .env file
load_dotenv()
api_token = os.getenv('IBMQ_API_TOKEN')

if not api_token:
    print("Error: API token not found in the .env file. Please check.")

from qiskit_ibm_runtime import QiskitRuntimeService
 
QiskitRuntimeService.save_account(
  token=api_token,
  channel="ibm_quantum" # `channel` distinguishes between different account types
)
