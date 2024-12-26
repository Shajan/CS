To use IBM's Quantum computer (Free Up to 10 minutes/month)
- Create account https://quantum.ibm.com/
- Copy API Token from https://quantum.ibm.com/ and create .env file with one line IBMQ_API_TOKEN=XXXX

Install Python packages
`pip install -r requirements.txt`

Save credentials locally
`python save_credentials.py`
saved to ~/.qiskit/qiskit-ibm.json

Run a simple job to test everyting is setup properly
`python hello_world.py`

NOTE:
- Qiskt is a python library for expressing quantum alogrithms
- The code needs to be converted and uploaded to IBM datacenter where it gets queued and then executed
- See example.py to get an idea of classical vs quantum way of solving problems (does not execute yet, in progress)

