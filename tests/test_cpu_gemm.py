import os
import torch
import numpy as np
import time

# Set environment variables for underlying libraries
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['OPENBLAS_NUM_THREADS'] = '16'
os.environ['NUMEXPR_NUM_THREADS'] = '16'

# Set the number of CPU threads for PyTorch
torch.set_num_threads(4)
print(f"Number of threads set: {torch.get_num_threads()}")

# Perform a sample GEMM operation
A = torch.randn(500, 500)
B = torch.randn(500, 500)

# Ensure both matrices are on the CPU
A = A.to('cpu')
B = B.to('cpu')

# Perform matrix multiplication
start = time.time()
for i in range(100):
    result = torch.mm(A, B)
duration = time.time() - start
print("Time ", duration)
print("GEMM operation completed")