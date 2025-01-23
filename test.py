from fastapi import FastAPI, HTTPException
import uvicorn
import numpy as np
from typing import List

app = FastAPI()

# Initialize M and B as np arrays
M = np.random.rand(5, 5)  # 5x5 random matrix
B = np.random.rand(5)     # 5x1 random vector

# Function to apply sigmoid activation
def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))

# Implement the formula MX + B (manual implementation)
def matrix_multiply_manual(matrix: List[List[float]], vector: List[float]) -> List[float]:
    rows = len(matrix)
    result = []
    for i in range(rows):
        element_sum = 0
        for j in range(len(vector)):
            element_sum += matrix[i][j] * vector[j]
        result.append(element_sum)
    return result

# POST decorator to expose calculation
@app.post("/calculate")
async def calculate(x: List[float]):
    """Calculate MX + B with and without NumPy, then apply sigmoid."""
    # Validate dimensions of X
    if len(x) != 5:
        raise HTTPException(status_code=400, detail="Input vector X must have length 5")

    # Manual calculation
    manual_mx = matrix_multiply_manual(M.tolist(), x)
    manual_result = [mx + b for mx, b in zip(manual_mx, B.tolist())]
    manual_sigmoid = [sigmoid(value) for value in manual_result]

    # NumPy calculation
    numpy_result = np.dot(M, x) + B
    numpy_sigmoid = 1 / (1 + np.exp(-numpy_result))

    return {
        "manual_calculation": {
            "mx_plus_b": manual_result,
            "sigmoid": manual_sigmoid,
        },
        "numpy_calculation": {
            "mx_plus_b": numpy_result.tolist(),
            "sigmoid": numpy_sigmoid.tolist(),
        }
    }

@app.get("/")
async def root():
    """Root endpoint with usage instructions."""
    return {
        "message": "Matrix Calculator API",
        "usage": {
            "endpoint": "/calculate",
            "method": "POST",
            "input_format": {
                "vector_x": "5x1 vector"
            }
        }
    }

if __name__ == "__main__":
    uvicorn.run(app)





