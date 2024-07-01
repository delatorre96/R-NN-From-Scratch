# Neural Network R Script

This R script implements a simple neural network with and without hidden layers. It includes functions for forward propagation, error calculation, and backpropagation to update the weights. The script is designed for educational purposes and provides a basic understanding of how neural networks work.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Functions](#functions)
3. [Usage](#usage)


## Getting Started

### Prerequisites

- R (version 3.6 or higher)

### Installing

1. Install R from [CRAN](https://cran.r-project.org/).
2. Optionally, use RStudio for an enhanced coding experience.

### Cloning the Repository

```bash
git clone <repository_url>

## Functions

### Error Calculation

- **error.total(p, phat)**: Computes the total error between the predicted and actual values.
  - **Parameters**:
    - `p`: Numeric vector of actual values.
    - `phat`: Numeric vector of predicted values.
  - **Returns**: Numeric value representing the total error.

### Activation Function

- **sigmoide(z)**: Applies the sigmoid activation function.
  - **Parameters**: 
    - `z`: Numeric vector or matrix of input values.
  - **Returns**: Numeric vector or matrix with the sigmoid function applied.

### Weight Initialization

- **func_pesos_totales(n_h, in_out_neur, min, max)**: Generates a matrix of random weights.
  - **Parameters**:
    - `n_h`: Integer number of neurons in the hidden layer.
    - `in_out_neur`: Integer sum of input and output neurons.
    - `min`: Numeric minimum value for random weights.
    - `max`: Numeric maximum value for random weights.
  - **Returns**: Matrix of random weights.

### Forward Propagation

- **func_reg_1(n_h, in_neur, pesos_totales, x)**: Computes the regression for the first hidden layer.
  - **Parameters**:
    - `n_h`: Integer number of neurons in the hidden layer.
    - `in_neur`: Integer number of input neurons.
    - `pesos_totales`: Matrix of total weights.
    - `x`: Numeric matrix of input data.
  - **Returns**: Numeric matrix where each column represents the regression for each hidden neuron.

- **func_reg_2(n_h, in_neur, pesos_totales, output_neur_h)**: Computes the regression for the second hidden layer.
  - **Parameters**:
    - `n_h`: Integer number of neurons in the hidden layer.
    - `in_neur`: Integer number of input neurons.
    - `pesos_totales`: Matrix of total weights.
    - `output_neur_h`: Numeric matrix of outputs from the hidden layer.
  - **Returns**: Numeric matrix representing the regression for the second layer.

### Backpropagation

- **der_error.total_sigmoide(y, output)**: Computes the derivative of the error with respect to the sigmoid function.
  - **Parameters**:
    - `y`: Numeric vector of actual values.
    - `output`: Numeric vector of predicted values.
  - **Returns**: Numeric vector of derivatives.

- **der_sigmoide(input)**: Computes the derivative of the sigmoid function.
  - **Parameters**:
    - `input`: Numeric vector or matrix of input values.
  - **Returns**: Numeric vector or matrix of derivatives.

- **der_reg_peso(peso)**: Computes the derivative of the regression with respect to the weight.
  - **Parameters**:
    - `peso`: Numeric value representing the weight.
  - **Returns**: Numeric value of the derivative.

### Neural Network Models

- **red_neuronal_Sin_capa_oculta(x, y, learning_rate, minimo, maximo)**: Trains a neural network without a hidden layer.
  - **Parameters**:
    - `x`: Numeric matrix of input data.
    - `y`: Numeric vector of actual values.
    - `learning_rate`: Numeric learning rate for weight updates.
    - `minimo`: Numeric minimum value for random weights.
    - `maximo`: Numeric maximum value for random weights.
  - **Returns**: List containing the output and the total error across epochs.

- **red_neuronal_con_capa_oculta(learning_rate, n_h, x, y, inicializacion, error_min = 0.001, epocas_por_defecto = 1000000)**: Trains a neural network with one hidden layer.
  - **Parameters**:
    - `learning_rate`: Numeric learning rate for weight updates.
    - `n_h`: Integer number of neurons in the hidden layer.
    - `x`: Numeric matrix of input data.
    - `y`: Numeric vector of actual values.
    - `inicializacion`: Numeric vector of length 2 specifying the minimum and maximum values for random weights.
    - `error_min`: Numeric minimum error threshold to stop training. Default is 0.001.
    - `epocas_por_defecto`: Integer maximum number of epochs. Default is 1000000.
  - **Returns**: List containing the final output, weight history, and error history.

## Usage

1. **Source the script in your R environment.**
   
```r
source('path_to_script.R')
# Generate sample data
x <- matrix(runif(30, 0, 1), ncol = 3)
y <- runif(10, 0, 1)

# Train neural network without hidden layer
result <- red_neuronal_Sin_capa_oculta(x, y, 0.01, -1, 1)
print(result$output)

# Train neural network with one hidden layer
result <- red_neuronal_con_capa_oculta(0.01, 5, x, y, c(-1, 1))
print(result$output)
```
### Notes
- Adjust the learning_rate, n_h, inicializacion, and other parameters as needed for your specific dataset.
- Monitor the error plot to ensure the network is learning correctly.
- For large datasets or more complex models, consider using specialized deep learning frameworks like TensorFlow or PyTorch.
- Have a look the jupyter notebook called AnalisisRedesNeuronales.ipynb for an example
