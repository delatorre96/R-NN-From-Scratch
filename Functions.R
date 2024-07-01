
error.total <- function(p, phat){
    #' Calculate Total Error
    #'
    #' This function computes the total error between the predicted and actual values.
    #'
    #' @param p Numeric vector of actual values.
    #' @param phat Numeric vector of predicted values.
    #' @return Numeric value representing the total error.
    #' @examples
    #' error.total(c(1, 0, 1), c(0.8, 0.2, 0.9))
  e <- 0
  for (i in 1:length(p)){
    e <- e + ((1/2) * (p[i] - phat[i])^2)
  }
  return((1/length(p)) * e)
}


sigmoide <- function(z){
    #' Sigmoid Activation Function
    #'
    #' This function applies the sigmoid activation function.
    #'
    #' @param z Numeric vector or matrix of input values.
    #' @return Numeric vector or matrix with the sigmoid function applied.
    #' @examples
    #' sigmoide(c(1, 2, 3))
  1/(1 + exp(-z))
}


func_pesos_totales <- function(n_h, in_out_neur, min, max){
    #' Generate Random Weights
    #'
    #' This function generates a matrix of random weights.
    #'
    #' @param n_h Integer number of neurons in the hidden layer.
    #' @param in_out_neur Integer sum of input and output neurons.
    #' @param min Numeric minimum value for random weights.
    #' @param max Numeric maximum value for random weights.
    #' @return Matrix of random weights.
    #' @examples
    #' func_pesos_totales(5, 10, -1, 1)
  cant.pesos = in_out_neur * n_h
  return(matrix(runif(cant.pesos, min, max), ncol = 1))
}

func_reg_1 <- function(n_h, in_neur, pesos_totales, x){
    #' First Layer Regression
    #'
    #' This function computes the regression for the first hidden layer.
    #'
    #' @param n_h Integer number of neurons in the hidden layer.
    #' @param in_neur Integer number of input neurons.
    #' @param pesos_totales Matrix of total weights.
    #' @param x Numeric matrix of input data.
    #' @return Numeric matrix where each column represents the regression for each hidden neuron.
    #' @examples
    #' func_reg_1(5, 3, matrix(runif(15, -1, 1), ncol = 1), matrix(c(1, 2, 3, 4, 5, 6), nrow = 2))
  cant.pesos_1capa = n_h * in_neur 
  pesos_utilizar = pesos_totales[1:cant.pesos_1capa]
  vector = seq(1, cant.pesos_1capa, by = in_neur)
  matriz_neur_h = matrix(data = 0, ncol = n_h, nrow = nrow(x))
  
  for (i in 1:length(vector)){
    w = pesos_totales[vector[i]:min(vector[i] + 2, length(pesos_totales))] 
    w = matrix(w)
    regresion_neur_oculta = x %*% w
    matriz_neur_h[, i] = regresion_neur_oculta
  }
  
  return(matriz_neur_h)
}


func_reg_2 <- function(n_h, in_neur, pesos_totales, output_neur_h){
    #' Second Layer Regression
    #'
    #' This function computes the regression for the second hidden layer.
    #'
    #' @param n_h Integer number of neurons in the hidden layer.
    #' @param in_neur Integer number of input neurons.
    #' @param pesos_totales Matrix of total weights.
    #' @param output_neur_h Numeric matrix of outputs from the hidden layer.
    #' @return Numeric matrix representing the regression for the second layer.
    #' @examples
    #' func_reg_2(5, 3, matrix(runif(15, -1, 1), ncol = 1), matrix(c(1, 2, 3, 4, 5, 6), nrow = 2))
  cant.pesos_1capa = n_h * in_neur 
  w_h = matrix(pesos_totales[(cant.pesos_1capa + 1):length(pesos_totales)]) 
  return(output_neur_h %*% w_h)
}

##########################  For Backpropagation


der_error.total_sigmoide <- function(y, output){
    #' Derivative of Error with Respect to Sigmoid
    #'
    #' This function computes the derivative of the error with respect to the sigmoid function.
    #'
    #' @param y Numeric vector of actual values.
    #' @param output Numeric vector of predicted values.
    #' @return Numeric vector of derivatives.
    #' @examples
    #' der_error.total_sigmoide(c(1, 0, 1), c(0.8, 0.2, 0.9))
  der_e = c()
  for (i in 1:length(y)){
    der_e = c(der_e, (y[i] - output[i]))
  }
  return((der_e))
}


der_sigmoide <- function(input){
    #' Derivative of Sigmoid Function
    #'
    #' This function computes the derivative of the sigmoid function.
    #'
    #' @param input Numeric vector or matrix of input values.
    #' @return Numeric vector or matrix of derivatives.
    #' @examples
    #' der_sigmoide(c(1, 2, 3))
  return((sigmoide(input) * (1 - sigmoide(input))))
}


der_reg_peso <- function(peso){
    #' Derivative of Regression with Respect to Weight
    #'
    #' This function computes the derivative of the regression with respect to the weight.
    #'
    #' @param peso Numeric value representing the weight.
    #' @return Numeric value of the derivative.
    #' @examples
    #' der_reg_peso(0.5)
  return(peso)
}


red_neuronal_Sin_capa_oculta <- function(x, y, learning_rate, minimo, maximo){
    #' Neural Network Without Hidden Layer
    #'
    #' This function trains a neural network without a hidden layer.
    #'
    #' @param x Numeric matrix of input data.
    #' @param y Numeric vector of actual values.
    #' @param learning_rate Numeric learning rate for weight updates.
    #' @param minimo Numeric minimum value for random weights.
    #' @param maximo Numeric maximum value for random weights.
    #' @return List containing the output and the total error across epochs.
    #' @examples
    #' red_neuronal_Sin_capa_oculta(matrix(c(1, 2, 3, 4, 5, 6), nrow = 2), c(0, 1), 0.01, -1, 1)
  w <- runif(ncol(x), minimo, maximo)
  input <- x %*% w
  output <- sigmoide(input)
  error_total <- error.total(y, output)
  error_min = 0.005 
  error_total_epocas = c()
  
  while (error_total >= error_min){
    input <- x %*% w
    output <- sigmoide(input)
    error_total <- error.total(y, output)
    error_total_epocas <- c(error_total_epocas, error_total)
    
    # Backpropagation
    dE_dout <- der_error.total_sigmoide(y, output)
    dout_din <- der_sigmoide(input)
    
    # Weight updates
    desv_w0 <- dE_dout * dout_din * x[, 1] * learning_rate
    desv_w1 <- dE_dout * dout_din * x[, 2] * learning_rate
    desv_w2 <- dE_dout * dout_din * x[, 3] * learning_rate
    
    w[1] <- w[1] + mean(desv_w0)
    w[2] <- w[2] + mean(desv_w1)
    w[3] <- w[3] + mean(desv_w2)
  }
  
  plot(error_total_epocas, xlab = 'Épocas', ylab = 'Error', main = 'Error de predicción por época', type = 'l')
  cat('El error de la predicción en la última época es: ', error_total)
  cat('\nLa cantidad de épocas que han pasado han sido:', length(error_total_epocas))
  return(list('output' = output, 'error_total_epocas' = error_total_epocas))
}


red_neuronal_con_capa_oculta <- function(learning_rate, n_h, x, y, inicializacion, error_min = 0.001, epocas_por_defecto = 1000000){
    #' Neural Network with One Hidden Layer
    #'
    #' This function trains a neural network with one hidden layer.
    #'
    #' @param learning_rate Numeric learning rate for weight updates.
    #' @param n_h Integer number of neurons in the hidden layer.
    #' @param x Numeric matrix of input data.
    #' @param y Numeric vector of actual values.
    #' @param inicializacion Numeric vector of length 2 specifying the minimum and maximum values for random weights.
    #' @param error_min Numeric minimum error threshold to stop training. Default is 0.001.
    #' @param epocas_por_defecto Integer maximum number of epochs. Default is 1000000.
    #' @return List containing the final output, weight history, and error history.
    #' @examples
    #' red_neuronal_con_capa_oculta(0.01, 5, matrix(c(1, 2, 3, 4, 5, 6), nrow = 2), c(0, 1), c(-1, 1))
  in_neur = ncol(x)
  out_neur = 1
  pesos_totales = func_pesos_totales(n_h, in_neur + out_neur, inicializacion[1], inicializacion[2])
  input_1 = func_reg_1(n_h, in_neur, pesos_totales, x)
  output_1 = sigmoide(input_1)
  input_2 = func_reg_2(n_h, in_neur, pesos_totales, output_1)
  output_2 = sigmoide(input_2)
  
  error_total = error.total(y, output_2)
  error_total_epocas = c()
  epocas = 1
  
  while (epocas <= epocas_por_defecto){
    if (error_total < error_min){
      break
    }
    
    input_1 = func_reg_1(n_h, in_neur, pesos_totales, x)
    output_1 = sigmoide(input_1)
    input_2 = func_reg_2(n_h, in_neur, pesos_totales, output_1)
    output_2 = sigmoide(input_2)
    
    error_total = error.total(y, output_2)
    error_total_epocas = c(error_total_epocas, error_total)
    
    dE_dout_2 = der_error.total_sigmoide(y, output_2)
    dout_din_2 = der_sigmoide(input_2)
    
    w_h = matrix(pesos_totales[(n_h * in_neur + 1):length(pesos_totales)]) 
    desv_2cap_pesos = learning_rate * dE_dout_2 * dout_din_2 %*% t(output_1)
    
    dE_din_1 = dout_din_2 %*% t(dE_dout_2) %*% w_h
    din_1_dout_1 = der_sigmoide(input_1)
    dE_dout_1 = dE_din_1 * din_1_dout_1
    
    desv_1cap_pesos = matrix(0, nrow = n_h, ncol = in_neur)
    for (i in 1:n_h){
      desv_1cap_pesos[i, ] = learning_rate * dE_dout_1[i, ] %*% t(x)
    }
    
    desv_pesos_totales = c(as.vector(desv_1cap_pesos), as.vector(desv_2cap_pesos))
    pesos_totales = pesos_totales + desv_pesos_totales
    
    epocas = epocas + 1
  }
  
  plot(error_total_epocas, xlab = 'Épocas', ylab = 'Error', main = 'Error de predicción por época', type = 'l')
  cat('El error de la predicción en la última época es: ', error_total)
  cat('\nLa cantidad de épocas que han pasado han sido:', length(error_total_epocas))
  return(list('output' = output_2, 'historial_pesos' = pesos_totales, 'error_total_epocas' = error_total_epocas))
}
