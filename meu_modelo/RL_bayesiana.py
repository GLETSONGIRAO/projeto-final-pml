#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


class RegLinBayesiana():
    """
    w ~ N(w|0, alpha^(-1)I)
    y = X @ w
    t ~ N(t|X @ w, beta^(-1)
    
    """
    
    def __init__(self, alpha:float=1., beta:float=1.):
        self.alpha = alpha
        self.beta = beta
        self.w_mean = None
        self.w_precision = None
        
        
    def fit(self, X_train:np.ndarray, y:np.ndarray):
        """
        Atualização dos parametros para ajustar ao conjunto de dados
        
        Parametros
        -----------
        
        X: (N, n_atributos) 
        Dados de treinamento
        
        y: (N)
        Variável alvo
        
        """
        X_train = np.c_[X_train, np.ones(len(X_train))]
        ndim = X_train.shape[1]
        mean_prev = np.zeros(ndim)
        precision_prev = np.eye(ndim)
        
        w_precision = precision_prev + self.beta * X_train.T @ X_train
        w_mean = np.linalg.solve(w_precision,precision_prev @ mean_prev + self.beta * X_train.T @ y)
        self.w_mean = w_mean
        self.w_precision = w_precision
        self.w_cov = np.linalg.inv(self.w_precision)
        
    def predict(self, X_test:np.ndarray):
        """
        Retorna média e desvio padrão da distribuição preditiva
            
        Paramêtro
        ----------
        X_test: (N_test, n_atributos) 
        Dados de teste
            
        Retorno
        ----------
        y_test : (N_test) np.ndarray
        Média predita
            
        y_test_dp : (N_test) np.ndarray
        Desvio padrão predito
                        
        """
        
        X_test = np.c_[X_test, np.ones(len(X_test))]
        
        y = X_test @ self.w_mean
            
        y_var = 1 / self.beta + np.sum(X_test @ self.w_cov * X_test, axis=1)
        y_test_dp = np.sqrt(y_var)
            
        return y, y_test_dp
            
            
            
            
        
        
        
        


# In[ ]:




