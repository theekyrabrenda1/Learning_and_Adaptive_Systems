#Python 
import numpy as np 

class MLP:    
    def __init__(self, input_size, hidden_size, output_size): 
        # Systematic initialization of weights and biases        
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01        
        self.b1 = np.zeros((1, hidden_size))        
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01        
        self.b2 = np.zeros((1, output_size))    
        
    def sigmoid(self, z):        
        return 1 / (1 + np.exp(-z))    
        
    def forward(self, X):        
        # Input to Hidden        
        self.z1 = np.dot(X, self.W1) + self.b1        
        self.a1 = self.sigmoid(self.z1)        
        # Hidden to Output        
        self.z2 = np.dot(self.a1, self.W2) + self.b2        
        self.a2 = self.sigmoid(self.z2)        
        return self.a2    
        
    def train(self, X, y, learning_rate=0.1):        
        # Forward pass        
        output = self.forward(X)  

        # Backward pass (Backpropagation of errors)        
        m = y.shape[0]        
        dz2 = output - y        
        dW2 = (1/m) * np.dot(self.a1.T, dz2)        
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)        
        
        dz1 = np.dot(dz2, self.W2.T) * (self.a1 * (1 - self.a1))        
        dW1 = (1/m) * np.dot(X.T, dz1)        
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)        
        
        # Systematic parameter update        
        self.W2 -= learning_rate * dW2        
        self.b2 -= learning_rate * db2        
        self.W1 -= learning_rate * dW1        
        self.b1 -= learning_rate * db1