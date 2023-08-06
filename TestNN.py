import numpy as np
class NN():
    def __init__(self, dim):
        self.layers = len(dim)
        self.dimensions = dim
        self.weight = [np.random.randn(i,j) for i,j in [(x,y) for x,y in zip(dim[1:], dim[:-1])]]
        self.bias = [np.random.randn(y, 1) for y in dim[1:]]
        
    def sigmoid(self,z):
        """The sigmoid function."""
        return 1.0/(1.0+np.exp(-z))
    
    def sigmoid_prime(self,z):
        """Derivative of the sigmoid function."""
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
    def feedforward(self,X):
        A = X
        for w, b in zip(self.weight, self.bias):
            A = self.sigmoid((np.dot(w,A)+b))            
        return A
    
    def backprop(self, X,y):
        #print("::::::: Backprop :::::::\n")
        #FeedForward
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        nabla_w = [np.zeros(w.shape) for w in self.weight]
        activation = X
        activations = [X]
        zs = []
        for b, w in zip(self.bias, self.weight):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
            
        # backward pass
        # Delta for Output Layer
        #delta = (activations[-1] - y) * sigmoid_prime(zs[-1]) # For mean squared error
        delta = (activations[-1] - y) # For Cross Entropy Error (Logistic Error)
        #nabla_b[-1] = delta
        nabla_b[-1] = (1/X.shape[1]) * np.sum(delta, axis=1, keepdims=True) #(1,1)        
        #nabla_w[-1] =  np.dot(delta, activations[-2].transpose())        
        nabla_w[-1] =  (1/X.shape[1]) * np.dot(delta, activations[-2].transpose())        
        #Delta for Hidden Layers
        for l in range(2, self.layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)            
            delta = np.dot(self.weight[-l+1].transpose(), delta) * sp           
            nabla_b[-l] = (1/X.shape[1]) * np.sum(delta, axis=1, keepdims=True)
            nabla_w[-l] = (1/X.shape[1]) * np.dot(delta, activations[-l-1].transpose())            
        #print("::::::: End Backprop :::::::\n")
        return (nabla_b, nabla_w)
    
    def update_weights(self,X, y,eta):
        #print("::::::: Update Weights :::::::\n")
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        nabla_w = [np.zeros(w.shape) for w in self.weight]
        
        delta_nabla_b, delta_nabla_w = self.backprop(X, y)              
        nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        self.weight = [w-(eta/1.0)*nw for w, nw in zip(self.weight, nabla_w)]   
        self.bias = [b-(eta/1.0)*nb for b, nb in zip(self.bias, nabla_b)]
        #print("::::::: End Update Weights :::::::\n")    
        
    def train(self, X, y, epochs=10, eta=0.01):
        for i in range(epochs):
            self.update_weights(X,y,eta)
        print("Epochs Complete: ",epochs)
            
    def predict(self, ip):
        return feedforward(ip)
    
    def display(self):
        print("Weights")
        for x in self.weight:
            print(x,x.shape,"\n")
        print("\nBias")
        for y in self.bias:
            print(y, y.shape,"\n")
            
    def display_2(self):
        for b,w in zip(self.bias, self.weight):
            print("Bias: ",b,b.shape,"\n", "Weight: ", w, w.shape)
