import numpy as np

def sigmoid(z):
	s = 1/(1+np.exp(-z))
	return s

def sigmoid_derivative(z):
	return z*(1-z)

inputs = np.array([[1,2,1],[0,0,1],[2,1,2],[1,2,1],[0,1,2],[2,2,2],[1,0,1]])
expected_output = np.array([[0],[0],[1],[1],[0],[1],[0]])

epochs = 10000
lr = 0.07
numInp, numOut = 3,1


W1 = np.ones((numInp,numOut))
b1 =np.ones((1,numOut))

print("W1",W1)
print("b1",b1)

for i in range(epochs):

	hidden_layer_activation = np.dot(inputs,W1)
	hidden_layer_activation += b1
	predicted_output = sigmoid(hidden_layer_activation)
	if i==0 or i==1: print(predicted_output) # A1
	#Backpropagation
	error = expected_output - predicted_output
	d_predicted_output = error * sigmoid_derivative(predicted_output) #dA2
	if i==0 or i==1: print(d_predicted_output)


	#Updating Weights and Biases
	W1 += inputs.T.dot(d_predicted_output)*lr
	b1 += np.sum(d_predicted_output,axis=0,keepdims=True)*lr
	if i==0 or i==1:print("W1",W1)
	if i==0 or i==1:print("b1",b1)


print("XNOR GATE")
print("INPUTS")
print(inputs)
print("Expected Output")
print(expected_output)
print("Predicted Output")
print(predicted_output)
print(W1)
print(b1)