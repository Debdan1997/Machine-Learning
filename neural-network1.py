#neural networks implementation 

import numpy as np 
import pickle

import csv
pic_inx = open('patternx.pickle','rb')#unpacking x and y values from training sets#
pic_iny = open('patterny.pickle','rb')

x_val = np.array(pickle.load(pic_inx)).astype(int)
y_val = pickle.load(pic_iny)

ones = np.ones((42000,1))

y_val = np.array(y_val).astype(int).reshape(42000,1)
x_val = np.concatenate((ones,x_val),axis = 1)

inodes = 785
hnodes = 385
onodes = 10

x_val = ((x_val/255.0)*0.99)+0.01 #normalization

def sigmoid(z):
	return 1/(1+np.exp(-z))

theta1 = np.random.normal(0.0,pow(hnodes,-0.5),(hnodes,inodes)) #creating weight matrices#
theta2 = np.random.normal(0.0,pow(onodes,-0.5),(onodes,hnodes))


def forward(theta1,theta2,input_x): #forward propagation of outputs
	a2 = sigmoid(np.dot(theta1,input_x))
	a3 = sigmoid(np.dot(theta2,a2))
	return a2,a3

def back(final,a2,a3): #back propagation of errors
	delta3 = a3-final
	delta2 = np.dot(np.transpose(theta2),delta3)*(a2*(1-a2))
	return delta2,delta3

def neural(theta1,theta2,input_x,final): #constructing neural nets
	a2,a3 = forward(theta1,theta2,input_x)
	efinal = final - a3
	ehidden = np.dot(np.transpose(theta2),efinal)
	
	change2 = np.dot((efinal*a3*(1.0-a3)),(np.transpose(a2)))
	change1 = np.dot((ehidden*a2*(1.0-a2)),(np.transpose(input_x)))

	theta1 = theta1+0.2*change1
	theta2 - theta2+0.2*change2
	return theta1,theta2

for k in range(5): #training
	for i in range(42000):
		final = np.zeros(10,)+0.01
		final[y_val[i]]=0.99
		final = np.array(final).reshape(onodes,1)

		x = x_val[i].reshape(inodes,1)

		theta1,theta2 = neural(theta1,theta2,x,final)

test = open('test.pickle','rb')
new_x = pickle.load(test) #getting test data
test.close()
num_test = len(new_x)
new_x = np.array(new_x).astype(float)
ones = np.ones((num_test,1))

new_x = np.concatenate((ones,new_x),axis = 1)
new_x = ((new_x/255.0)*0.99)+0.01

new_x = new_x.tolist()

predict = [] #pradictions

for i in new_x:
	x = np.array(i).reshape(inodes,1)
	a2,a3 = forward(theta1,theta2,x)
	
	a3 = a3.reshape(onodes,)
	predict.append(a3.tolist())

l = 1

f = open("foo.csv","w") #submission file
for i in predict:
	m = 0
	ind = 0
	for j in range(len(i)):
		if i[j]>m: #getting highest probability value
			m=i[j]
			ind = j
	#ans.append([l,ind])
	f.write(str(l)+","+str(ind)+"\n")
	l+=1

