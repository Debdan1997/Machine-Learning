# digit recognition using logistic regression 
from PIL import Image
import csv
import numpy as np 
import pickle
import io
from scipy import optimize

pic_inx = open('patternx.pickle','rb')
pic_iny = open('patterny.pickle','rb')

x_val = pickle.load(pic_inx) #getting X values as features (each pixel)
y_val = pickle.load(pic_iny) #getting Y values
x_val = np.array(x_val).astype(float)

num_obs = len(x_val)
num_features = len(x_val[0])+1

x_val = (x_val-np.mean(x_val))/np.std(x_val) #mean normalization


y_val = np.array(y_val).astype(int).reshape(num_obs,1)

def sigmoid(x_val):
	return 1/(1+np.exp(-x_val))

theta = np.zeros((10,num_features))
ones = np.ones((num_obs,1))
x_val = np.concatenate((ones,x_val),axis = 1) # making first column of ones

x = np.transpose(x_val)
x.reshape((num_features,num_obs))

s = []

def gradient_des(theta,x,x_val,y_val,it = 500,alpha = 0.001):
	y_val = y_val.reshape((num_obs,1))

	for i in range(it): #500 iterations (default) 

		si = np.transpose(theta)
		hx = sigmoid(np.dot(x_val,si))
		k = theta.tolist()

		theta_0 = np.array(k[0][0]).reshape(1,1)
		theta_oth = np.array(k[0][1:]).reshape(1,num_features-1)

		#descending the first feature without regularization
		theta_0 = theta_0 - (1/num_obs)*alpha*(np.dot(np.transpose(hx-y_val),x_val[:,0]))

		#descending rest of the features with regularization
		x = np.delete(x_val,0,axis = 1)
		theta_oth = theta_oth - (1/num_obs)*alpha*((np.dot(np.transpose(hx-y_val),x))+1000*(theta_oth))
		k = theta_0.reshape(1,).tolist()+theta_oth.reshape(num_features-1,).tolist()
		theta = np.array(k)
		theta = theta.reshape((1,num_features))

	return theta

y_val_t = np.array(y_val)
for i in range(10):
	y_val_tt = (y_val_t==i).astype(int).reshape(num_obs,1)
	theta[i] = gradient_des(theta[i].reshape((1,num_features)),x,x_val,y_val_tt)


#theta_out = open('theta_out.pickle','wb')
#pickle.dump(theta,theta_out)
#theta_out.close()

test = open('test.pickle','rb')
new_x = pickle.load(test) #test features
test.close()
num_test = len(new_x)
new_x = np.array(new_x).astype(float)
ones = np.ones((num_test,1))

new_x = np.concatenate((ones,new_x),axis = 1) #making first column of ones


new_x = (new_x-np.mean(new_x))/np.std(new_x) #mean normalization

predict = np.dot(new_x,np.transpose(theta))


predict = sigmoid(predict)
ans = []
l = 1
f = open("foo.csv","w") #submission file
for i in predict.tolist():
	m = 0
	ind = 0
	for j in range(len(i)):
		if i[j]>m: #getting highest probability value
			m=i[j]
			ind = j
	ans.append([l,ind])
	f.write(str(l)+","+str(ind)+"\n")
	l+=1
