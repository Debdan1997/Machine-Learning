import numpy as np 
import pickle
import csv

x = []
y = []
file = r'train.csv'


with open(file,'r') as f:
	data = csv.reader(f,delimiter = ',')
	s=1;
	for row in data:
		k = []
		for i in row[1:]:
			k.append(i)
		x.append(k)
		y.append(row[0])
		s+=1

f = open('patternx.pickle','wb') #dumping x_values for training 
pickle.dump(x,f)
f.close()
f = open('patterny.pickle','wb') #dumping y_values for training
pickle.dump(y,f) 
f.close()

x = []
file = r'test.csv'
with open(file,'r') as f:
	data = csv.reader(f,delimiter = ',')
	s=1;
	for row in data:
		k = []
		for i in row:
			k.append(i)
		x.append(k)
		s+=1

test = open('test.pickle','wb')
pickle.dump(x,test) #dumping x_values for testing
test.close()
