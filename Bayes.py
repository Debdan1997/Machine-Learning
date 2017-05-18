import numpy as np 
import pandas as pd 
import math
import csv
import string
from collections import Counter,defaultdict
import random
from math import log

label = []
message = []
sms = []

file = r'spam.csv'
with open(file,'r') as file:
	for row in csv.reader(file,delimiter = ','):
		sms.append(row)
	""" shuffling the set """
	random.shuffle(sms)
	for row in sms:
		label.append(row[0])
		text = row[1].translate(str.maketrans('','',string.punctuation))
		message.append([word.lower() for word in text.split()])

train_test_ratio = (7,3) """train: test split of given set""" 

train_test_boundary = int(len(sms)*(train_test_ratio[0])/(train_test_ratio[0]+train_test_ratio[1]))

label_train = label[:train_test_boundary]
message_train = message[:train_test_boundary]

label_test = label[train_test_boundary:]
message_test = message[train_test_boundary:]

def calculate_prior(label_train,message_train):
	prior = Counter()
	likelihood = defaultdict(Counter)

	for i in range(len(message_train)):
		prior[label_train[i]]+=1
		for word in message_train[i]:
			likelihood[label_train[i]][word]+=1
	return prior,likelihood

prior,likelihood = calculate_prior(label_train,message_train)

def naive_bayes(message,prior,likelihood):
	m_class = (-1E6,'')
	for c in prior.keys():
		p = (prior[c])
		n = sum(likelihood[c].values())
		for word in message:
			p=p*(max(1E-9,likelihood[c][word])/n)
		if p>m_class[0]:
			m_class=(p,c)
	return m_class[1]


def classify(label_train,message_train,label_test,message_test):
	prior,likelihood = calculate_prior(label_train,message_train)

	total = len(message_test)
	count = 0
	for i in range(len(message_test)):
		if naive_bayes(message_test[i],prior,likelihood) == label_test[i]:
			count+=1
	
	print('accuracy is '+str(count/(total*1.0)))

classify(label_train,message_train,label_test,message_test)




