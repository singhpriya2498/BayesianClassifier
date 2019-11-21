from __future__ import print_function
import statistics_func as sf
import math
import matplotlib.pyplot as plt
import contour
import numpy as np
class Model():
	Matrix=[[0,0],[0,0]]
	mew=[]
	des=[[0,0,0],[0,0,0],[0,0,0]]
	g_x=[[0,0,0],[0,0,0],[0,0,0]]
	DATA=[]
	RANGE=[]
	def __init__(self,DATASET,RANGE):
		self.DATA=DATASET
		self.RANGE=RANGE
		Class1_train_Matrix=sf.get_Matrix(DATASET[0])
		Class2_train_Matrix=sf.get_Matrix(DATASET[1])
		Class3_train_Matrix=sf.get_Matrix(DATASET[2])
		for i in range(len(DATASET)):
			self.mew.append(sf.Mean(DATASET[i]))
		for i in range(2):
			for j in range(2):
				self.Matrix[i][j]=Class1_train_Matrix[i][j]+Class2_train_Matrix[i][j]+Class3_train_Matrix[i][j]
		for i in range(2):
			for j in range(2):
				self.Matrix[i][j]=self.Matrix[i][j]/3
	def print_Matrix(self):
		for i in range(2):
			for j in range(2):
				print (self.Matrix[i][j], end= ' ')
			print ("")
	def get_lines(self):
		Inv=np.linalg.inv(self.Matrix)	
		X=[0,0]
		for i in range(3):
			for j in range(2):
				X[j]=(self.mew[i][j])
			self.g_x[i][0]=(X[0]*Inv[0][0]+X[1]*Inv[1][0])
			self.g_x[i][1]=(X[0]*Inv[0][1]+X[1]*Inv[1][1])
			self.g_x[i][2]=math.log(len(self.DATA[i]))-0.5*((self.mew[i][0]*(self.mew[i][0]*Inv[0][0]+self.mew[i][1]*Inv[0][1])+self.mew[i][1]*(self.mew[i][0]*Inv[0][1]+self.mew[i][1]*Inv[1][1])))
	def plot_model(self,val):
		sf.plot_gx(self.g_x,self.RANGE,val)
		sf.plot(self.DATA[0],'mo',"",True,self.mew[0],self.Matrix)
		sf.plot(self.DATA[1],'yo',"",True,self.mew[1],self.Matrix)
		sf.plot(self.DATA[2],'co',"",True,self.mew[2],self.Matrix)
		sf.plot([sf.Mean(self.DATA[0])],'ko')
		sf.plot([sf.Mean(self.DATA[1])],'ko')
		sf.plot([sf.Mean(self.DATA[2])],'ko')
		plt.legend()
		plt.show()
	def plot_classes(self,val):
		i=self.RANGE[0][0]
		temp=[[],[]]
		while i<=self.RANGE[0][1]:
			j=self.RANGE[1][0]
			while j<=self.RANGE[1][1]:
				Max=-100000000000.0
				index=-1
				for k in range(2):
					if(Max<((self.g_x[k][0]*i)+(self.g_x[k][1]*j)+self.g_x[k][2])):
						Max=(self.g_x[k][0]*i)+(self.g_x[k][1]*j)+self.g_x[k][2]
						index=k
				temp[index].append([i,j])
				j=j+val
			i=i+val
		sf.plot(temp[0],'r',"Class1")
		sf.plot(temp[1],'b',"Class2")
		sf.plot(self.DATA[0],'go',"",True,self.mew[0],self.Matrix)
		sf.plot(self.DATA[1],'yo',"",True,self.mew[1],self.Matrix)
		plt.show()
		i=self.RANGE[0][0]
		temp=[[],[]]
		while i<=self.RANGE[0][1]:
			j=self.RANGE[1][0]
			while j<=self.RANGE[1][1]:
				Max=-100000000000.0
				index=-1
				if(((self.g_x[1][0]*i)+(self.g_x[1][1]*j)+(self.g_x[1][2]))>((self.g_x[2][0]*i)+(self.g_x[2][1]*j)+(self.g_x[2][2]))):
					temp[0].append([i,j])
				else:
					temp[1].append([i,j])
				# temp[index].append([i,j])
				j=j+val
			i=i+val
		sf.plot(temp[0],'r',"Class1")
		sf.plot(temp[1],'b',"Class2")
		sf.plot(self.DATA[1],'go',"",True,self.mew[1],self.Matrix)
		sf.plot(self.DATA[2],'yo',"",True,self.mew[2],self.Matrix)
		plt.show()
		i=self.RANGE[0][0]
		temp=[[],[]]
		while i<=self.RANGE[0][1]:
			j=self.RANGE[1][0]
			while j<=self.RANGE[1][1]:
				Max=-100000000000.0
				index=-1
				if(((self.g_x[0][0]*i)+(self.g_x[0][1]*j)+(self.g_x[0][2]))<((self.g_x[2][0]*i)+(self.g_x[2][1]*j)+(self.g_x[2][2]))):
					temp[0].append([i,j])
				else:
					temp[1].append([i,j])
				j=j+val
			i=i+val	
		sf.plot(temp[0],'r',"Class1")
		sf.plot(temp[1],'b',"Class2")
		sf.plot(self.DATA[0],'go',"",True,self.mew[0],self.Matrix)
		sf.plot(self.DATA[2],'yo',"",True,self.mew[2],self.Matrix)
		plt.show()
	def get_ConfMatrix_pair(self, TESTSET):
		CONF=[[0,0],[0,0]]
		arr=[0,1]
		for i in arr:
			for j in range(len(TESTSET[i])):
				Max=-100000000000.0
				index=-1
				for k in arr:
					if(Max<((self.g_x[k][0]*TESTSET[i][j][0])+(self.g_x[k][1]*TESTSET[i][j][1])+self.g_x[k][2])):
						Max=(self.g_x[k][0]*TESTSET[i][j][0])+(self.g_x[k][1]*TESTSET[i][j][1])+self.g_x[k][2]
						index=k
				CONF[i][index]=CONF[i][index]+1
		for i in range(2):
			for j in range(2):
				print(CONF[i][j], end=" ")
			print("")
		print("For class1 and 2")
		sf.get_Score(CONF)
		CONF=[[0,0],[0,0]]
		arr=[1,2]
		for i in range(2):
			for j in range(len(TESTSET[arr[i]])):
				Max=-100000000000.0
				index=-1
				for k in range(2):
					if(Max<((self.g_x[arr[k]][0]*TESTSET[arr[i]][j][0])+(self.g_x[arr[k]][1]*TESTSET[arr[i]][j][1])+self.g_x[arr[k]][2])):
						Max=((self.g_x[arr[k]][0]*TESTSET[arr[i]][j][0])+(self.g_x[arr[k]][1]*TESTSET[arr[i]][j][1])+self.g_x[arr[k]][2])
						index=k
				CONF[i][index]=CONF[i][index]+1
		for i in range(2):
			for j in range(2):
				print(CONF[i][j], end=" ")
			print("")
		print("For class2 and 3")
		sf.get_Score(CONF)		
		CONF=[[0,0],[0,0]]
		arr=[0,2]
		for i in range(2):
			for j in range(len(TESTSET[arr[i]])):
				Max=-100000000000.0
				index=-1
				for k in range(2):
					if(Max<((self.g_x[arr[k]][0]*TESTSET[arr[i]][j][0])+(self.g_x[arr[k]][1]*TESTSET[arr[i]][j][1])+self.g_x[arr[k]][2])):
						Max=((self.g_x[arr[k]][0]*TESTSET[arr[i]][j][0])+(self.g_x[arr[k]][1]*TESTSET[arr[i]][j][1])+self.g_x[arr[k]][2])
						index=k
				CONF[i][index]=CONF[i][index]+1
		for i in range(2):
			for j in range(2):
				print(CONF[i][j], end=" ")
			print("")
		print("For class1 and 3")
		sf.get_Score(CONF)
	def get_ConfMatrix(self,TESTSET):
		CONF=[[0,0,0],[0,0,0],[0,0,0]]
		for i in range(len(TESTSET)):
			for j in range(len(TESTSET[i])):
				Max=-100000000000.0
				index=-1
				for k in range(3):
					if(Max<((self.g_x[k][0]*TESTSET[i][j][0])+(self.g_x[k][1]*TESTSET[i][j][1])+self.g_x[k][2])):
						Max=(self.g_x[k][0]*TESTSET[i][j][0])+(self.g_x[k][1]*TESTSET[i][j][1])+self.g_x[k][2]
						index=k
				CONF[i][index]=CONF[i][index]+1
		for i in range(3):
			for j in range(3):
				print(CONF[i][j], end=" ")
			print("")
		sf.get_Score(CONF)
		