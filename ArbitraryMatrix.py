from __future__ import print_function
import statistics_func as sf
import math
import numpy as np
class Model():
	Class1_train_Matrix=[[0,0],[0,0]]
	Class2_train_Matrix=[[0,0],[0,0]]
	Class3_train_Matrix=[[0,0],[0,0]]
	Class1_test_Matrix=[[0,0],[0,0]]
	Class2_test_Matrix=[[0,0],[0,0]]
	Class3_test_Matrix=[[0,0],[0,0]]
	mew=[]
	des=[[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
	DATA=[]
	class1 ,class2, class3= [],[],[]
	def __init__(self,DATASET):
		self.DATA=DATASET
		self.Class1_train_Matrix=sf.get_Matrix(DATASET[0])
		self.Class2_train_Matrix=sf.get_Matrix(DATASET[1])
		self.Class3_train_Matrix=sf.get_Matrix(DATASET[2])
		for i in range(len(DATASET)):
			self.mew.append(sf.Mean(DATASET[i]))
	def getGx(self,x,y,inv, mean,matrix,ci):
		term1 = sf.get_Product([x,y],inv)
		term2 = sf.get_Product(mean, inv)
		term3 = np.matmul(mean,inv)
		term3 = -2 * np.matmul(term3, [x,y])
		# term4 = math.log(np.linalg.det(np.array(matrix)))
		term4 = math.log(matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0])
		term5 = math.log((float(ci)))
		return 0.5 * (term1 + term2 + term3 + term4) + term5

	def get_lines(self, data_id):
		step = 1
		left_margin, right_margin, top_margin, bottom_margin = 0,0,0,0
		if (data_id == 1):
			step = 0.2
			left_margin, right_margin, top_margin, bottom_margin = -10,25,-15,20
		elif( data_id == 2):
			step = 0.05
			left_margin, right_margin, top_margin, bottom_margin = -3,3,-3,3
		elif( data_id == 3):
			step = 20
			left_margin, right_margin, top_margin, bottom_margin = -500,2100,0,3000

		inv_class1=sf.get_Inverse(self.Class1_train_Matrix)
		inv_class2=sf.get_Inverse(self.Class2_train_Matrix)
		inv_class3=sf.get_Inverse(self.Class3_train_Matrix)
		
		i = left_margin
		while(i<right_margin+1):
			j = top_margin
			while(j<bottom_margin+1):
				val1 = self.getGx(i,j,inv_class1,self.mew[0], self.Class1_train_Matrix,len(self.DATA[0]))
				val2 = self.getGx(i,j,inv_class2,self.mew[1], self.Class2_train_Matrix,len(self.DATA[1]))
				val3 = self.getGx(i,j,inv_class3,self.mew[2], self.Class3_train_Matrix,len(self.DATA[2]))
				if(max(val1,val2,val3)==val1): self.class1.append([i,j])
				elif(max(val1,val2,val3)==val2): self.class2.append([i,j])
				else: self.class3.append([i,j])
				j+=step
			i+=step
		return

	def get_pair(self, data_id, DATASET):
		step = 1
		left_margin, right_margin, top_margin, bottom_margin = 0,0,0,0
		if (data_id == 1):
			step = 0.2
			left_margin, right_margin, top_margin, bottom_margin = -10,25,-15,20
		elif( data_id == 2):
			step = 0.05
			left_margin, right_margin, top_margin, bottom_margin = -3,3,-3,3
		elif( data_id == 3):
			step = 20
			left_margin, right_margin, top_margin, bottom_margin = -500,2100,0,3000

		self.Class1_train_Matrix=sf.get_Matrix(DATASET[0])
		self.Class2_train_Matrix=sf.get_Matrix(DATASET[1])
		self.Class3_train_Matrix=sf.get_Matrix(DATASET[2])


		inv_class1=sf.get_Inverse(self.Class1_train_Matrix)
		inv_class2=sf.get_Inverse(self.Class2_train_Matrix)
		inv_class3=sf.get_Inverse(self.Class3_train_Matrix)

		temp1=sf.get_Matrix(DATASET[0])
		temp2=sf.get_Matrix(DATASET[1])
		temp3=sf.get_Matrix(DATASET[2])
		
		classA,classB = [],[]
		i = left_margin
		while(i<right_margin+1):
			j = top_margin
			while(j<bottom_margin+1):
				val1 = self.getGx(i,j,inv_class1,self.mew[0], self.Class1_train_Matrix,len(self.DATA[0]))
				val2 = self.getGx(i,j,inv_class2,self.mew[1], self.Class2_train_Matrix,len(self.DATA[1]))
				if(val1>val2): classA.append([i,j])
				else: classB.append([i,j])
				j+=step
			i+=step
		# sf.plot_fourth_pair(classA, classB,1,2,self.DATA,self.Class1_train_Matrix,self.Class2_train_Matrix,self.mew)
		sf.plot_fourth_pair(classA, classB,1,2,self.DATA,temp1,temp2,self.mew)
		
		classB,classC = [],[]
		i = left_margin
		while(i<right_margin+1):
			j = top_margin
			while(j<bottom_margin+1):
				val2 = self.getGx(i,j,inv_class2,self.mew[1], self.Class2_train_Matrix,len(self.DATA[1]))
				val3 = self.getGx(i,j,inv_class3,self.mew[2], self.Class3_train_Matrix,len(self.DATA[2]))
				if(val3>val2): classC.append([i,j])
				else: classB.append([i,j])

				j+=step
			i+=step
		sf.plot_fourth_pair(classB, classC,2,3,self.DATA,temp2,temp3,self.mew)
		
		classA,classC = [],[]
		i = left_margin
		while(i<right_margin+1):
			j = top_margin
			while(j<bottom_margin+1):
				val1 = self.getGx(i,j,inv_class1,self.mew[0], self.Class1_train_Matrix,len(self.DATA[0]))
				val3 = self.getGx(i,j,inv_class3,self.mew[2], self.Class3_train_Matrix,len(self.DATA[2]))
				if(val3>val1): classC.append([i,j])
				else: classA.append([i,j])

				j+=step
			i+=step
		sf.plot_fourth_pair(classA, classC,1,3,self.DATA,temp1,temp3,self.mew)
		return

	def plot_model(self):
		self.Class1_train_Matrix=sf.get_Matrix(self.DATA[0])
		self.Class2_train_Matrix=sf.get_Matrix(self.DATA[1])
		self.Class3_train_Matrix=sf.get_Matrix(self.DATA[2])
		sf.plot_fourth(self.class1,self.class2,self.class3,self.DATA,self.Class1_train_Matrix,self.Class2_train_Matrix,self.Class3_train_Matrix,self.mew)
		# sf.plot_quadritic(self.des)

	def get_ConfMatrix_pair(self, TESTSET):
		CONF=[[0,0],[0,0]]
		temp1=sf.get_Matrix(TESTSET[0])
		temp2=sf.get_Matrix(TESTSET[1])
		temp3=sf.get_Matrix(TESTSET[2])
		self.mew = []
		for i in range(len(TESTSET)):
			self.mew.append(sf.Mean(TESTSET[i]))
		inv_class1=sf.get_Inverse(temp1)
		inv_class2=sf.get_Inverse(temp2)
		inv_class3=sf.get_Inverse(temp3)
		self.Class1_test_Matrix=sf.get_Matrix(TESTSET[0])
		self.Class2_test_Matrix=sf.get_Matrix(TESTSET[1])
		self.Class3_test_Matrix=sf.get_Matrix(TESTSET[2])
		# for i in range(len(TESTSET)):
		case = [0,1]
		for i in case:
			for j in range(len(TESTSET[i])):
				temp=[0,0,0]
				val1 = self.getGx(TESTSET[i][j][0],TESTSET[i][j][1],inv_class1,self.mew[0], self.Class1_test_Matrix,len(TESTSET[0]))
				val2 = self.getGx(TESTSET[i][j][0],TESTSET[i][j][1],inv_class2,self.mew[1], self.Class2_test_Matrix,len(TESTSET[1]))
				if(val1>val2): 	CONF[i][0]=CONF[i][0]+1
				else: CONF[i][1]=CONF[i][1]+1
		print ("Confusion Matrix for class 1 & 2")
		for i in range(2):
			for j in range(2):
				print(CONF[i][j], end=" ")
			print("")
		sf.get_Score(CONF)

		CONF=[[0,0],[0,0]]

		case = [1,2]
		for i in case:
			for j in range(len(TESTSET[i])):
				temp=[0,0,0]
				val2 = self.getGx(TESTSET[i][j][0],TESTSET[i][j][1],inv_class2,self.mew[1], self.Class2_test_Matrix,len(TESTSET[1]))
				val3 = self.getGx(TESTSET[i][j][0],TESTSET[i][j][1],inv_class3,self.mew[2], self.Class3_test_Matrix,len(TESTSET[2]))
				if(val2>val3): 	CONF[i-1][0]=CONF[i-1][0]+1
				else: CONF[i-1][1]=CONF[i-1][1]+1
		print ("Confusion Matrix for class 2 & 3")
		for i in range(2):
			for j in range(2):
				print(CONF[i][j], end=" ")
			print("")
		sf.get_Score(CONF)

		CONF=[[0,0],[0,0]]

		case = [0,2]
		for i in case:
			for j in range(len(TESTSET[i])):
				temp=[0,0,0]
				val1 = self.getGx(TESTSET[i][j][0],TESTSET[i][j][1],inv_class1,self.mew[0], self.Class1_test_Matrix,len(TESTSET[0]))
				val3 = self.getGx(TESTSET[i][j][0],TESTSET[i][j][1],inv_class3,self.mew[2], self.Class3_test_Matrix,len(TESTSET[2]))
				k = 0
				if(i==2): k =1
				if(val1>val3): 	CONF[k][0]=CONF[k][0]+1
				else: CONF[k][1]=CONF[k][1]+1
		print ("Confusion Matrix for class 2 & 3")
		for i in range(2):
			for j in range(2):
				print(CONF[i][j], end=" ")
			print("")
		sf.get_Score(CONF)
		


	def get_ConfMatrix(self,TESTSET):
		CONF=[[0,0,0],[0,0,0],[0,0,0]]
		self.Class1_test_Matrix=sf.get_Matrix(TESTSET[0])
		self.Class2_test_Matrix=sf.get_Matrix(TESTSET[1])
		self.Class3_test_Matrix=sf.get_Matrix(TESTSET[2])
		self.mew = []
		for i in range(len(TESTSET)):
			self.mew.append(sf.Mean(TESTSET[i]))
		# print (self.Class1_test_Matrix,self.Class2_test_Matrix)
		inv_class1=sf.get_Inverse(self.Class1_test_Matrix)
		inv_class2=sf.get_Inverse(self.Class2_test_Matrix)
		inv_class3=sf.get_Inverse(self.Class3_test_Matrix)
		temp1=sf.get_Matrix(TESTSET[0])
		temp2=sf.get_Matrix(TESTSET[1])
		temp3=sf.get_Matrix(TESTSET[2])
		# print (temp1,temp2)
		for i in range(len(TESTSET)):
			for j in range(len(TESTSET[i])):
				temp=[0,0,0]
				val1 = self.getGx(TESTSET[i][j][0],TESTSET[i][j][1],inv_class1,self.mew[0], temp1,len(TESTSET[0]))
				val2 = self.getGx(TESTSET[i][j][0],TESTSET[i][j][1],inv_class2,self.mew[1], temp2,len(TESTSET[1]))
				val3 = self.getGx(TESTSET[i][j][0],TESTSET[i][j][1],inv_class3,self.mew[2], temp3,len(TESTSET[2]))
				if(max(val1,val2,val3)==val1): 	CONF[i][0]=CONF[i][0]+1
				elif(max(val1,val2,val3)==val2): CONF[i][1]=CONF[i][1]+1
				else: CONF[i][2]=CONF[i][2]+1
		for i in range(3):
			for j in range(3):
				print(CONF[i][j], end=" ")
			print("")
		sf.get_Score(CONF)