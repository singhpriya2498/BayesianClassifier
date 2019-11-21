from __future__ import print_function		
import sys
import LinearlySeperableWithDiagonalMatrix
import DiagonalSeparateMatrix
import LinearlySeperableWithNonDiagonalMatrix
import ArbitraryMatrix
import statistics_func as sf	
import matplotlib.pyplot as plt
RANGE=[]
val=0.05
if sys.argv[2]=='1':
	RANGE=[[-10,25],[-15,20]]
	Class1_train,Class1_test=sf.get_data("Data1/Class1.txt")
	Class2_train,Class2_test=sf.get_data("Data1/Class2.txt")
	Class3_train,Class3_test=sf.get_data("Data1/Class3.txt")
if sys.argv[2]=='2':
	RANGE=[[-10,15],[-15,20]]
	Class1_train,Class1_test=sf.get_data("Data2/Class1.txt")
	Class2_train,Class2_test=sf.get_data("Data2/Class2.txt")
	Class3_train,Class3_test=sf.get_data("Data2/Class3.txt")
if sys.argv[2]=='3':
	RANGE=[[-10,1000],[0,2500]]
	val=1
	Class1_train,Class1_test=sf.get_data("Data3/Class1.txt")
	Class2_train,Class2_test=sf.get_data("Data3/Class2.txt")
	Class3_train,Class3_test=sf.get_data("Data3/Class3.txt")
DATASET=[Class1_train,Class2_train,Class3_train]
TESTSET=[Class1_test,Class2_test,Class3_test]
if sys.argv[1]=='1':
	model=LinearlySeperableWithDiagonalMatrix.Model(DATASET,RANGE)
	model.get_lines()
	model.plot_classes(val)
	model.plot_model(val)
	model.get_ConfMatrix_pair(TESTSET)
	model.get_ConfMatrix(TESTSET)
if sys.argv[1]=='2':
	model=LinearlySeperableWithNonDiagonalMatrix.Model(DATASET,RANGE)
	model.get_lines()
	model.plot_classes(val)
	model.print_Matrix()
	model.plot_model(val)
	model.get_ConfMatrix_pair(TESTSET)
	model.get_ConfMatrix(TESTSET)
if sys.argv[1]=='3':
	model=DiagonalSeparateMatrix.Model(DATASET)
	model.get_lines(int(sys.argv[2]))
	model.get_pair(int(sys.argv[2]),DATASET)
	model.plot_model()
	model.get_ConfMatrix_pair(TESTSET)
	model.get_ConfMatrix(TESTSET)
if sys.argv[1]=='4':
	model=ArbitraryMatrix.Model(DATASET)
	model.get_lines(int(sys.argv[2]))
	model.get_pair(int(sys.argv[2]), DATASET)
	model.plot_model()
	model.get_ConfMatrix_pair(TESTSET)
	model.get_ConfMatrix(TESTSET)