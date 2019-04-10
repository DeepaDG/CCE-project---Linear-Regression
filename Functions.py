import numpy as np
import pandas as pd
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import operator
import math
import csv
import collections

CorrelationCoeff = collections.namedtuple('CorrelationCoeff', ['corr', 'isSignificant'])

def sumOfColumn( dataSet, columnName) :
	return sum(dataSet[columnName])

def meanOfColumn( dataSet, columnName) :
	return sum(dataSet[columnName])/len(dataSet)

def sumOfXminusXbar( dataSet, columnName) :
	mean = meanOfColumn(dataSet, columnName)
	tempCol = dataSet[columnName]- mean
	return sum(tempCol)

def sumOfSquares( dataSet, columnName) :
	return sum(dataSet[columnName]**2)

def sumOfColumnsMuliplication( dataSet, columnName1, columnName2) :
	return sum(dataSet[columnName1]*dataSet[columnName2])

def stdDev(dataSet, columnName) :
	return math.sqrt(sum((dataSet[columnName]- meanOfColumn(dataSet, columnName))**2)/(len(dataSet)-1))


def covariance(dataSet, featureColumn, actualValueColumn) :
	x_min_xbar = dataSet[featureColumn]- meanOfColumn(dataSet, featureColumn)
	y_min_ybar = dataSet[actualValueColumn]- meanOfColumn(dataSet, actualValueColumn)
	cov_numerator = x_min_xbar * y_min_ybar
	cor_num = sum(cov_numerator)/(len(dataSet)-1)
	return cor_num

def findCorrCoff(dataSet, featureColumn, actualValueColumn) :
	cor_num = covariance(dataSet, featureColumn, actualValueColumn)
	stdDevOfFeatureColumn = stdDev(dataSet, featureColumn)
	stdDevOfValueColumn = stdDev(dataSet, actualValueColumn)
	cor_r = (cor_num)/(stdDevOfFeatureColumn*stdDevOfValueColumn)
	return CorrelationCoeff(cor_r, (cor_r > math.sqrt(1.96)/len(dataSet)) )

def testMain(inputFile , valueColumn, *featureColumns ) :
	data = pd.read_csv(inputFile)
	for feature in featureColumns:
			corr_value = findCorrCoff(data, feature, valueColumn)
			significant_stmt = "It is significant" if [corr_value.isSignificant] else "It is not significant"
			print ("Correlation Coefficient of feature column %s with value column %s is %s. %s." % (feature, valueColumn, corr_value.corr, significant_stmt))


def findParameters(inputFile, featureColumn, actualValueColumn) :
	# y = mx + c
	data = pd.read_csv(inputFile)
	r1c1 = sumOfSquares(data, featureColumn)
	r1c2 = sumOfColumn(data,featureColumn)
	r2c1 = sumOfColumn(data, featureColumn)
	r2c2 = len(data)
	A = np.array([[r1c1, r1c2],[r2c1, r2c2]])

	b1 = sumOfColumnsMuliplication(data, featureColumn,actualValueColumn)
	b2 =  sumOfColumn(data, actualValueColumn)
	B = np.array([b1, b2]) 
	
	return np.linalg.solve(A, B)


def findParametersForMultiVariate(inputFile, *featureColumn, actualValueColumn) :
	# y = mx + c
	features = list(featureColumn)
	features.append("c")
	dataset = pd.read_csv(inputFile)
	data = dataset
	data["c"] = 1

	matrix = []
	i =0
	y_matrix = []
	while i < len(features) :
		row1 = []
		j=0
		while j < len(features) :
			row1.append(sumOfColumnsMuliplication(data, features[i], features[j]))
			j+=1
		y_matrix.append(sumOfColumnsMuliplication(data, features[i],actualValueColumn))
		matrix.append(row1)
		i+=1
	A = np.array(matrix)
	print("A =", A)
	B = np.array(y_matrix) 
	print("B =", B)
	return np.linalg.solve(A, B)

def estimateValue(params, *featureValue) :
	parameters = params[:-1]
	print(params[-1])
	output = (parameters.dot(featureValue)) + params[-1]
	return output




if __name__ == "__main__" :
	testMain("multivariate-date.csv","Salary","Education","Experience","Hours per week")
	params = findParametersForMultiVariate("multivariate-date.csv", "Education","Experience","Hours per week", actualValueColumn= "Salary")
	SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
	strval =""
	i = 0
	while i < len(params) -1 :
		strval = strval + str(params[i]) + "x" + str(i) + "+ "
		#strval = strval + str(params[i]) + "x" + str(i).translate(SUB) + "+ "
		i += 1
	strval  = strval + str(params[i])
	output = estimateValue(params, 16, 5,50)
	print("Output : ", output )


