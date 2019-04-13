import numpy as np
import pandas as pd
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import operator
import math
import csv
import collections

CorrelationCoeff = collections.namedtuple('CorrelationCoeff', ['corr', 'isSignificant'])

def roundFloat(floatValue) :
	return round(floatValue, 2)
def sumOfColumn( dataSet, columnName) :
	return sum(dataSet[columnName])

def meanOfColumn( dataSet, columnName) :
	return roundFloat(sum(dataSet[columnName])/len(dataSet))

def sumOfXminusXbar( dataSet, columnName) :
	mean = roundFloat(meanOfColumn(dataSet, columnName))
	tempCol = dataSet[columnName]- mean
	return roundFloat(sum(tempCol))

def sumOfSquares( dataSet, columnName) :
	return roundFloat(sum(dataSet[columnName]**2))

def sumOfColumnsMuliplication( dataSet, columnName1, columnName2) :
	return roundFloat(sum(dataSet[columnName1]*dataSet[columnName2]))

def stdDev(dataSet, columnName) :
	return roundFloat(math.sqrt(sum((dataSet[columnName]- meanOfColumn(dataSet, columnName))**2)/(len(dataSet)-1)))


def covariance(dataSet, featureColumn, actualValueColumn) :
	x_min_xbar = dataSet[featureColumn]- meanOfColumn(dataSet, featureColumn)
	y_min_ybar = dataSet[actualValueColumn]- meanOfColumn(dataSet, actualValueColumn)
	cov_numerator = x_min_xbar * y_min_ybar
	cor_num = sum(cov_numerator)/(len(dataSet)-1)
	return roundFloat(cor_num)

def findCorrCoff(dataSet, featureColumn, actualValueColumn) :
	cor_num = covariance(dataSet, featureColumn, actualValueColumn)
	stdDevOfFeatureColumn = stdDev(dataSet, featureColumn)
	stdDevOfValueColumn = stdDev(dataSet, actualValueColumn)
	cor_r = roundFloat((cor_num)/(stdDevOfFeatureColumn*stdDevOfValueColumn))
	return CorrelationCoeff(cor_r, (abs(cor_r) > (1.96/math.sqrt(len(dataSet)))))

def checkCorrelationCoeff(inputFile , valueColumn, *featureColumns ) :
	data = pd.read_csv(inputFile)
	for feature in featureColumns:
			corr_value = findCorrCoff(data, feature, valueColumn)
			significant_stmt = "It is significant" if corr_value.isSignificant else "It is not significant"
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
	print("------------------------------------------------------------------------")
	print("Matrix for solving equations")
	print("A = \n", A)
	B = np.array(y_matrix) 
	print("B =", B)
	print("------------------------------------------------------------------------")
	params = np.linalg.solve(A, B)
	roundedParams = np.asarray([roundFloat(param) for param in params])
	buildAnovaTable(inputFile, params,actualValueColumn, *featureColumn)
	return roundedParams


def estimateValue(params, *featureValue) :
	parameters = params[:-1]
	output = (parameters.dot(featureValue)) + params[-1]
	return output

def buildAnovaTable(inputFile, params, targetColumn, *featureNames) :
	features = list(featureNames)
	features.append("c")
	dataSet = pd.read_csv(inputFile)
	dataSet["estimateValue"] = params[len(params)-1]
	i = 0
	while i < len(featureNames) :
		dataSet["estimateValue"] = dataSet["estimateValue"] + params[i] * dataSet[featureNames[i]]
		i = i+1
	dataSet["squaredError"] = (dataSet["estimateValue"] - dataSet[targetColumn])**2
	dataSet["squaredErrorRegression"] = (dataSet["estimateValue"] - meanOfColumn(dataSet,targetColumn))**2
	SSE = sum(dataSet["squaredError"])
	SSR = sum(dataSet["squaredErrorRegression"])
	MSR = SSR / (len(params) -1)
	MSE = SSE / (len(dataSet) -len(params))
	F = MSR/MSE
	anovastats = pd.DataFrame(columns=('source', 'df', 'SS','MS','F'))
	anovastats["source"] = ["regression", "error" ," total"]
	anovastats["df"] = [len(params) -1, (len(dataSet) -len(params)), (len(dataSet) -1)]
	anovastats["SS"] = [SSR, SSE, SSE+SSR]
	anovastats["MS"] = [MSR, MSE, float('nan')]
	anovastats["F"] = [F, float('nan'),float('nan')]
	print("---------------------------Anova stats----------------------------------")
	print(anovastats)
	print("------------------------------------------------------------------------")

if __name__ == "__main__" :
	checkCorrelationCoeff("multivariate-date.csv","Salary","Education","Experience","Hours per week")
	params = findParametersForMultiVariate("multivariate-date.csv", "Education","Experience","Hours per week", actualValueColumn= "Salary")
	SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
	strval =""
	i = 0
	while i < len(params) -1 :
		strval = strval + str(params[i]) + "x" + str(i+1) + " + "
		#strval = strval + str(params[i]) + "x" + str(i+1).translate(SUB) + "+ "
		i += 1
	strval  = strval + str(params[i])
	output = estimateValue(params, 16, 5,50)
	print("Equation : ", strval )
	print("Output : ", output )


