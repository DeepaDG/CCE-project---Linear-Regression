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

def findCorrCoff(dataSet , featureColumn, actualValueColumn) :
	tempData = dataSet
	n = len(dataSet)
	tempData["x-xbar"] = dataSet[featureColumn]- meanOfColumn(dataSet, featureColumn)
	tempData["y-ybar"] = dataSet[actualValueColumn]- meanOfColumn(dataSet, actualValueColumn)
	tempData["cov_numerator"] = tempData["x-xbar"] * tempData["y-ybar"]
	cor_num = sum(tempData.cov_numerator)/(len(dataSet)-1)
	stdDevOfFeatureColumn = math.sqrt(sum(tempData["x-xbar"]**2)/(n-1))
	stdDevOfValueColumn = math.sqrt(sum(tempData["y-ybar"]**2)/(n-1))
	cor_r = (cor_num)/(stdDevOfFeatureColumn*stdDevOfValueColumn)

	return CorrelationCoeff(cor_r, (cor_r > math.sqrt(1.96)/n) )

def testMain() :
	data = pd.read_csv('Python_project.csv')
	corr_value = findCorrCoff(data, "x","y")
	print(corr_value.corr)
	print(corr_value.isSignificant)


if __name__ == "__main__" :
	testMain()
	


