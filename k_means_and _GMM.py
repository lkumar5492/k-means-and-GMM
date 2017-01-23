import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import copy

def kMeans(k, data, dataFile):
	randList = np.random.choice(data.index.size, k, replace = False)
	centreList = []
	for val in randList:
		centre_x = float(data.loc[val]["x"])
		centre_y = float(data.loc[val]["y"])
		centre = [centre_x, centre_y]
		centreList.append(centre)
	#print centreList

	
	for i in range(k):
		data[str(i)] = ((data["x"]-centreList[i][0])**2) + ((data["y"] - centreList[i][1])**2)
		data[str(i)] = data[str(i)]**(0.5)
	#print data
	colList = map(str, range(k)) 
	data["cluster"] = data[colList].idxmin(axis=1)

	while True:
		#### RECALCULATING CLUSTER CENTERS ####
		centreList = []
		for i in range(k):
			count = data.loc[data["cluster"] == str(i)]["cluster"].count()
			sum_x = data.loc[data["cluster"] == str(i)]["x"].sum()
			sum_y = data.loc[data["cluster"] == str(i)]["y"].sum()
			newCentre_x = (1.0/count) * sum_x
			newCentre_y = (1.0/count) * sum_y
			newCentre = [newCentre_x, newCentre_y]
			centreList.append(newCentre)

		for i in range(k):
			data[str(i)] = ((data["x"]-centreList[i][0])**2) + ((data["y"] - centreList[i][1])**2)
			data[str(i)] = data[str(i)]**(0.5)

		#print data
		newCluster = data[colList].idxmin(axis=1)

		#print newCluster
		if data["cluster"].equals(newCluster):
			#print True
			break
		else:
			data["cluster"] = newCluster

	#print data
	colors = ['r','b','g','k','c']
	for i in range(k):
		xList = data.loc[data["cluster"] == str(i)]["x"]
		yList = data.loc[data["cluster"] == str(i)]["y"]
		clusterList =  data.loc[data["cluster"] == str(i)]["cluster"]
		plt.scatter(xList, yList, label=clusterList,color = colors[i])
		plt.xlabel("K = " + str(k) + " for " + str(dataFile))

	plt.show()

def EM(dataFile, colName):
	dataFrame = pd.read_csv(dataFile, header=None, names= colName)
	K = 3

	maxLogLikelihood = float("-inf")
	maxMeanList = []
	maxCoVarList = {}
	maxPriorProb = []
	for run in range(5):
		data = dataFrame.copy()
		randList = np.random.choice(data.index.size, K, replace = False)
		meanList = []
		coVarList = {}
		coVarList["0"] = [[1,0], [0,1]]
		coVarList["1"] = [[1,0], [0,1]]
		coVarList["2"] = [[1,0], [0,1]]
		priorProb = [ 1.0/3, 1.0/3, 1.0/3]

		for val in randList:
			mean_x = float(data.loc[val]["x"])
			mean_y = float(data.loc[val]["y"])
			mean = [mean_x, mean_y]
			meanList.append(mean)

		
		logList = []
		while True:
			# print "MEAN LIST:"
			# print meanList
			for i in range(K):
				probList = []
				for j in range(data.index.size):
					x_val = data["x"][j]
					y_val = data["y"][j]
					prob = multivariate_normal.pdf([x_val, y_val], meanList[i], coVarList[str(i)])
					#print prob
					probList.append(prob)
				data["prob_" + str(i)] = probList
			#print probList
			#print data
			
			for i in range(K):
				data[str(i)] = (data["prob_" + str(i)] * priorProb[i])

			#print data

			data["totalProb"] = (data[str(0)] + data[str(1)] + data[str(2)])

			for i in range(K):
				data[str(i)] = (data[str(i)]) / (data["totalProb"])

			
			log = np.log(data["totalProb"]).sum()
			# print logList
			# print logList[-1:]
			# print log
			meanList = []
			for i in range(K):
				mean_x = 0.0
				mean_y = 0.0
				for j in range(data.index.size):
					mean_x = mean_x + (data[str(i)][j] * data["x"][j])
					mean_y = mean_y + (data[str(i)][j] * data["y"][j])
				mean_x = float(mean_x) / float(data[str(i)].sum())
				mean_y = float(mean_y) / float(data[str(i)].sum())
				mean = [mean_x, mean_y]
				meanList.append(mean)

			#print meanList
			coVarList = {}
			for i in range(K):
				x_x = (((data["x"] - meanList[i][0]) * (data["x"] - meanList[i][0]) * (data[str(i)])).sum())/float(data[str(i)].sum())
				x_y = ((data["x"] - meanList[i][0]) * (data["y"] - meanList[i][1]) * (data[str(i)])).sum()/float(data[str(i)].sum())
				y_x = ((data["y"] - meanList[i][1]) * (data["x"] - meanList[i][0]) * (data[str(i)])).sum()/float(data[str(i)].sum())
				y_y = ((data["y"] - meanList[i][1]) * (data["y"] - meanList[i][1]) * (data[str(i)])).sum()/float(data[str(i)].sum())

				coVar = [[x_x, x_y],[y_x, y_y]]
				coVarList[str(i)] = coVar

			#print coVarList
			priorProb = []
			for i in range(K):
				priorProb.append( float(data[str(i)].sum())/ float(data.index.size))
			#print priorProb
			if log in logList[-1:]:
				print "RUN " + str(run+1) + "::::::::::::"
				print "CLUSTER CENTRES:"
				print meanList

				print "CLUSTER CO-VARIANCE:"
				print coVarList

				print "PRIOR PROB.:"
				print priorProb

				print "LOG LIKELIHOOD:"
				print log

				if log > maxLogLikelihood:
					maxLogLikelihood = log
					for i in range(K):
						dataFrame[str(i)] = data[str(i)]
					maxMeanList = copy.deepcopy(meanList)
					maxCoVarList = copy.deepcopy(coVarList)
					maxPriorProb = copy.deepcopy(priorProb)

				break

			logList.append(log)
		
		#print data

		# print "==========   Histogram  ============"
		
		plt.plot(range(len(logList)), logList)
		plt.xlabel("No. of iterations")
		plt.ylabel("Log-likelihood")
	
	plt.show()

	#log = max(logList)
	print ""
	print "MAX LOG-LIKELIHOOD:"
	print maxLogLikelihood

	print "BEST CLUSTER CENTRES:"
	print maxMeanList

	print "BEST CLUSTER CO-VARIANCE:"
	print maxCoVarList

	print "BEST PRIOR PROB.:"
	print maxPriorProb

	colList = map(str, range(K)) 
	dataFrame["cluster"] = dataFrame[colList].idxmax(axis=1)

	#print data 
	colors = ['r','b','g','k','c']
	for i in range(K):
		xList = dataFrame.loc[dataFrame["cluster"] == str(i)]["x"]
		yList = dataFrame.loc[dataFrame["cluster"] == str(i)]["y"]
		clusterList =  dataFrame.loc[dataFrame["cluster"] == str(i)]["cluster"]
		plt.scatter(xList, yList, label=clusterList,color = colors[i])

	plt.show()

############### RBF L2 ###################
# def fetchKernelMatrix(data):
# 	kernelMatrix = pd.DataFrame()
# 	for i in range(data.index.size):
# 		kernelMatrix[str(i)] = np.exp((np.sqrt(((data - data.loc[i])**2).sum(axis = 1)))/ (2.0 * (-1.0)))
# 	return kernelMatrix

############### X^2 + Y^2 ###################
def fetchKernelMatrix(data):
	kernelMatrix = pd.DataFrame()
	for i in range(data.index.size): 
		kernelMatrix[str(i)] = ((data * data.loc[i]).sum(axis = 1)) + ((5.0 * (data ** 2).sum(axis = 1)) * (5.0 * (data.loc[i] ** 2).sum()))
	return kernelMatrix

############### RBF L1 ###################
# def fetchKernelMatrix(data):
# 	kernelMatrix = pd.DataFrame()
# 	for i in range(data.index.size):
# 		kernelMatrix[str(i)] = np.exp((np.sqrt((np.abs(data - data.loc[i])).sum(axis = 1)))/ (2.0 * (-1.0)))
# 	return kernelMatrix


################### POLYNOMIAL ###################
# def fetchKernelMatrix(data):
# 	kernelMatrix = pd.DataFrame()
# 	for i in range(data.index.size):
# 		kernelMatrix[str(i)] = (((data * data.loc[i])**2) + ((data * data.loc[i])**2)).sum(axis = 1)
	
# 	return kernelMatrix

# ################### COS ###################
# def fetchKernelMatrix(data):
# 	kernelMatrix = pd.DataFrame()
# 	for i in range(data.index.size):
# 		kernelMatrix[str(i)] = (np.sin(data * data.loc[i]) + np.cos(data * data.loc[i])).sum(axis = 1)
	
# 	return kernelMatrix

def kernelKMeans(K, data, dataFile):
	kernelMatrix = fetchKernelMatrix(data)
	#print kernelMatrix
	clusterList = []
	for i in range(data.index.size):
		clusterList.append(str(np.random.choice(K, 1)[0]))
	data["cluster"] = clusterList

	# print data
	while True:
		for i in range(K):
			xnList = data.loc[data["cluster"] == str(i)].index.tolist()
			total_sum = 0.0
			for l in xnList:
				total_sum = total_sum + (kernelMatrix.iloc[xnList][str(l)]).sum()
			
			distList = []
			for j in range(data.index.size):
				term1 = kernelMatrix[str(j)][j]
				term2 =  float(((kernelMatrix.iloc[xnList][str(j)]).sum()) * 2.0)/float(len(xnList))
				term3 = float(total_sum) / float((len(xnList))**2)
				finalTerm = term1 - term2 + term3
				#print finalTerm
				distList.append(finalTerm)
			data[str(i)] = distList

		colList = map(str, range(K)) 
		newCluster = data[colList].idxmin(axis=1)
		#print newCluster
		#print newCluster
		if data["cluster"].equals(newCluster):
			#print True
			break
		else:
			data["cluster"] = newCluster

	colors = ['r','b','g','k','c']
	for i in range(K):
		xList = data.loc[data["cluster"] == str(i)]["x"]
		yList = data.loc[data["cluster"] == str(i)]["y"]
		clusterList =  data.loc[data["cluster"] == str(i)]["cluster"]
		plt.scatter(xList, yList, label=clusterList,color = colors[i])
		plt.xlabel("K = " + str(K) + " for " + str(dataFile))

	plt.show()

if __name__ == "__main__":
	dataFile = "hw5_blob.csv"
	colName = ["x","y"]
	dataFrame = pd.read_csv(dataFile, header=None, names= colName)
	#print data

	print "RUNNING K-MEANS FOR hw5_blob.csv :"
	# ############ K_MEANS CLUSTERING #####################
	K = [2, 3, 5]
	for k in K:
		data = dataFrame.copy()
		kMeans(k, data, dataFile)

	dataFile = "hw5_circle.csv"
	colName = ["x","y"]
	dataFrame = pd.read_csv(dataFile, header=None, names= colName)
	#print data


	print "RUNNING K-MEANS FOR hw5_circle.csv :"
	K = [2, 3, 5]
	for k in K:
		data = dataFrame.copy()
		kMeans(k, data, dataFile)

	############  KERNEL K_MEANS CLUSTERING ####################
	print "RUNNING KERNEL K-MEANS FOR hw5_circle.csv :"
	dataFile = "hw5_circle.csv"
	colName = ["x","y"]
	dataFrame = pd.read_csv(dataFile, header=None, names= colName)
	K = 2
	data = dataFrame.copy()
	kernelKMeans(K, data, dataFile)

	############ EXPECTATION_MAXIMIZATION #####################
	print "RUNNING EM ALGORITHM FOR hw5_blob.csv :"
	dataFile = "hw5_blob.csv"
	colName = ["x","y"]
	EM(dataFile, colName)



