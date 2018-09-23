# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 11:20:23 2018

Note: the following code is written in Python 3.5.2 not Python 2.7
"""
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

import time

from func import runpower, runpower_extracredit, fill_missing, breakexit

# a function to convert integers to ordinals, e.g. 1 to '1st', 5 to '5th'
ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(math.floor(n/10)%10!=1)*(n%10<4)*n%10::4])

if __name__ == "__main__":
	if len(sys.argv) != 3:  # the program name and the datafile
		# stop the program and print an error message
		sys.exit("usage: eigen.py datafile ")
	
	filename = sys.argv[1]
	tolerance = float(sys.argv[2])
	print("input", sys.argv[1], sys.argv[2])
	
	try:
		f = open(filename, 'r')
	except IOError:
		print ("Cannot open file %s\n" % filename)
		sys.exit("bye")
	
	# read data
	data = f.readlines()
	f.close()
	
	line0 = data[0].split()
	#print line0
	
	if len(line0) == 0:
		sys.exit("empty first line")
	
	n = int(line0[1])
	T = int(line0[3])
	
	print("n = ", n, ", T = ", T)
	
	matrix_raw = np.zeros((n,T))

	for i in range(n):
		#read line i + 2
		theline = data[i+1].split()
		#print i, " -> ", theline
		for j in range(T):
			if theline[j] == 'NA':
				valueij = np.nan
			else:
				valueij = float(theline[j])
			#print i, j, numberij
			matrix_raw[i][j] = valueij
	# missing data
	price_matrix = fill_missing(matrix_raw)
	#price_matrix = fill_missing(matrix_raw, method='ma')
	ret_matrix = price_matrix.shift(-1, axis=1) / price_matrix - 1
	ret_matrix = ret_matrix.dropna(axis=1)
	ret_mean = ret_matrix.mean(axis=1)
	ret_matrix = ret_matrix.sub(ret_mean, axis=0)
	
	breakexit('compute covariance matrix?')
	print("Computing covariance matrix...")
	cov = np.matmul(ret_matrix, ret_matrix.T) / (T - 2)
	## calcaulting the eigenvalues and eigenvectors given by numpy
	lmbda_np, v_np = np.linalg.eigh(cov)
	# numpy returns in ascending order, we prefer descending order
	lmbda_np = lmbda_np[::-1]
	v_np = v_np[:,::-1]
	print("Covariance matrix:")
	print(cov)
	
	breakexit('run algo?')
	print("Running power method...")

	start = time.clock()
	eigenvalue_list, eigenvectors = runpower(cov, n, tolerance, return_vector = True)
	end = time.clock()
	print("eigenvalues: (in descending order)")
	print(eigenvalue_list)
	print('------------------------------------')
	print('eigenvectors: in the order of corresponding eigenvalues')
	print(eigenvectors)
	print("Power method takes ",end-start, " seconds. (CPU time)")
	
	# calculate the difference between our eigenvalues with numpy's result
	evalue_diff = pd.DataFrame(np.nan, columns=["original"], index=range(n))
	length = len(eigenvalue_list)
	evalue_diff.loc[:length-1,"original"] = lmbda_np[:length] - np.asarray(eigenvalue_list)

## Question 4
	breakexit('start question4?')
	print("Doing question4.")
	for log2k in range(5,11):
		k = 2 ** log2k
		
		#print("Now running power method using the \'power of 2\' version, \
		#	where k = " + str(k) + ".")
		
		start = time.clock()
		eigenvalue_list_new = runpower_extracredit(cov, n, tolerance, k)
		length = len(eigenvalue_list_new)
		evalue_diff.loc[:length-1,k] = lmbda_np[:length] - np.asarray(eigenvalue_list_new)

		end = time.clock()
		#print(eigenvalue_list_extracredit)
		print("When k = ", k, " it takes ",end-start, " seconds. (CPU time)")
	evalue_diff.to_csv("differences.csv")
	print("We are not printing the eigenvalues and eigenvectors in the console this time (too many).\n "
	+ "However, in differences.csv you can see the calculated eigenvalues using different methods compared"
	+ " against the result given by np.linalg.eigh().")
	
## Question 3
	"""
	Top 5 eigenvectors over time
	L2 norms of the changes of leading 5 eigenvectors over time
	"""
	breakexit('start question3?')
	print("Doing question3.")
	num_evalues_toshow = 5
	eigenvalues_df = pd.DataFrame(np.nan, index=range(ret_matrix.shape[1]),
										columns=[ordinal(i+1) for i in range(num_evalues_toshow)])
	evector_change_df = pd.DataFrame(np.nan, index=range(ret_matrix.shape[1]),
										columns=[ordinal(i+1) for i in range(num_evalues_toshow)])
	
	evector_prev = np.zeros((n,num_evalues_toshow))	
	for t in eigenvalues_df.index:
		if (t+2)%10 == 0:
			print("Calculating submatrix of size:", t+2)
		ret_submatrix = ret_matrix.iloc[:,:t+2]
		cov_sub = np.matmul(ret_submatrix, ret_submatrix.T) / (t+1)
		tmp_evalue_arr, tmp_evectors = runpower(cov_sub, n, tolerance, num_evalues_toshow, True)
		
		# eigenvalues
		tmp_evalue_arr = np.asarray(tmp_evalue_arr)
		num_evalues = min(num_evalues_toshow, len(tmp_evalue_arr))
		eigenvalues_df.iloc[t,:num_evalues] = tmp_evalue_arr[:num_evalues]
		
		# the L2 norm of changes of leading eigenvectors
		# min(|| v_i(current) - v_i(previous) ||_2, || v_i(current) + v_i(previous) ||)
		# minus case
		evector_current = np.zeros((n, num_evalues_toshow))	
		evector_current[:,:num_evalues] = tmp_evectors
		evector_change = evector_current - evector_prev
		evector_change_norm = np.linalg.norm(evector_change, 2, axis=0)
		# plus case
		evector_change2 = evector_current + evector_prev
		evector_change_norm2 = np.linalg.norm(evector_change2, 2, axis=0)
		# which is smaller?
		evector_change_df.iloc[t, :num_evalues_toshow] = np.min([evector_change_norm,evector_change_norm2],axis=0)
		# update eigenvectors		
		evector_prev = evector_current
	
	# plot eigenvalues over time
	eigenvalues_df.plot()
	plt.title("Top " + str(num_evalues_toshow) + " eigenvalues over time")
	plt.savefig("eigenvalues.png")
	plt.close()
	# zoom in the plot
	eigenvalues_df.iloc[100:,].plot()
	plt.title("Top " + str(num_evalues_toshow) + " eigenvalues over time (zoom)")
	plt.savefig("eigenvalues_zoom.png")
	plt.close()
	
	# plot the change of eigenvectors over time
	for col in evector_change_df.columns:
		evector_change_df.loc[:,col].plot()
		plt.title("Change of the calculated" + col + " eigenvector (in L2 norm) over time")
		plt.savefig(col+"-eigenvector.png")
		plt.close()
	print("The evolution of eigenvalues and eigenvectors are saved as png files in the working directory.")