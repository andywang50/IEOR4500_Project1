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

ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(math.floor(n/10)%10!=1)*(n%10<4)*n%10::4])


def fill_missing(matrix_raw, method='simple', window_size=5):
	"""
	fill the na entries in the price matrix
	Parameters
	----------
	matrix_raw: numpy array
		raw data
	method: string. 'simple' or 'ma'
		if simple, use linear interpolation on logarithmic scale
			e.g. P0 = 1.0, P1 = NA, P2 = 1.21. will fill P1 with 1.1
				so that r1 = r2 = 10%
		if 'ma', calculate center moving average on logarithmic scale,
			and then replace na with the corresponding moving average
	window_size: int
		only used if method == 'ma', size of moving average window.
	Returns
	-------
	a price matrix where na entries are filled
	"""
	assert method in ['simple', 'ma']
	matrix_log_df = pd.DataFrame(np.log(matrix_raw))
	if method == 'simple':
		matrix_log_df.interpolate(axis=1, inplace=True, limit_direction = 'both')
		result = np.exp(matrix_log_df)
	else:
		rolling_mean= matrix_log_df.rolling(window_size, center=True, min_periods=1, axis=1).mean()
		rolling_mean.update(matrix_log_df)
		result = np.exp(rolling_mean)
	return result

def breakexit(foo):
	stuff = input("("+foo+") break> ")
	if stuff == 'x' or stuff == 'q':
		sys.exit("bye")


def runpower_one(matrix, n):
	"""
	Calculate the leading eigenvalue and its corresponding eigenvector (normalized),
	using power method.
	Parameters
	----------
	matrix: 2D numpy array. 
		Assumed to be positive semi-definite.
	n: int
		size of the square matrix
	Returns
	-------
	eigenvalue: float
	eigenvector: np array. normalized so that L2 norm = 1.0
	"""
	#get initial vector
	v = np.zeros(n)
	w = np.zeros(n)
	for j in range(n):
		v[j] = np.random.uniform(0,1)
	#print 'matrix', matrix
	#print 'v', v
	T = 10000 #number of iterations
	tol = 1e-06
	oldnormw = 0
	for t in range(T):
		w = matrix.dot(v)
		#print 't', t, 'w',w
		normw = (np.inner(w,w))**.5
		v = w/normw
		#print 't',t,'v',v
		#print 't',t,'normw',normw, 'old', oldnormw
		if np.abs(normw - oldnormw)/normw < tol:
			#print ' breaking'
			break
		oldnormw = normw
	return normw, v
 

def runpower(matrix, n, tolerance, max_num=None, return_vector=False):
	"""
	Returns all the eigenvalues such that they are no smaller than a specific 
	fraction (specified by 'tolerance') than the leading eigenvalue.
	Calculation of eigenvalues is done using power method.
	Parameters
	----------
	matrix: 2D numpy array. 
		Assumed to be positive semi-definite.
	n: int
		size of the square matrix
	tolerance: float
		the tolerance e.g. 0.01
	max_num: int
		maximum number of eigenvalues to return, default=None
	return_vector: boolean
		whether eigenvectors will also be returned
	Returns
	-------
	list of eigenvalues in decreasing order
	"""
	calculate_next = True
	eigenvalue_list = []
	eigenvector_list = []
	while(calculate_next):	
		new_eigenvalue, v = runpower_one(matrix, n)
		if len(eigenvalue_list) == 0:
			leading_eigenvalue = new_eigenvalue
		eigenvalue_list.append(new_eigenvalue)
		eigenvector_list.append(v)
		if max_num is not None and len(eigenvalue_list) == max_num:
			break
		if abs(1.0 * new_eigenvalue / leading_eigenvalue) < tolerance:
			calculate_next = False
		else:
			matrix = matrix - new_eigenvalue * np.outer(v,v)
	if return_vector:
		return eigenvalue_list, np.asarray(eigenvector_list)
	else:
		return eigenvalue_list


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
	#price_matrix = fill_missing(matrix_raw)
	price_matrix = fill_missing(matrix_raw, method='ma')
	ret_matrix = price_matrix.shift(-1, axis=1) / price_matrix - 1
	ret_matrix = ret_matrix.dropna(axis=1)
	ret_mean = ret_matrix.mean(axis=1)
	ret_matrix = ret_matrix.sub(ret_mean, axis=0)
	
	breakexit('compute covariance matrix?')
	print("Computing covariance matrix...")
	cov = np.matmul(ret_matrix, ret_matrix.T) / (T - 2)
	
	breakexit('run algo?')
	print("Running power method...")

	start = time.clock()
	eigenvalue_list = runpower(cov, n, tolerance)
	end = time.clock()
	
	print("Power method takes ",end-start, " seconds.")

## Question 3
	"""
	Top 5 eigenvectors over time
	L2 norms of the changes of leading 5 eigenvectors over time
	"""
	num_evalues_toshow = 5
	eigenvalues_df = pd.DataFrame(np.nan, index=range(ret_matrix.shape[1]),
										columns=[ordinal(i+1) for i in range(num_evalues_toshow)])
	evector_change_df = pd.DataFrame(np.nan, index=range(ret_matrix.shape[1]),
										columns=[ordinal(i+1) for i in range(num_evalues_toshow)])
	evector_prev = np.zeros((num_evalues_toshow,n))	
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
		# || v_i(current) - v_i(previous) ||_2
		evector_current = np.zeros((num_evalues_toshow,n))	
		evector_current[:num_evalues,:] = tmp_evectors
		evector_change = evector_current - evector_prev
		evector_change_norm = np.linalg.norm(evector_change, 2, axis=1)
		evector_change_df.iloc[t, :num_evalues_toshow] = evector_change_norm
		evector_prev = evector_current
		
	eigenvalues_df.plot()
	plt.title("Top " + str(num_evalues_toshow) + " eigenvectors over time")
	plt.savefig("eigenvalues.png")
	plt.show()
	eigenvalues_df.iloc[100:,].plot()
	plt.title("Top " + str(num_evalues_toshow) + " eigenvectors over time (zoom)")
	plt.savefig("eigenvalues_zoom.png")
	plt.show()
	for col in evector_change_df.columns:
		evector_change_df.loc[:,col].plot()
		plt.title("Change of the calculated" + col + " eigenvector (in L2 norm) over time")
		plt.savefig(col+"-eigenvector.png")
		plt.show()