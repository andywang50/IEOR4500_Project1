# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 13:24:35 2018
Note: the following code is written in Python 3.5.2 not Python 2.7

"""
import math
import numpy as np
import pandas as pd
import sys

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
	leading_eigenvalue = np.nan
	while(calculate_next):	
		new_eigenvalue, v = runpower_one(matrix, n)
		if np.isnan(leading_eigenvalue):
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
		return eigenvalue_list, np.asarray(eigenvector_list).T
	else:
		return eigenvalue_list
def runpower_one_extracredit(matrix, n, k=1024):
	"""
	Returns the leading eigenvalue and its corresponding eigenvector (normalized),
	using the power method described in extracredit#2.
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

	m = matrix
	log2k = math.log2(k) 
	assert log2k == int(log2k)
	log2k = int(log2k)
	#normalize_factor = 1.0
	for i in range(log2k):
		largest_entry = np.max(abs(m))
#==============================================================================
# 		if largest_entry > 1:
# 			m = m / largest_entry
# 			normalize_arr[i] = largest_entry
#==============================================================================
		m = m / largest_entry
		#normalize_factor *= largest_entry ** ( 2 ** (log2k - i))
		m = np.matmul(m, m)
	v = np.zeros(n)
	w = np.zeros(n)
	for j in range(n):
		v[j] = np.random.uniform(0,1)		
	v = m.dot(v)
	normv = (np.inner(v, v)) ** .5
	v = v / normv
	w = matrix.dot(v)
	normw = (np.inner(w, w)) ** .5
	lmbda = normw #* normalize_factor
	return lmbda, w / normw

def runpower_extracredit(matrix, n, tolerance, k=1024):
	"""
	Returns all the eigenvalues such that they are no smaller than a specific
	fraction (specified by 'tolerance') than the leading eigenvalue.
	Calculation of eigenvalues is done using power method specified in extracredit #2.
	Parameters
	----------
	matrix: 2D numpy array.
	Assumed to be positive semi-definite.
	n: int
	size of the square matrix
	tolerance: float
	the tolerance e.g. 0.01
	Returns
	-------
	list of eigenvalues in decreasing order
	"""
	calculate_next = True
	eigenvalue_list = []
	while(calculate_next):	
		new_eigenvalue, v = runpower_one_extracredit(matrix, n, k)
		if len(eigenvalue_list) == 0:
			leading_eigenvalue = new_eigenvalue
		eigenvalue_list.append(new_eigenvalue)
		if abs(1.0 * new_eigenvalue / leading_eigenvalue) < tolerance:
			calculate_next = False
		else:
			matrix = matrix - new_eigenvalue * np.outer(v,v)
	return eigenvalue_list	