# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 11:20:23 2018

Note: the following code is written in Python 3.5.2 not Python 2.7
"""
import math
import numpy as np
import sys

import time
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
	tolerance = 1e-06
	oldnormw = 0
	for t in range(T):
		w = matrix.dot(v)
		#print 't', t, 'w',w
		normw = (np.inner(w,w))**.5
		v = w/normw
		#print 't',t,'v',v
		#print 't',t,'normw',normw, 'old', oldnormw
		if np.abs(normw - oldnormw)/normw < tolerance:
			#print ' breaking'
			break
		oldnormw = normw
	return normw, v
 



def runpower(matrix, n, tolerance):
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
	Returns
	-------
	list of eigenvalues in decreasing order
	"""
	calculate_next = True
	eigenvalue_list = []
	while(calculate_next):	
		new_eigenvalue, v = runpower_one(matrix, n)
		if len(eigenvalue_list) == 0:
			leading_eigenvalue = new_eigenvalue
		eigenvalue_list.append(new_eigenvalue)
		if abs(1.0 * new_eigenvalue / leading_eigenvalue) < tolerance:
			calculate_next = False
		else:
			matrix = matrix - new_eigenvalue * np.outer(v,v)
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
	print("n = ", n)
	
	
	matrix = np.zeros((n,n))
	
	line1 = data[1].split()
	#should check that line1[0] is the string 'matrix'
	for i in range(n):
		#read line i + 2
		theline = data[i+2].split()
		#print i, " -> ", theline
		for j in range(n):
			valueij = float(theline[j])
			#print i, j, numberij
			matrix[i][j] = valueij
	
	breakexit('run algo?')
	print("Running power method...")

	start = time.clock()
	eigenvalue_list = runpower(matrix, n, tolerance)
	end = time.clock()
	
	print(eigenvalue_list)
	
	print("Power method takes ",end-start, " seconds.")
	
