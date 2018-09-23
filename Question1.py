# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 11:20:23 2018

Note: the following code is written in Python 3.5.2 not Python 2.7
"""
import numpy as np
import sys
import time

from func import breakexit, runpower
	
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
	eigenvalue_list, eigenvectors = runpower(matrix, n, tolerance, return_vector=True)
	end = time.clock()
	
	print('eigenvalues: (in descending order)')
	print(eigenvalue_list)
	print('------------------------------------')
	print('eigenvectors: in the order of corresponding eigenvalues')
	print(eigenvectors)
	
	print("Power method takes ",end-start, " seconds. (CPU Time)")
	
