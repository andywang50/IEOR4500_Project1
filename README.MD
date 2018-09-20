To run:

open console, go to the working directory. Then type "python Question1.py russell_cov.txt 0.01" and hit enter.

Note: 0.01 corresponds to the tolerance, as required by the project prompt.

Question1.py

	Question1 simply reuses professor's code.


Question2to4.py 

    Question 2

        Now the code reads the price data and fill the NAs using (which I would call) "geometric linear interpolation".
        E.g. If P0 = 1.0, P1 = NA, P2 = 1.21; then r1 = r2 = 10% 

        From today's (Sept.18th, 2018) discussion, it also makes sense to do a centered moving average (also on logarithmic scale) to fill the NAs. Have updated the fill_missing function so that both methods are available.

    Question 3
    
        How many eigenvalues to show? For now, 5.
        How to visualize? Plot the evolutions of the 5 leading eigenvalues over time
        Although it seems we don't need to worry about eigenvectors (according to wikipedia, the term 'spectra' 'only refers to eigenvalues of a matrix), the code also plot the changes in eigenvectors over time in the following way.

            E.g. For the 1st eigenvector v_1. compute min(|| v_1^(t) - v_1^(t-1) ||, (||v_1^(t) + v_1^(t-1) ||.
                
                plot the evolution of this value over time. If the eigenvectors stalize as T gets larger, we would expect the change decays to 0. (Although it seems not to be the case for 2nd, 3rd... eigenvectors). 

    Question4 
        uses k = 32, 64, 128, ..., 1024. Compare these results as well as result from question 2 to the results from np.linalg.eigh() function. Will save as a csv file (differences.csv).
