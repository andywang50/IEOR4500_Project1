To run:

open console, go to the working directory. Then type "python Question1and4.py russell_cov.txt 0.01" and hit enter.

Note: 0.01 corresponds to the tolerance, as required by the project prompt.

Question1and4.py (pretty much done)

	Question1 simply reuses professor's code.

	Question4 uses a default k = 32. Since n is big in our case, this runs slower than the original power method.

Question2and3.py (in progress)

    Question 2

        Now the code reads the price data and fill the NAs using (which I would call) "geometric linear interpolation".
        E.g. If P0 = 1.0, P1 = NA, P2 = 1.21; then r1 = r2 = 10% 

        From today's (Sept.18th, 2018) discussion, it also makes sense to do a centered moving average (also on logarithmic scale) to fill the NAs. Have updated the fill_missing function so that both methods are available.

    Question 3
    
        How many eigenvalues to show? For now, 5.
        How to visualize? Plot the evolutions of the 5 leading eigenvalues over time
        Although it seems we don't need to worry about eigenvectors (according to wikipedia, the term 'spectra' 'only refers to eigenvalues of a matrix), the code also plot the changes in eigenvectors over time in the following way.

            E.g. For the 1st eigenvector v_1. compute || v_1^(t) - v_1^(t-1) || _2 (L2 norm). plot the evolution of this value over time. If the eigenvectors stalize as T gets larger, we would expect the change decays to 0. (Although it seems not to be the case for 2nd, 3rd... eigenvectors). 
