To run:
open console, go to the working directory. Then type "python Question1and4.py russell_cov.txt 0.01" and enter
Note: 0.01 corresponds to the tolerance, as required by the project prompt.

Question1and4.py (pretty much done)
	Question1 simply reuses professor's code.
	Question4 uses a default k = 32. Since n is big in our case, this runs slower than the original power method.

Question2and3.py (not done yet)
	Now the code reads the price data and fill the NAs using (which I would call) "geometric linear interpolation".
	E.g. If P0 = 1.0, P1 = NA, P2 = 1.21; then r1 = r2 = 10% 
	Haven't implemented question3 yet.
