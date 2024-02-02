# Scenario Discovery
This is the initial Scenario Discovery library for eMotional Cities project.

•	**PRIM_baseline800.ipynb**: given 800 LHS input points and their corresponding simulation values, this notebook performs the PRIM algorithm to find the most suitable box.

•	**ModelSD.py**: for a given number of experiments, it computes the total number of leisure trips distributed in a LHS input space for the studied dimensions: beta_female_travel, beta_TRANS_travel and beta_student_travel.

•	**Model.py**: Given specific coordinates for beta_female_travel, beta_TRANS_travel and beta_student_travel, it returns the computed total number of leisure trips.

•	**ModelSD_AL_PRIM.py**:

	1. Load an initial 200 LHS input sample point with their corresponding simulation output value.

	2. For a given number of iterations:
   
			a. Fit a GP with the initial points + new sampling points (in the first iteration, consider only the initial points)

			b. Calculate the posterior in 800 LHS sample points of the input space.
 
			c. Perform PRIM in these 800 LHS computed points.

			d. Uniformly sample from the simulation model a point inside or on the border of the selected PRIM box.

			e. Compute the output of the selected point given the simulation model and add it to the initial dataset
	Repeat these steps until the stopping criteria (normally defined by the number of iterations) is reached.

	3. Fit a GP with the final dataset of simulation points.

	4. Calculate the posterior value in 800 LHS samples of the input space.

•	**PRIM_AL.ipynb**: a notebook that, given the previously computed points, plots the points distributions presented in Figure 8 and finds the final box by performing the PRIM algorithm over the final posterior distribution.

•	**ModelSD_AL_PRIM_borders.py**: performs the same algorithm as before but sampling new points only from the borders of the boxes at each iteration.

•	**EI.ipynb**: notebook that computes the Expected Improvement (EI) algorithm with 200 random initial samples and 200 query samples.
•	Model_BO.py: given the coordinates of the three specific dimensions, it computes the number of leisure trips in negative.

In MAPs folder:

•	**cvt.py**: functions needed to perform the MAPs Elites algorithm defined and explained

•	**Common.py**: more functions of MAPs Elites algorithm defined and explained

•	**MAPs.py**: file that computes the MAPs algorithm with the specified set-up parameters.

•	**cvt_20000.dat**: file containing, for each raw, the number of iterations performed so far by the algorithm, the number of grids filled, the mean, median, and the 5 and 95 percentiles of the fitness values of the points in the grid

•	**archive_20000.dat**: file containing the archive for 20000 number of iterations. Each solution of the archive is described in each row of the file with its corresponding fitness value, the centroid coordinates where it is assigned, and the value of its coordinates in the features space: Beta_female_travel, beta_TRANS_travel and beta_student_travel.

•	**centroids_800_3.dat**: a file containing the coordinates of all the centroids that define the feature space.

•	**Model _extended.py** computes the total number of leisure trips changing all the 19 beta parameters of the day pattern binary. Returns the total number of leisure trips in negative (since we aim to minimize instead of maximize), and its dimensions in the features space (Beta_female_travel, beta_TRANS_travel and beta_student_travel).

•	**PRIM_MAPs.ipynb**: notebook that loads the final values of the archive, computes some statistics, and performs PRIM in the fitness values to find a box with the lowest number of leisure trips.

