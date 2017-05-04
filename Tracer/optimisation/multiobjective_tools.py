from scipy.spatial import ConvexHull
import numpy as N

'''
A multiobjectives optimisation toolbox.
'''

def pareto_screening(objectives, ICs):
	'''
	A function that detects the good and bad candidates of a population according to their objective values and confidence intervals in a n-dimensional space, n being the number of objectives.
	Inputs:
	- objectives: the current evaluation of the objectives of teh population elements.
	- ICs, the respective confidence inthervals on these objectives
	Outputs:
	- good_candidates; The boolean array of the candidates whose current best case scenario objectives place on or over the worst case scenarion pareto front.
	- bad_candidates: The boolean array of the candidates whose current best case scenario objectives are dominated by the worst case scenario pareto front.
	- pareto: The indices of the population elements belonging to the n-dimensional pareto front.
	'''
	# A-Pareto front detection
	# 1 Add the referential boundary points for the hull to be computed properly. 1 per axis at the max of a on the axis and one at the origin of the referential.
	for i in xrange(N.shape(objectives)[0]):
		add = N.zeros((N.shape(objectives)[0], 1))
		ind = N.argmax(objectives[i])
		add[i] = objectives[i,ind]
		objectives = N.concatenate((objectives, add), axis=1)
	objectives = N.concatenate((objectives, N.zeros((N.shape(objectives)[0],1))), axis=1)
	# NaN filter for objectives at 0. that get multiplied by infinity ICs
	nans = N.isnan(objectives)
	objectives[nans] = 0.
	#print 'objectives :', objectives
	#print 'ICs :',ICs
	# 2 Compute the convexhull with qhull using the scipy wrapper.
	hull = ConvexHull(objectives.T, qhull_options='Qc E1e-9 QbB')
	# 3 get indices of the hull points out of the vertices
	pareto = N.hstack(hull.vertices)
	pareto = N.unique(pareto)
	# 4 remove the boundary points from pareto indices:
	objectives = objectives[:,:-N.shape(objectives)[0]-1]
	pareto = pareto[:-N.shape(objectives)[0]-1]

	# B-Pareto region screening:
	bad_candidates = N.zeros(N.shape(objectives)[1], dtype=N.bool)
	good_candidates = N.zeros(N.shape(objectives)[1], dtype=N.bool)
	# 1 Pre-screen the objectives population using distance to the origin:
	# 	Take worst case scenario for all objectives:
	objectives_wc = objectives*(1.-ICs)
	# 	Detect minimum and maximum distance to the origin for the worst-case scenario pareto front members:
	dist_wc = N.sqrt(N.sum(objectives_wc**2., axis=0))
	min_d_objectives_wc_pareto = N.amin(dist_wc[pareto])
	max_d_objectives_wc_pareto = N.amax(dist_wc[pareto])
	# 	Take best case scenario for all objectives:
	objectives_bc = objectives*(1.+ICs)
	#	 compute their distances to the origin:
	dist_bc = N.sqrt(N.sum(objectives_bc**2., axis=0))
	# 	Take all points that have best-case distance lower than the minimum distance of the worst-case scenario pareto front to be underperforming:
	bad_candidates[dist_bc<min_d_objectives_wc_pareto] = True
	print 'Bad pre screening', N.sum(bad_candidates)
	# 	Take all points that have best-case distance higher than the maximum distance of the worst-case pareto front to be performing well.
	good_candidates[dist_bc>max_d_objectives_wc_pareto] = True
	# 	Add pareto indices to the good_candidates:
	good_candidates[pareto] = True
	# 2 Deal wth the rest of the points that are uncertain, they are yet not good_candidates nor bad candidates. Get the worst case scenario pareto front and add points that are uncertain and still to check. If some points that were not in the original pareto front appear in the new front when going through the convex hull method, they are good. Then run iteratively. When the hull only returns members of the original pareto front, all uncertain candidates that were good have been identified.
	pareto_bool = N.zeros(N.shape(objectives)[1], dtype=N.bool)
	pareto_bool[pareto] = True
	pareto_check = []
	region = N.zeros(N.shape(objectives))
	# stopping criterion init:
	stop = False
	while ~stop:
		# Identify candidates that still have to be screened:
		uncertain = ~N.logical_or(good_candidates, bad_candidates)
		# Start by adding all of them. 
		region[:,uncertain] = objectives_bc[:,uncertain]
		# Use the worst case scenario values for the pareto from members:
		region[:, pareto] = objectives_wc[:, pareto]
		valid = N.logical_and(uncertain, pareto_bool)

		# Build the necessary referential boundaries for the pareto front detection using a convex hull routine.
		add = N.zeros((N.shape(region)[0], N.shape(region)[0]))
		for i in xrange(N.shape(region)[0]):
			ind = N.argmax(region[i])
			add[i,i] = region[i,ind]
		region = N.concatenate((region, add), axis=1)
		# NaN filter for objectives at 0. that get multiplied by infinity ICs
		nans = N.isnan(region)
		region[nans] = 0.
		# Get the convex hull from the actual wc pareto and bc rest array.
		hull = ConvexHull(region.T, qhull_options='Qc E1e-9 QbB')
		pareto_check = N.hstack(hull.vertices)
		pareto_check = N.unique(pareto_check)
		# Remove the axes boundary points from the pareto indices array:
		region = region[:, :-N.shape(region)[0]]
		pareto_check = pareto_check[:-N.shape(region)[0]]
		# Delete all the elements of the new pareto front that are at the origin due to the region building method. 1st, create o boolean version of teh new pareto front.
		pareto_check_bool = N.zeros(N.shape(region)[1], dtype=N.bool)
		pareto_check_bool[pareto_check] = True
		# Find all the elements that are not uncertain or part of the original front in it and set them false.
		pareto_check_bool[~N.logical_or(pareto_bool,uncertain)] = False
		# If the hull is modified, take the new points that have appeared in the hull and get them to well_performing. This will automatically wthdraw them from the uncertain category on the next while iteration.
		pareto_identical = N.logical_and(pareto_check_bool, pareto_bool)
		pareto_check_bool[pareto_identical] = False
		good_candidates[pareto_check_bool] = True
		# Runs again by updating uncertain until the hull gets back to the worst cass pareto one.
		# WARNING: The convexity of the true pareto*(1-IC) front is not ensured due to the random nature of the confidence intervals. However, the convex hull method is used to detect the pareto_check front. As a consequence, the final pareto_check is convex but its objective might not be. In the present implementation, we choose to stick with the pareto_check convex front and ignore uncertain points that might lie in the convex regions between it and the pareto*(1-IC) front. By doing so we stick to a purely convex solution, keep the algorithm simple, robust and save computational time. The stoping criterion is not an exact match of the pareto and pareto check indices but that the pareto_check indices member are all in the pareto front.
		stop = ~pareto_check_bool.any()

	bad_candidates[uncertain] = True
	print 'Bad post screening', N.sum(bad_candidates)
	return good_candidates, bad_candidates, pareto
