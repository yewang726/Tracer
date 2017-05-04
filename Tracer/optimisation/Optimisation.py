import numpy as N
import time

from optimisation.Scene_objectives import *
from optimisation.multiobjective_tools import *
#import matplotlib.pyplot as plt
import pickle

#from Optimisation_test_results_analysis import *

class Population():
	'''
	Population class to manage population of candidates being simulated and track their performance.
	'''
	def __init__(self, pop_count, scene_optim, fname='/home/charles/Documents/Boulot/These/Optimisation/Multiobjective/test_population'):
		# Initialises the population

		scenes = N.zeros(pop_count, dtype=dict)
		self.fname = fname
		f = open(self.fname+'_scenes','w')
		pickle.dump(scenes, f)
		f.close()

		self.scene_model = scene_optim
		self.termination_criteria = scene_optim.termination_criteria
		self.objectives_thresholds = scene_optim.objectives_thresholds

		self.good_scenes = N.arange(pop_count)
		self.bad_scenes = []
		self.faulty_scenes = []
		optim_progress = {'good_scenes':self.good_scenes, 'bad_scenes':self.bad_scenes, 'faulty_scenes': self.faulty_scenes}
		f = open(self.fname+'_optim_progress','w')
		pickle.dump(optim_progress, f)
		f.close()

		
		for i in xrange(pop_count):
			new_scene = self.scene_model.generate_scene()
			f = open(self.fname+'_scenes','r')
			scenes = pickle.load(f)
			scenes[i]={'params':new_scene}
			scenes[i]['generation'] = 0
			f = open(self.fname+'_scenes','w')
			pickle.dump(scenes,f)
			f.close()
			print 'Scene %s out of %s generated.'%(str(i+1), str(pop_count))
			print ' '
		self.geometry_boundaries = self.scene_model.starting_geometry_boundaries

	def update(self, scenes, progress):
		'''
		Updates the cerated population with existing scenes and progress data fro continuation of a previous run.
		'''
		self.good_scenes = progress['good_scenes']
		self.bad_scenes = progress['bad_scenes']
		self.faulty_scenes = progress['faulty_scenes']
		print 'Population updated'
		print 'Saving new data'
		f = open(self.fname+'_optim_progress','w')
		pickle.dump(progress, f)
		f.close()
		f = open(self.fname+'_scenes','w')
		pickle.dump(scenes,f)
		f.close()
		del scenes 
		del progress
		print 'data_saved'
				

	def brute_force(self, procs=8):
		# Method to run the optimisation. Contains the algorithm:

		# Initialises the optimisation timer
		t0 = time.clock()
		self.T_init = time.clock()-t0

		# 1st run to initialise the dicts of the scenes and the stats work.
		t0 = time.clock()
		f = open(self.fname+'_scenes','r')
		self.scenes = pickle.load(f)
		f.close()
		for i in xrange(len(self.scenes)):
			scene  = self.scenes[i]

			scene_new = self.scene_model
			objectives_new = scene_new.get_objectives(scene['params'], procs=procs)

			scene['rays'] = [float(scene_new.num_rays)]
			scene['objectives'] = [N.array(objectives_new)]
			scene['Q'] = [N.zeros(len(objectives_new))]
			scene['confidence_intervals'] = [N.ones(len(objectives_new))*N.inf]
		
		iterator = ((good_scene['confidence_intervals'][-1]>self.termination_criteria).any() for good_scene in self.scenes[self.good_scenes])
		
		stillgoing = N.fromiter(iterator, dtype=N.bool)

		while stillgoing.any():
			print str(N.sum(stillgoing))+'/'+str(len(stillgoing))+' are still valid candidates.'
			for i in xrange(len(self.scenes)):
				scene  = self.scenes[i]
				if (i == self.good_scenes).any():
					# Build the scene from the parameters created when initialising the population:
					scene_new = self.scene_model
					# Simulate and get the results:
					objectives_new = scene_new.get_objectives(scene['params'], procs=procs)

					# Append the scene dicts:
					scene['rays'].append(scene['rays'][-1]+float(scene_new.num_rays))
					scene['objectives'].append((scene['objectives'][-1]*scene['rays'][-2]+objectives_new*(scene['rays'][-1]-scene['rays'][-2]))/scene['rays'][-1])
					#scene['Q'].append(scene['Q'][-1]+scene['rays'][-2]/scene['rays'][-1]*(scene['objectives'][-1]-scene['objectives'][-2])**2)
					scene['Q'].append(scene['Q'][-1]+(objectives_new-scene['objectives'][-2])*(objectives_new-scene['objectives'][-1]))
					sigma = N.sqrt(scene['Q'][-1]/scene['rays'][-2])
					scene['confidence_intervals'].append(3.*sigma*N.sqrt(scene['rays'][-1]/scene['rays'][-2])/scene['objectives'][-1])


			# All scenes are simulated, now the statistical tests are performed:
			bests = N.zeros(len(self.scenes[0]['objectives'][-1]))
			cutoffs = N.zeros(len(bests))

			for o in xrange(len(bests)):
				iter_obj = (self.scenes[i]['objectives'][-1][o] for i in xrange(len(self.scenes)))
				bests[o] = N.argmax(N.fromiter(iter_obj, dtype=N.float, count=len(self.scenes)))
				cutoffs[o] = self.scenes[bests[o]]['objectives'][-1][o]*(1.-self.scenes[bests[o]]['confidence_intervals'][-1][o])

			print 'cutoffs: ', cutoffs
			print "best scenes indices: ",bests

			optima = N.zeros((len(self.scenes),len(bests)))

			# Flag good candidates to be simulated in the next iteration:
			for i in xrange(len(self.scenes)):
				optima[i] = (self.scenes[i]['objectives'][-1]*(1.+self.scenes[i]['confidence_intervals'][-1]))>cutoffs
			optima = N.hstack(optima.any(axis=1))

			self.good_scenes = N.hstack(N.argwhere(optima))
			print 'good_scenes :', self.good_scenes
			for s in xrange(len(self.good_scenes)):
				print 'confidence intervals :', self.scenes[self.good_scenes[s]]['confidence_intervals']

			# Flag bad scenes:
			if N.sum(optima) != len(self.scenes):
				self.bad_scenes = N.hstack(N.argwhere(~optima))

			iterator = ((good_scene['confidence_intervals'][-1]>self.termination_criteria).any() for good_scene in self.scenes[self.good_scenes])

			stillgoing = N.fromiter(iterator, dtype=N.bool)
			'''
			if stat_stab > 20:
				print 'max iter'
				break
			'''
		self.topt = time.clock()-t0

		f = open(self.fname+'_scenes','w')
		pickle.dump(self.scenes, f)
		f.close()

		return self.scenes, self.good_scenes, self.bad_scenes

	def MO_brute_force(self, procs=8):

		# Initialises the optimisation timer
		t0 = time.clock()
		self.T_init = time.clock()-t0

		# 1st run to initialise the dicts of the scenes and the stats work.
		t0 = time.clock()
		f = open(self.fname+'_scenes','r')
		self.scenes = pickle.load(f)
		f.close()
		for i in xrange(len(self.scenes)):
			print 'Evaluating scene ', i
			scene  = self.scenes[i]

			scene_new = self.scene_model
			objectives_new = scene_new.get_objectives(scene['params'], procs=procs)

			scene['rays'] = [float(scene_new.num_rays)]
			scene['objectives'] = [N.array(objectives_new)]
			scene['Q'] = [N.zeros(len(objectives_new))]
			scene['confidence_intervals'] = [N.ones(len(objectives_new))*N.inf]

		# Iterator to check if all valid candidates have reached the precision termination criterion.
		iterator = ((good_scene['confidence_intervals'][-1]>self.termination_criteria).any() for good_scene in self.scenes[self.good_scenes])
		stillgoing = N.fromiter(iterator, dtype=N.bool)

		while stillgoing.any():
			Objs = N.zeros((len(scene['objectives'][-1]), len(self.scenes)))
			ICs = N.zeros((len(scene['objectives'][-1]), len(self.scenes)))
			print str(N.sum(stillgoing))+'/'+str(len(stillgoing))+' are still valid candidates.'
			for i in xrange(len(self.scenes)):
				print 'Evaluating scene ', i
				scene  = self.scenes[i]
				if (i == self.good_scenes).any():
					# Build the scene from the parameters created when initialising the population:
					scene_new = self.scene_model
					# Simulate and get the results:
					objectives_new = scene_new.get_objectives(scene['params'], procs=procs)
					# Append the scene dicts:
					scene['rays'].append(scene['rays'][-1]+float(scene_new.num_rays))
					scene['objectives'].append((scene['objectives'][-1]*scene['rays'][-2]+objectives_new*(scene['rays'][-1]-scene['rays'][-2]))/scene['rays'][-1])
					scene['Q'].append(scene['Q'][-1]+(objectives_new-scene['objectives'][-2])*(objectives_new-scene['objectives'][-1]))
					sigma = N.sqrt(scene['Q'][-1]/scene['rays'][-2])
					scene['confidence_intervals'].append(3.*sigma*N.sqrt(scene['rays'][-1]/scene['rays'][-2])/scene['objectives'][-1])

				# Get the last objectives and confidence intervals to perform the stats:
				Objs[:,i] = scene['objectives'][-1]
				ICs[:,i] = scene['confidence_intervals'][-1]

			# All scenes are simulated, now the statistical tests are performed:
			good, bad, pareto = pareto_screening(objectives=Objs, ICs=ICs)

			self.good_scenes = N.hstack(N.argwhere(good))
			self.bad_scenes = N.hstack(N.argwhere(bad))

			iterator = ((good_scene['confidence_intervals'][-1]>self.termination_criteria).any() for good_scene in self.scenes[self.good_scenes])

			stillgoing = N.fromiter(iterator, dtype=N.bool)
			'''
			if stat_stab > 20:
				print 'max iter'
				break
			'''
		self.topt = time.clock()-t0

		f = open(self.fname+'_scenes','w')
		pickle.dump(self.scenes)
		f.close()

		return self.scenes, self.good_scenes, self.bad_scenes

	def MO_evo(self, active_pop_size=None, final_pop_size=None, max_population=1000, explo_coeff=0.5, procs=8):
		'''
		An evolutionary algorithm for multiobjective problems. Any candidate under-performing in the pool is replaced by a new candidate inspired from the good candidates or exploring the parameter space. The bad candidates are kept and still evaluated to check if they are back in the evaluation loop. The optimisation stops when a number of good_candidates whose uncertainity is below the termination criterion equal to the starting population size has been found
		'''
		# Initialises the optimisation timer
		t0 = time.clock()
		self.T_init = time.clock()-t0

		print 'Loading the population file in the optimiser'
		# Read the population file to get the scenes
		t0 = time.clock()
		f = open(self.fname+'_scenes','r')
		self.scenes = pickle.load(f)
		f.close()
		print 'Scenes loaded'

		# Active population size:
		if active_pop_size == None:
			self.active_pop_size = len(self.scenes)
		else:
			self.active_pop_size = active_pop_size

		# Final population target size:
		if final_pop_size == None:
			self.final_pop_size = len(self.scenes)
		else:
			self.final_pop_size = final_pop_size

		# Actualise the generation counter:
		generation = 0
		for i in xrange(len(self.scenes)):
			scn_gen = self.scenes[i]['generation']
			if scn_gen > generation:
				generation = scn_gen 

		#If the optimisation is a new one, a first run is necessary to initialise the active population count and start the statistics sampling.
		if generation == 0:
			# 1st run to initialise the dicts of the scenes and the stats work.
			for i in xrange(len(self.scenes)):
				print 'Evaluating scene ', i
				scene  = self.scenes[i]

				scene_new = self.scene_model
				objectives_new = scene_new.get_objectives(scene['params'], 0, procs=procs)
				# A test to ensure that the generated scene is not a bad one. If so, re run creation and 1st run:
				while N.isnan(objectives_new).any():
					new_scene = self.scene_model.generate_scene()
					self.scenes[i]={'params':new_scene}
					self.scenes[i]['generation'] = 0
					scene  = self.scenes[i]
					scene_new = self.scene_model
					objectives_new = scene_new.get_objectives(scene['params'], 0, procs=procs)
			
				scene['rays'] = [float(scene_new.num_rays)]
				scene['objectives'] = [N.array(objectives_new)]
				scene['Q'] = [N.zeros(len(objectives_new))]
				scene['confidence_intervals'] = [N.ones(len(objectives_new))*N.inf]

		
		# Initialisation of the iterator used to initialise stillgoing, active and locked_in.
		iterator = ((good_scene['confidence_intervals'][-1]>self.termination_criteria).any() for good_scene in self.scenes[self.good_scenes])
		stillgoing = N.fromiter(iterator, dtype=N.bool)
		active = N.sum(stillgoing)
		locked_in = len(self.good_scenes)-active

		# Statistics stability condition and counter to use if sampling is bad at early stages of the ray-trace.
		self.stats_cond = 0
		stats_stab = 0

		while locked_in < self.final_pop_size:
			print 'Still going: ', N.hstack(N.argwhere(stillgoing))
			# Initialise objectives and confidence intervals arrays for the pareto detection function.
			Objs = N.zeros((len(scene['objectives'][-1]), len(self.scenes)))
			ICs = N.zeros((len(scene['objectives'][-1]), len(self.scenes)))
			
			print str(active)+'/'+str(len(self.good_scenes))+' are still over the termination criterion'

			# Browse through the population and run only the valid ones:
			for i in xrange(len(self.scenes)):
				scene  = self.scenes[i]
				if (i == N.hstack(N.argwhere(stillgoing))).any():
					print ' '
					print 'Evaluating scene ', i
					# Build the scene from the parameters created when initialising the population:
					scene_new = self.scene_model

					# Simulate and get the results:
					objectives_new = scene_new.get_objectives(scene['params'], len(scene['rays']), procs=procs)
					while N.isnan(objectives_new).any(): # if by any chance there is a statistical issue on the second sampling and the results is wrong.
						objectives_new = scene_new.get_objectives(scene['params'], len(scene['rays']), procs=procs) # This is where we get the new results for this iteration

					# Append the scene dicts and perform the stats on the objectives:
					scene['params'] = scene_new.params
					scene['rays'].append(scene['rays'][-1]+float(scene_new.num_rays))
					scene['objectives'].append((scene['objectives'][-1]*scene['rays'][-2]+objectives_new*(scene['rays'][-1]-scene['rays'][-2]))/scene['rays'][-1])
					scene['Q'].append(scene['Q'][-1]+scene['rays'][-2]/scene['rays'][-1]*(scene['objectives'][-1]-scene['objectives'][-2])**2)
					sigma = N.sqrt(scene['Q'][-1]*scene['rays'][0]/scene['rays'][-2])
					scene['confidence_intervals'].append(3.*sigma/N.sqrt(scene['rays'][-1]/scene['rays'][0])/scene['objectives'][-1])

					print 'Objectives :', N.round(scene['objectives'][-1], decimals=4)
					print 'CIs',scene['confidence_intervals'][-1]
				'''
				# Faulty scenes filter:
				if N.isnan(objectives_new).any():
					self.scenes = self.scenes[:i]
					self.good_scenes = 
				'''
				# Get the last objectives and confidence intervals for each scene to perform the stats:
				Objs[:,i] = scene['objectives'][-1]
				ICs[:,i] = scene['confidence_intervals'][-1]

			# Test if the best possible objective respects the objective threshold:
			bad_objs = ((Objs*(1.+ICs)) <= N.vstack(self.objectives_thresholds)).any(axis=0)
			print 'Bad after checking thresholds', N.sum(bad_objs)

			if N.sum(~bad_objs)>len(self.scenes[0]['objectives'][0]): # if there are enough scenes in the good set, perform pareto front detection and screening. Limit is imposed by Qhull.
				# Statistical tests are performed:
				good_p, bad_p, pareto_p = pareto_screening(objectives=Objs[:,~bad_objs], ICs=ICs[:,~bad_objs])

				good = N.zeros(N.shape(Objs)[1], dtype=bool)
				bad = N.zeros(N.shape(Objs)[1], dtype=bool)
				good[~bad_objs] = good_p
				bad[bad_objs] = True
				bad[~bad_objs] = bad_p

				print 'Pareto front: ', N.hstack(N.argwhere(~bad_objs))[pareto_p]

			else:
				good = ~bad_objs
				bad = bad_objs
				print 'Not enough good_scenes to compute the pareto front'

				#print 'faulty_scenes', self.faulty_scenes
			# Indexing of good and bad scenes.
			if good.any():
				self.good_scenes = N.hstack(N.argwhere(good))
			else:
				self.good_scenes = []
			if bad.any():
				self.bad_scenes = N.hstack(N.argwhere(bad))
			else:
				self.bad_scenes = []		
		
			print 'good_scenes: ', self.good_scenes
			print 'bad_scenes: ', self.bad_scenes

			stats_stab+=1 # Statistics stability condition to use if sampling is bad at early stages of the ray-trace.
			# Actualise trhe iterator taking into account the termination criterion and statistical threshold.
			#iterator = ((((self.scenes[scn]['confidence_intervals'][-1]-self.termination_criteria>0.).any() or (stats_stab<self.stats_cond)) and (scn==self.good_scenes).any()) for scn in xrange(len(self.scenes)))
			#stillgoing = N.fromiter(iterator, dtype=N.bool)
			precise = (ICs<N.vstack(self.termination_criteria)).all(axis=0)
			stillgoing = N.logical_and(~precise, good)
			
			active = N.sum(stillgoing)
			locked_in = len(self.good_scenes)-active
			if N.sum(stillgoing)>0:
				print 'Still going: ', N.hstack(N.argwhere(stillgoing))

			print locked_in,'/',self.final_pop_size,' are locked-in good so far.'

			# Check if the number of candidates reaching the thermination criterion is large enough to finish.
			if locked_in < self.final_pop_size:
				# Check if there are empty slots in the active population to generate new candidates.		
				if active < self.active_pop_size:

					# The number of active scenes remains constant. Identify if any slot is available.
					new_can = N.copy(self.active_pop_size - active)
					new_scenes = []
					
					# Fill any free space with a new scene and initialise the stats on it:
					cand = 0 # Generated candidates counter
					if len(self.good_scenes)>1:
						# Get geometrical boundaries for the newly generated candidates:
						geoms_shape = N.shape(self.scene_model.get_geometry(self.scenes[0]['params'])) # Get the shape of the geometry array
						geoms = N.zeros(N.hstack((len(self.good_scenes),geoms_shape))) # Initialise a geometry container array for the valid candidates population
						geoms_mins = N.zeros(geoms_shape) # Minimum values container for each parameter
						geoms_maxs = N.zeros(geoms_shape) # Maximum values container for each parameter

						# Fill the geoms array with the geometries of the good candidates:
						for gs in xrange(len(self.good_scenes)):
							geoms[gs] = self.scene_model.get_geometry(self.scenes[gs]['params'])

						# Identify minimums and maximums for each parameter:
						for par in xrange(N.shape(geoms)[1]):
							geoms_mins[par] = N.amin(geoms[:,par,:], axis=0)
							geoms_maxs[par] = N.amax(geoms[:,par,:], axis=0)

						# Actualise the generation counter and save the identified geometry boundaries for results investigation
						if len(self.bad_scenes)>0:
							generation += 1
							self.geometry_boundaries.append([geoms_mins,geoms_maxs])

					# Loop to create new candidates based on the previously identified boundaries.
					while (self.active_pop_size - active)>0: # positive test in case a previously bad candidate has been grabbed-back and the poputaion is too numerous.
						cand +=1
						print 'Generating new candidates: ', cand,'/', new_can
						scene = {}
						objectives_new = N.nan
						while N.isnan(objectives_new).any(): # A test to ensure that the generated scene is not a bad one. If so, re run creation and 1st run.
							if len(self.good_scenes)>1: # if there are enough good scenes to have different mins and maxs.
								# Choose between exploration and intensification of the search:
								behavior_selector = N.random.uniform(size=1)

								if behavior_selector < explo_coeff: # Exploration
									print 'Exploration'
									scene['params'] = self.scene_model.generate_scene()
									scene['generation'] = 0
								else: # Exploitation
									print 'Exploitation'
									scene['params'] = self.scene_model.generate_scene(geoms_mins, geoms_maxs)
									scene['generation'] = generation
							else: # otherwise, just expolre.
								print 'Exploration'
								scene['params'] = self.scene_model.generate_scene()
								scene['generation'] = 0
							# 1st simulation ring for each new candidate to initialise the scene dicts.
							scene_new = self.scene_model
							objectives_new = scene_new.get_objectives(scene['params'], source_idx=0, procs=procs)
							if N.isnan(objectives_new).any():
								self.faulty_scenes.append(scene)
						
						scene['rays'] = [float(scene_new.num_rays)]
						scene['objectives'] = [N.array(objectives_new)]
						scene['Q'] = [N.zeros(len(objectives_new))]
						scene['confidence_intervals'] = [N.ones(len(objectives_new))*N.inf]
						print 'Objectives :', N.round(scene['objectives'][-1], decimals=4)
						self.scenes = N.hstack((self.scenes, scene))
						new_scenes.append(len(self.scenes)-1)
						active += 1

				self.good_scenes = N.append(self.good_scenes, new_scenes)
				stillgoing = N.append(stillgoing, N.ones(new_can))				
				active = N.sum(stillgoing)
				locked_in = len(self.good_scenes)-active

			f = open(self.fname+'_scenes','w')
			pickle.dump(self.scenes, f)
			f.close()

			self.topt = time.clock()-t0

			optim_progress = {'good_scenes':self.good_scenes, 'bad_scenes':self.bad_scenes, 'faulty_scenes': self.faulty_scenes, 'time': self.topt}
			f = open(self.fname+'_optim_progress','w')
			pickle.dump(optim_progress, f)
			f.close()


		final_scenes = []
		for i in self.good_scenes:
			if (self.scenes[i]['confidence_intervals'][-1]<self.termination_criteria).all():
				final_scenes.append(i)

		return self.scenes, self.good_scenes, self.bad_scenes, final_scenes, self.geometry_boundaries


