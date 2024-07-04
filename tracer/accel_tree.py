import numpy as N
import time
from tracer.object import AssembledObject
from ray_trace_utils.vector_manipulations import AABB
import logging

class Kdtodo(object):
	'''
	A class to keep track of the node traversal. Used by the traversal method of the tree.
	'''
	def __init__(self, node, tmin, tmax):
		self.node = node
		self.tmin = tmin
		self.tmax = tmax

class KdTree(object):
	'''
	Acceleration tree stucture class containing building methods and traversal methods.
	'''
	def __init__(self, assembly, max_depth=N.inf, min_leaf=1, debug=False, fast=False):
		logging.debug('Building tree')
		self.nodes = []
		t0 = time.time()

		# Get boundaries
		if isinstance(assembly, AssembledObject):
			objects = [assembly]
		else:
			objects = assembly.get_objects()

		n_surfs = len(assembly.get_surfaces())
		boundaries = [o.get_boundaries() for o in objects]
		bounds_per_surf = N.array([len(b) for b in boundaries]) # needed for multi[ple boundaries per shape
		total_bounds = N.sum(bounds_per_surf) # total number of boundaries
		surfs_idx = N.repeat(N.arange(n_surfs), bounds_per_surf) # indices of shapes belonging to each boundary

		minpoints = N.empty((3,total_bounds))
		maxpoints = N.empty((3,total_bounds))
		bounds = N.empty((3,2*total_bounds))
		
		self.always_relevant = [] # This handles situations in which we have no declared boundaries. This attribute gets used bythe traversal algorithm to always make the listed bjects relevant in the ray-trace.
		i = 0
		# load all the data

		for index, bounds_o in enumerate(boundaries):
			if bounds_o == []:
				self.always_relevant.append(index)
			else:
				for b in bounds_o:
					minpoints[:,i] = b._minpoint
					maxpoints[:,i] = b._maxpoint
					bounds[:,2*i] = b._minpoint
					bounds[:,2*i+1] = b._maxpoint
					i+=1

		# find the first bounding box
		self.minpoint, self.maxpoint = AABB(bounds)
		# Initialise the root node
		root_info = NodeInfo(minpoint=self.minpoint[:,None], maxpoint=self.maxpoint[:,None], level=0)
		nodes_info = self.add_node(1, [root_info])
		n_nodes = len(self.nodes) # Number of nodes actively added to the tree
		node_idx = 0 # index of the currently allocated node.

		if fast == True:
			n_bounds = 12
		else:
			n_bounds = None
			
		building = True
		while building:
			if n_nodes >= len(self.nodes):
				nodes_info = self.add_node(n_nodes*2, nodes_info)

			# Identify current node
			minpoint, maxpoint = nodes_info[node_idx].get_bounds()
				
			# find surfaces in this node
			in_node = N.logical_and((maxpoints>=minpoint).all(axis=0), (minpoints<=maxpoint).all(axis=0))
			in_node_count = N.count_nonzero(in_node)

			# Set the current level:
			node_level = nodes_info[node_idx].get_level()

			if (in_node_count<=min_leaf) or (node_level>=max_depth):
				# make node a leaf:
				self.nodes[node_idx].flag = 3
				self.nodes[node_idx].surfaces_idxs = surfs_idx[in_node]
			else:
				# find/determine split
				bounds_in_node = bounds[:, N.tile(in_node,2)]
				split = self.determine_split(minpoint, maxpoint, minpoints[:,in_node], maxpoints[:,in_node], bounds_in_node, n_bounds=n_bounds)
				if split[0] == 3:
					# make parent node a leaf:
					self.nodes[node_idx].flag = 3
					self.nodes[node_idx].surfaces_idxs = surfs_idx[in_node]
				else:
					self.nodes[node_idx].flag = int(split[0])
					self.nodes[node_idx].split = split[1]
					self.nodes[node_idx].child = n_nodes

					# create child 1:
					maxpoint_split = N.copy(maxpoint)
					maxpoint_split[split[0]] = split[1]
					nodes_info[n_nodes].set_bounds(minpoint, maxpoint_split)
					nodes_info[n_nodes].set_level(node_level+1)

					# create child 2:
					minpoint_split = N.copy(minpoint)
					minpoint_split[split[0]] = split[1]
					nodes_info[n_nodes+1].set_bounds(minpoint_split, maxpoint)
					nodes_info[n_nodes+1].set_level(node_level+1)

					n_nodes += 2

			node_idx += 1
			building = node_idx != n_nodes

		# remove unused node data
		self.nodes = self.nodes[:node_idx]
		self.n_surfs = n_surfs
		if debug:
			self.nodes_info = nodes_info[:node_idx]

		t1 = time.time()-t0
		logging.debug('build_time: ', t1, 's')
		logging.debug('maximum level', node_level)
		self.build_time = t1
		logging.debug('Kd-Tree built')


	def determine_split(self, minpoint_parent, maxpoint_parent, minpoints, maxpoints, bounds, n_bounds=None, t_trav=1., t_isec=500., emptyBonus=0.2):
		'''
		Based on:
		https://pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Kd-Tree_Accelerator

		returns:
		- axis, split value, N_A, N_B: when the split is successful
		- 3, [list of surfaces indices], if the node is a leaf
		- n_bounds is to try to fix a number of tested splits
		'''

		Ns = minpoints.shape[1]
		diag = maxpoint_parent-minpoint_parent
		sorted_axes = N.argsort(diag, axis=0)[::-1]
		S_inv = 1./(diag[0]*diag[1]+diag[1]*diag[2]+diag[2]*diag[0])
		basecost = t_trav-Ns*t_isec

		for a in sorted_axes:
			bounds_axis = bounds[a]
			minpoint_parent_axis = minpoint_parent[a]
			maxpoint_parent_axis = maxpoint_parent[a]
			above_min = bounds_axis>minpoint_parent_axis
			below_max = bounds_axis<maxpoint_parent_axis
			bounds_axis = bounds_axis[N.logical_and(above_min, below_max)]

			if len(bounds_axis)>0:
				minpoints_axis = minpoints[a]
				maxpoints_axis = maxpoints[a]
				bounds_axis = N.unique(bounds_axis)
				otheraxis0, otheraxis1 = (a+1)%3, (a+2)%3
				diag0, diag1 = diag[otheraxis0], diag[otheraxis1]
				diag0tdiag1 = diag0*diag1
				diag0pdiag1 = diag0+diag1
				if n_bounds is not None:
					if n_bounds<len(bounds_axis):
						idx = N.round(N.linspace(0, len(bounds_axis) - 1, n_bounds)).astype(int)
						bounds_axis = bounds_axis[idx]

				cost, SAH_res = N.inf, None
				i = 0
				for b in bounds_axis:
					N_A = N.count_nonzero(maxpoints_axis >= b) # Above
					N_B = N.count_nonzero(minpoints_axis <= b) # Below
					p_A = S_inv*(diag0tdiag1 +(maxpoint_parent_axis-b)*diag0pdiag1)
					p_B = S_inv*(diag0tdiag1 +(b-minpoint_parent_axis)*diag0pdiag1)
					b_e = ((N_A == 0) or (N_B == 0))*emptyBonus
					newcost = basecost + t_isec*(1.-b_e)*(p_A*N_A+p_B*N_B)

					if newcost<cost:
						cost = newcost
						SAH_res = i
					i+=1

				if SAH_res is not None:
					#logging.debug 'split', SAH_res, a, bounds_axis[SAH_res]
					return int(a), bounds_axis[SAH_res]
		return 3, Ns
			
		
	def add_node(self, n, nodes_info):
		for i in range(n):
			self.nodes.append(Node())
			nodes_info.append(NodeInfo())
		return nodes_info

	def direct_traversal(self, bundle, surfaces):
		'''
		!NOT WORKING!
		This is an attempt to introduce sequential intersections in the traversal instead of outside as is typically done in tracer. With this implementation we should theoretically be able to save some intersections.
		Close to pure PBRT, with some numpy indexing to attempt to benefit from array computations and storage advantage.
		'''
		t0 = time.time()

		poss, dirs = bundle.get_vertices(), bundle.get_directions()
		inv_dirs = 1./dirs

		bounds = N.array([self.minpoint, self.maxpoint])
		inters, t_mins, t_maxs = self.intersect_bounds(poss, dirs, inv_dirs, bounds)

		earliest_surf = -1*N.ones(poss.shape[1], dtype=N.int)
		owned_rays = N.empty((len(surfaces), poss.shape[1]), dtype=N.bool)

		n_inters = len(inters)

		any_inter = False

		if inters.any():
			# for rays that do intersect, go down the tree (or up?):
			for r in range(n_inters):
				if inters[r] == False:
					continue
				t_min, t_max = t_mins[r], t_maxs[r]
				todopos = 0
				to_do = []
				for i in range(64):
					to_do.append(Kdtodo(0, t_min, t_max))
				node = self.nodes[to_do[todopos].node]
				while True:
					if (t_maxs[r] < t_min):
						break
					# Is node a leaf?
					if node.flag != 3: #interior
						# is ray intersecting split? 
						split_axis, split_pos = node.flag, node.split
						t_plane = (split_pos-poss[split_axis,r]) * inv_dirs[split_axis,r]

						# Get node children and sort them:
						c1 = node.child
						c2 = c1 + 1
						belowfirst = (poss[split_axis,r] < split_pos) or (poss[split_axis,r] == split_pos and dirs[split_axis,r]<=0.)

						if ~belowfirst:
							c1, c2 = c2, c1

						if (t_plane>t_max) or (t_plane<=0.):
							node = self.nodes[c1]
						elif (t_plane<t_min):
							node = self.nodes[c2]
						else:
							to_do[todopos].node = c2
							to_do[todopos].tmin = t_plane
							to_do[todopos].tmax = t_max
							todopos += 1
							node = self.nodes[c1]
							t_max = t_plane
					else: # leaf
						in_rays = bundle.inherit([r])
						for s in node.surfaces_idxs:
							surf_stack = surfaces[s].register_incoming(in_rays)[0]
							if surf_stack<t_max:
								t_maxs[r] = surf_stack
								earliest_surf[r] = s
							owned_rays[s,r] = True
						any_inter = True

						if todopos>0:
							todopos -= 1
							node = self.nodes[to_do[todopos].node]
							t_min = to_do[todopos].tmin
							t_max = to_do[todopos].tmax
						else:
							break
		t1 = time.time()-t0
		logging.debug('traversal_time: ', t1, 's')
		return earliest_surf, owned_rays

	def traversal(self, bundle, ordered=False):
		'''
		Close to pure PBRT but the ray bundles have to test all potential surfaces in the relevant trea leaves simulatneously (ie. it is not checking the closest box first. this is due to the Tracer way of doing ray-tracing and is not modified yet.
		ordered is an attemps to replace the binary stack that determins the intersection-tested surfaces with a stack of integers that indicate which surfaces to test first and avoid doing too many tests. It does not seem to provide much benefits so far...
		'''
		t0 = time.time()

		poss, dirs = bundle.get_vertices(), bundle.get_directions()
		inv_dirs = 1./dirs

		bounds = N.array([self.minpoint, self.maxpoint])
		inters, t_mins, t_maxs = self.intersect_bounds(poss, dirs, inv_dirs, bounds)
		n_inters = len(inters)
		any_inter = False

		if ordered:
			surfaces_relevancy = -1*N.ones((self.n_surfs, bundle.get_num_rays()), dtype=N.int)
		else:
			surfaces_relevancy = N.zeros((self.n_surfs, bundle.get_num_rays()), dtype=N.bool)

		if inters.any():
			# for rays that do intersect, go down the tree (or up?):
			for r in range(n_inters):
				if inters[r] == False:
					continue
				t_min, t_max = t_mins[r], t_maxs[r]
				todopos = 0
				to_do = []
				for i in range(64):
					to_do.append(Kdtodo(0, t_min, t_max))
				node = self.nodes[to_do[todopos].node]
				if ordered:
					order = 0
				while True:

					if (t_maxs[r] < t_min):
						break
					# Is node a leaf?
					if node.flag != 3: #interior
						# is ray intersecting split? 
						split_axis, split_pos = node.flag, node.split
						t_plane = (split_pos-poss[split_axis,r]) * inv_dirs[split_axis,r]

						# Get node children and sort them:
						c1 = node.child
						c2 = c1 + 1
						belowfirst = (poss[split_axis,r] < split_pos) or (poss[split_axis,r] == split_pos and dirs[split_axis,r]<=0.)

						if ~belowfirst:
							c1, c2 = c2, c1

						if (t_plane>t_max) or (t_plane<=0.):
							node = self.nodes[c1]
						elif (t_plane<t_min):
							node = self.nodes[c2]
						else:
							to_do[todopos].node = c2
							to_do[todopos].tmin = t_plane
							to_do[todopos].tmax = t_max
							todopos += 1
							node = self.nodes[c1]
							t_max = t_plane
					else: # leaf
						if ordered:
							surfaces_relevancy[node.surfaces_idxs.tolist(),r] = order
							order += 1
						else:
							surfaces_relevancy[node.surfaces_idxs.tolist(),r] = True

						any_inter = True

						if todopos>0:
							todopos -= 1
							node = self.nodes[to_do[todopos].node]
							t_min = to_do[todopos].tmin
							t_max = to_do[todopos].tmax
						else:
							break

		# If some objects had no bounds, we malke them permanently relevant for all rays.
		if ordered:
			surfaces_relevancy[self.always_relevant,:] = 0
			any_inter = True
		else:
			surfaces_relevancy[self.always_relevant,:] = True
			any_inter = True

		t1 = time.time()-t0
		logging.debug('traversal_time: ', t1, 's')
		return any_inter, surfaces_relevancy

	def intersect_bounds(self, poss, dirs, inv_dirs, bounds):
		neg_dirs = N.array(dirs<0, dtype=int)
		t_mins = N.zeros(poss.shape[1])
		t_maxs = N.ones(poss.shape[1])*N.inf

		for i in range(3):
			t_mins_i = (bounds[neg_dirs[i],i]-poss[i])*inv_dirs[i]
			t_maxs_i = (bounds[1-neg_dirs[i],i]-poss[i])*inv_dirs[i]
			swap = t_mins_i>t_maxs_i

			t_mins_i[swap], t_maxs_i[swap] = t_maxs_i[swap], t_mins_i[swap]
			t_mins = N.amax([t_mins, t_mins_i], axis=0)
			t_maxs = N.amin([t_maxs, t_maxs_i], axis=0)

		inters = t_maxs>0
		inters[t_mins>t_maxs] = False
		return inters, t_mins, t_maxs

class Node(object):
	'''
	Lightweight object to keep nodes information in the Tree.
	node types: 
	- 0: interior x-split
	- 1: interior y-split
	- 2: interior z-split
	- 3: leaves
	If it is a leaf, it gets self.surfaces_idxs list with reference to surfaces indices in the assembly get_surfaces() output
	If it is an interior node, it gets a child.
	'''
	pass


class NodeInfo():
	def __init__(self, minpoint=None, maxpoint=None, level=None):
		"""
		NodeInfo instances are used to keep track of the information when building the tree.
		"""
		self.set_bounds(minpoint, maxpoint)
		self.set_level(level)

	def set_level(self, level):
		self.level = level

	def get_level(self):
		return self.level

	def set_bounds(self, minpoint, maxpoint):
		self.minpoint = minpoint
		self.maxpoint = maxpoint

	def get_bounds(self):
		return [self.minpoint, self.maxpoint]
