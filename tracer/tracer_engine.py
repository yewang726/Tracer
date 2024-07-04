# Implements a tracer engine class

import numpy as N
from tracer.ray_bundle import RayBundle, concatenate_rays
from tracer.trace_tree import RayTree
from tracer.accel_tree import KdTree
import gc
import time
import logging

class TracerEngine():
	"""
	Tracer Engine implements that actual ray tracing. It keeps track of the number
	of objects, and determines which rays intersected which object.
	"""
	def __init__(self, parent_assembly):
		"""
		Arguments:
		parent_assembly - the highest level assembly
		
		Attributes:
		_asm - the Assembly instance containing the model to trace through.
		"""
		self._asm = parent_assembly

	def intersect_ray(self, bundle, surfaces, surf_relevancy):
		"""
		Finds the first surface intersected by each ray.
		
		Arguments:
		bundle - the RayBundle instance holding incoming rays.

		
		Returns:
		
		"""
		nrays = bundle.get_num_rays()
		nsurfs = len(surfaces)
		rays_mins = N.ones(nrays)*N.inf
		earlier_hit = N.zeros(nrays, dtype=N.bool)
		earliest_surf = -1*N.ones(nrays, dtype=int)

		# Bounce rays off each object
		for surf_num in range(nsurfs):
			if surf_relevancy[surf_num].any():
				# If all rays are relevant all the bundle goes into in_rays
				if surf_relevancy[surf_num].all():
					in_rays = bundle
				else: # ...Otherwise, the bundle inherits the relevant only
					in_rays = bundle.inherit(surf_relevancy[surf_num])   
			else: # if not a relevant surface for intersection, next surface:
				continue
				
			# Fills the stack assigning rays to surfaces hit.
			surf_stack = surfaces[surf_num].register_incoming(in_rays) # find intersections using surface geometry manager
			surf_stack[surf_stack==0.] = N.inf # if intersection distance is 0, set this as not an intersection
			earlier_hit = surf_stack < rays_mins[surf_relevancy[surf_num]] # compare intersection distance with this surface with rays_mins relevant for this surface to find if the currently detected hit is the earliest.
			if earlier_hit.any():
				rays_mins[surf_relevancy[surf_num]] = N.where(earlier_hit, surf_stack, rays_mins[surf_relevancy[surf_num]]) # update earliest ray_intersection
				earliest_surf[surf_relevancy[surf_num]] = N.where(earlier_hit, surf_num, earliest_surf[surf_relevancy[surf_num]]) # update earliest surface intersected

		return earliest_surf, surf_relevancy

	def intersect_ray_old(self, bundle, surfaces, objects, surf_ownership, ray_ownership, surf_relevancy):
		"""
		Finds the first surface intersected by each ray.
		
		Arguments:
		bundle - the RayBundle instance holding incoming rays.
		ownership - an array with the owning object instance for each ray in the
			bundle, or -1 for no ownership.
		
		Returns:
		stack - an s by r boolean array for s surfaces and r rays, stating
			for each surface i=1..s if it is intersected by ray j=1..r
		owned_rays - same size as stack, stating whether ray j was tested at all
			by surface i
		"""

		nrays = bundle.get_num_rays()
		nsurfs = len(surfaces)
		ret_shape = (nsurfs, nrays)
		owned_rays = N.empty(ret_shape, dtype=N.bool)
		rays_mins = N.ones(nrays)*N.inf
		earlier_hit = N.zeros(nrays, dtype=N.bool)
		earliest_surf = -1*N.ones(nrays, dtype=int)
		surf_stack = N.zeros(nrays)

		# Bounce rays off each object
		for surf_num in range(nsurfs):
			# Elements of owned_rays[surfnum] set to 1 if (rays dont own any surface or rays own the actual surface) and the surface is relevant to these rays.
			owned_rays[surf_num] = ((ray_ownership == -1) | (ray_ownership == surf_ownership[surf_num])) & surf_relevancy[surf_num]
			# If no ray is owned, skip the rest
			if not owned_rays[surf_num].any():
				continue
			# If some rays are not owned, the bundle inherits the owned_rays only
			if (~owned_rays[surf_num]).any():
				in_rays = bundle.inherit(owned_rays[surf_num])   
			else: # ...Otherwise all the bundle goes into in_rays
				in_rays = bundle
			# Fills the stack assigning rays to surfaces hit.
			surf_stack = surfaces[surf_num].register_incoming(in_rays) # find intersections using surface geometry manager
			surf_stack[surf_stack==0.] = N.inf # if intersection distance is 0, set this as not an intersection
			earlier_hit = surf_stack < rays_mins[owned_rays[surf_num]] # compare intersection distance with this surface with rays_mins relevant for this surface to find if the currently detected hit is the earliest.
			#print 'earlier_hit', earlier_hit
			if earlier_hit.any():
				rays_mins[owned_rays[surf_num]] = N.where(earlier_hit, surf_stack, rays_mins[owned_rays[surf_num]]) # update earliest ray_intersection
				earliest_surf[owned_rays[surf_num]] = N.where(earlier_hit, surf_num, earliest_surf[owned_rays[surf_num]]) # update earliest surface intersected

		return earliest_surf, owned_rays

	def intersect_ray_accel(self, bundle, surfaces, objects, surf_relevancy):
		"""
		Finds the first surface intersected by each ray.
		!!WORK IN PROGRESS!!
		
		Arguments:
		bundle - the RayBundle instance holding incoming rays.
		ownership - an array with the owning object instance for each ray in the
			bundle, or -1 for no ownership.
		
		Returns:
		stack - an s by r boolean array for s surfaces and r rays, stating
			for each surface i=1..s if it is intersected by ray j=1..r
		owned_rays - same size as stack, stating whether ray j was tested at all
			by surface i
		"""
		nsurfs = len(surfaces)
		nrays = bundle.get_num_rays()
		ret_shape = (nsurfs, nrays)
		owned_rays = N.empty(ret_shape, dtype=N.bool)
		rays_mins = N.ones(nrays)*N.inf
		earliest_surf = -1*N.ones(nrays, dtype=int)
		latest_order = N.amax(surf_relevancy, axis=0)
		# Bounce rays off each object
		for surf_num in range(len(surfaces)):
			# Elements of owned_rays[surfnum] set to 1 if (rays dont own any surface or rays own the actual surface) and the surface is relevant to these rays.
			owned_rays[surf_num] = (surf_relevancy[surf_num]>=0) & (surf_relevancy[surf_num]<=latest_order)
			# If no ray is owned, skip the rest and build the stack
			if not owned_rays[surf_num].any():
				continue
			# If some rays are not owned, the bundle inherits the owned_rays only
			if (~owned_rays[surf_num]).any():
				in_rays = bundle.inherit(owned_rays[surf_num])
			   # ...Otherwise all the bundle goes into in_rays
			else:
				in_rays = bundle

			# Fills the stack assigning rays to surfaces hit.
			surf_stack = surfaces[surf_num].register_incoming(in_rays)
			surf_stack[surf_stack==0.] = N.inf
			earlier_hit = surf_stack < rays_mins[owned_rays[surf_num]]
			if earlier_hit.any():
				rays_mins[owned_rays[surf_num]] = N.where(earlier_hit, surf_stack, rays_mins[owned_rays[surf_num]])
				earliest_surf[owned_rays[surf_num]] = N.where(earlier_hit, surf_num, earliest_surf[owned_rays[surf_num]])
				# Remove relevancy of intersectiosn that would occur later in ray paths:
				
				latest_order[owned_rays[surf_num]] = N.where(earlier_hit, surf_relevancy[surf_num, owned_rays[surf_num]], latest_order[owned_rays[surf_num]])

		return earliest_surf, owned_rays

	def ray_tracer(self, bundle, reps=100, min_energy=1e-10, tree=True, accel=False, Kd_Tree=None):
		"""
		Creates a ray bundle or uses a reflected ray bundle, and intersects it
		with all objects, uses intersect_ray(). Based on the intersections,
		generates an outgoing ray in accordance with way the incoming ray
		reflects or refracts off any surfaces.
		
		Arguments:
		bundle - the initial incoming bundle
		reps - stop iteration after this many ray bundles were generated (i.e. 
			after the original rays intersected some surface this many times).
		min_energy - the minimum energy the rays have to have continue tracking
			them; rays with a lower energy are discarded. A float.
		tree - a list  used for track parent rays and child rays. Each element
			of the list is a ray bundle created after one iteration of the 
			tracer. Each bundle contains an array listing the parent ray in the
			previous bundle (see ray_bundle.py). When a ray branches, the child
			rays point back to the same index representing that same parent ray.
			Otherwise, the index of each ray points to the ray in the previous
			branch.tree. If True, register each bundle in self.tree, otherwise only
			register the last bundle.
		accel - Enables Kd_Tree acceleration if True.
		Kd_Tree - pre-determined acceleration structure given to avoid the tree buidling.
		
		Returns: 
		A tuple containing an array of vertices and an array of the the direcitons
		of the last outgoing raybundle (note that the vertices of the new bundle are the 
		intersection points of the previous incoming bundle)
		
		NB: the order of the rays within the arrays may change, but they are tracked
		by the ray tree
		"""

		self.reps = reps
		self.minener = min_energy
		self.tree = RayTree()
		bund = bundle
		if tree is True:
			self.tree.append(bund)
		
		# A list of surfaces and their matching objects:
		surfaces = self._asm.get_surfaces()
		objects = self._asm.get_objects()
		
		num_surfs = len(surfaces)
		num_rays = bund.get_num_rays()
		surfs_relevancy = N.ones((num_surfs, num_rays), dtype=N.bool)

		if accel:
			if Kd_Tree is None:
				max_depth = 8+1.3*N.log(num_surfs)
				logging.info('Maximum Kd tree depth %i'%max_depth)
				min_leaf = 1
				fast = False
				if accel == 'fast':
					fast = True

				self.Kd_Tree = KdTree(self._asm, max_depth, min_leaf, fast=fast)
			else:
				self.Kd_Tree = Kd_Tree
		else: # these are legacy arrays from the original code from Y. Meller. The objective is that the objects, through their own_rays and surface_for_next_iteration methods, drive the next bundle restrictions. It requires object-specific interaction managementm which is less relevant if we use a generic binary tree type acceleration.
			surfs_per_obj = [len(obj.get_surfaces()) for obj in objects]
			surfs_until_obj = N.hstack((N.r_[0], N.add.accumulate(surfs_per_obj)))
			surf_ownership = N.repeat(N.arange(len(objects)), surfs_per_obj)
			ray_ownership = -1*N.ones(num_rays)

		for i in range(reps):
			t0 = time.time()
			if accel:
				if accel == 'ordered': # not bringing any advantage.
					any_inter, surfs_relevancy = self.Kd_Tree.traversal(bund, ordered=True)
					t0 = time.time()
					if any_inter:
						front_surf, owned_rays = self.intersect_ray_accel(bund, surfaces, objects, surfs_relevancy)
				else:
					any_inter, surfs_relevancy = self.Kd_Tree.traversal(bund, ordered=False)
					t0 = time.time()
					if any_inter:
						front_surf, owned_rays = self.intersect_ray(bund, surfaces, surfs_relevancy)
			else:
				front_surf, owned_rays = self.intersect_ray_old(bund, surfaces, objects, surf_ownership, ray_ownership, surfs_relevancy)
				
			outg = []
			record = []
			out_ray_own = []
			new_surfs_relevancy = []
			weak_ray_pos = []

			for surf_idx in range(num_surfs):

				intersections = front_surf[owned_rays[surf_idx]] == surf_idx # -1 are excluded automagically here

				if not any(intersections):
					surfaces[surf_idx].done()
					continue
				surfaces[surf_idx].select_rays(N.nonzero(intersections)[0])
				new_outg = surfaces[surf_idx].get_outgoing()
				
				# Fix parent indexing to refer to the full original bundle:
				parents = N.nonzero(owned_rays[surf_idx])[0][new_outg.get_parents()]
				new_outg.set_parents(parents)

				# Add to record before culling low_energy rays
				record.append(new_outg)

				# Delete rays with negligible energies
				delete = new_outg.get_energy() <= min_energy
				weak_ray_pos.append(delete)
				if delete.any():
					new_outg = new_outg.delete_rays(N.nonzero(delete)[0])
				surfaces[surf_idx].done()

				# Aggregate outgoing bundles from all the objects
				outg.append(new_outg)

				if not accel:
					# Add new ray-ownership information to the total list:
					obj_idx = surf_ownership[surf_idx]
					surf_rel_idx = surf_idx - surfs_until_obj[obj_idx]
					object_owns_outg = objects[obj_idx].own_rays(new_outg, surf_rel_idx)
					out_ray_own.append(N.where(object_owns_outg, obj_idx, -1))

					# Add new surface-relevancy information, saying which surfaces
					# of the full list of surfaces must be checked next. This is
					# somewhat memory-intensize and requires optimization.
					surf_relev = N.ones((num_surfs, new_outg.get_num_rays()), dtype=N.bool)
					surf_relev[surf_ownership == obj_idx] = \
						objects[obj_idx].surfaces_for_next_iteration(new_outg, surf_rel_idx)
					new_surfs_relevancy.append(surf_relev)					

			bund = concatenate_rays(outg)

			if tree:
				# stores parent branch for purposes of ray tracking
				record = concatenate_rays(record)
				if record.get_num_rays() != 0:
					weak_ray_pos = N.hstack(weak_ray_pos)
					record = bund + record.inherit(N.nonzero(weak_ray_pos)[0])
					self.tree.append(record)
				#gc.collect() # This was found useful to avoid memory error when using large bundles and/or broadband sources.
                # This is not useful in Python3
			if bund.get_num_rays() == 0:
				# All rays escaping
				logging.info('Ray bundle depleted')
				break

			t1 = time.time()-t0
			if not accel:
				ray_ownership = N.hstack(out_ray_own)
				surfs_relevancy = N.hstack(new_surfs_relevancy)
			else:
				logging.info('trace time %s s' %t1)

		if not tree:
			# Save only the last bundle. Don't bother moving weak rays to end.
			record = concatenate_rays(record)
			self.tree.append(record)
		if bund.get_num_rays() != 0:
			logging.warning(bund.get_num_rays(), 'rays left at the end of the simulation')
			logging.warning('Remaining energy in last bundle:', N.sum(bund.get_energy())/N.sum(bundle.get_energy())*100., '%')
		return bund.get_vertices(), bund.get_directions()


# vim: ts=4
