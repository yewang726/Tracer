# Implements a tracer engine class

import numpy as N
from ray_bundle import RayBundle, concatenate_rays
from trace_tree import RayTree

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
        tree - a list  used for track parent rays and child rays. Each element
            of the list is a ray bundle created after one iteration of the 
            tracer. Each bundle contains an array listing the parent ray in the
            previous bundle (see ray_bundle.py). When a ray branches, the child
            rays point back to the same index representing that same parent ray.
            Otherwise, the index of each ray points to the ray in the previous
            branch.
        """
        self._asm = parent_assembly
        
    def intersect_ray(self, bundle, surfaces, objects, surf_ownership, \
        ray_ownership, surf_relevancy):
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
        ret_shape = (len(surfaces), bundle.get_num_rays())
        #print ''
        #print 'ret_shape', ret_shape
        #print ''
        stack = N.zeros(ret_shape)
        owned_rays = N.empty(ret_shape, dtype=N.bool)
        
        # Bounce rays off each object
        for surf_num in xrange(len(surfaces)):
            # Elements of owned_rays[surfnum] set to 1 if (rays dont own any surface or rays own the actual surface) and the surface is relevant to these rays.
            owned_rays[surf_num] = ((ray_ownership == -1) | (ray_ownership == surf_ownership[surf_num])) & surf_relevancy[surf_num]
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
            stack[surf_num, owned_rays[surf_num]] = surfaces[surf_num].register_incoming(in_rays)

        # Raise an error if any of the parameters is negative
        if (stack < 0.).any():
            raise ValueError("Parameters must all be positive")
        
        # If parameter == 0, ray does not actually hit object, but originates from there; 
        # so it should be ignored in considering intersections.
      
        if (stack == 0.).any():
            zeros = N.where(stack == 0.)
            stack[zeros] = N.inf

        # Find the smallest parameter for each ray, and use that as the final one,
        # returns the indices.  If an entire column of the stack is N.inf (the ray misses
        # any surfaces), then take that column to be all False
        stack = ((stack == stack.min(axis=0)) & ~N.isinf(stack))
        
        return stack, owned_rays

    def ray_tracer(self, bundle, reps=100, min_energy=1e-10, tree=True):
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
        tree - if True, register each bundle in self.tree, otherwise only
            register the last bundle.
        
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
        
        surfs_per_obj = [len(obj.get_surfaces()) for obj in objects]
        surfs_until_obj = N.hstack((N.r_[0], N.add.accumulate(surfs_per_obj)))
        surf_ownership = N.repeat(N.arange(len(objects)), surfs_per_obj)
        ray_ownership = -1*N.ones(bund.get_num_rays())
        surfs_relevancy = N.ones((num_surfs, bund.get_num_rays()), dtype=N.bool)

        for i in xrange(reps):
            front_surf, owned_rays = self.intersect_ray(bund, surfaces, objects, \
                surf_ownership, ray_ownership, surfs_relevancy)

            outg = []
            record = []
            out_ray_own = []
            new_surfs_relevancy = []
            weak_ray_pos = []

            for surf_idx in xrange(num_surfs):
                inters = front_surf[surf_idx, owned_rays[surf_idx]]
                if not any(inters): 
                    surfaces[surf_idx].done()
                    continue
                surfaces[surf_idx].select_rays(N.nonzero(inters)[0])
                new_outg = surfaces[surf_idx].get_outgoing()
                new_record = new_outg
                #print 'surface index', surf_idx
                #print 'nonzero[0]',N.nonzero(inters[0])
                #print 'nonzero',N.nonzero(inters)                
                # Fix parent indexing to refer to the full original bundle:
                parents = N.nonzero(owned_rays[surf_idx])[0][new_outg.get_parents()]
                new_outg.set_parents(parents)
        
                # Delete rays with negligible energies
                delete = new_outg.get_energy() <= min_energy
                weak_ray_pos.append(delete)
                if delete.any():
                    new_outg = new_outg.delete_rays(N.nonzero(delete)[0])
                surfaces[surf_idx].done()

                # Aggregate outgoing bundles from all the objects
                outg.append(new_outg)
                record.append(new_record)
                
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

            if bund.get_num_rays() == 0:
                # All rays escaping
                break
            
            ray_ownership = N.hstack(out_ray_own)
            surfs_relevancy = N.hstack(new_surfs_relevancy)

        if not tree:
            # Save only the last bundle. Don't bother moving weak rays to end.
            record = concatenate_rays(record)
            self.tree.append(record)
        
        return bund.get_vertices(), bund.get_directions()


# vim: et:ts=4
