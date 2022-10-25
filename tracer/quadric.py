# Implements quadric surfaces.
#
# References:
# [1] http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter4.htm
# [2] http://en.wikipedia.org/wiki/Quadric

import numpy as N
from tracer.geometry_manager import GeometryManager

class QuadricGM(GeometryManager):
    """
    A base class for quadric surfaces, to be derived for creation of specific
    quadric geometries. Each subclass should define the following methods:
    
    get_ABC(ray_bundle) - Given a RAyBundle instance, return A, B, C, the
        coefficients of a quadratic equation of t, the parametric position
        on each ray where it hits the surface (each of A, B, C is as long as
        the number of rays in ray_bundle).
    
    _normals(verts, dirs)
        Arguments:
        verts - an n by 3 array whose rows are points on the surace in global
            coordinates
        dirs - an n by 3 array whose columns are the respective incidence directions
        
        Returns:
        A 3 by n array with the rewpective normals to the surface at each of `verts`
    
    Additionally, overriding _select_coords(self, coords, prm) may be required.
    """
    
    def find_intersections(self, frame, ray_bundle):
        """
        Register the working frame and ray bundle, calculate intersections
        and save the parametric locations of intersection on the surface.

        Arguments:
        frame - the current frame, represented as a homogenous transformation
            matrix stored in a 4x4 array.
        ray_bundle - a RayBundle object with the incoming rays' data.

        Returns:
        A 1D array with the parametric position of intersection along each of
            the rays. Rays that missed the surface return +infinity.
        """
        GeometryManager.find_intersections(self, frame, ray_bundle)
        
        d = ray_bundle.get_directions()
        v = ray_bundle.get_vertices()
        n = ray_bundle.get_num_rays()
        c = self._working_frame[:3,3]
  
        params = N.empty(n)
        params.fill(N.inf)
        vertices = N.empty((3,n))
        
        # Gets the relevant A, B, C from whichever quadric surface, see [1]
        A, B, C = self.get_ABC(ray_bundle)

        # Identify quadric intersections        
        delta = B**2. - 4.*A*C
        any_inters = delta >= 1e-6
        num_inters = any_inters.sum()

        if num_inters == 0:
            self._vertices = vertices
            self._params = params #
            return params      

        A = A[any_inters]
        B = B[any_inters]
        C = C[any_inters]        
        
        delta = N.sqrt(B**2. - 4.*A*C)

        hits = N.empty((2,num_inters))
        hits.fill(N.nan)

        # Identify linear equations
        is_linear = A == 0
        # Identify B = 0 cases
        is_Bnull = B == 0

        # Solve linear intersections        
        hits[:,is_linear & ~is_Bnull] = N.tile(-C[is_linear & ~is_Bnull]/B[is_linear & ~is_Bnull], (2,1))     
        # Solve B = 0 cases (give bad information on N.sign(0))
        hits[0,~is_linear & is_Bnull] = -N.sqrt(-C[~is_linear & is_Bnull]/A[~is_linear & is_Bnull])
        hits[1,~is_linear & is_Bnull] = N.sqrt(-C[~is_linear & is_Bnull]/A[~is_linear & is_Bnull])
        # Solve quadric regular intersections
        q = -0.5*(B+N.sign(B)*delta)
        hits[0,~is_linear & ~is_Bnull] = q[~is_linear & ~is_Bnull]/A[~is_linear & ~is_Bnull]
        hits[1,~is_linear & ~is_Bnull] = C[~is_linear & ~is_Bnull]/q[~is_linear & ~is_Bnull]
       
        # Get intersection coordinates using rays parameters
        inters_coords = v[:,any_inters] + d[:,any_inters]*hits.reshape(2,1,-1)     

        # Quadrics can have two intersections. Here we allow child classes
        # to choose based on own method:
        select = self._select_coords(inters_coords, hits)
        not_missed = ~N.isnan(select)
        any_inters[any_inters] = not_missed
        select = N.array(select[not_missed], dtype=N.int_)
        params[any_inters] = N.choose(select, hits[:,not_missed])
        vertices[:,any_inters] = N.choose(select, inters_coords[...,not_missed])
        
        # Storage for later reference:
        self._vertices = vertices
        self._params = params
  
        return params
    
    def _select_coords(self, coords, prm):
        """
        Choose between two intersection points on a quadric surface.
        This is a default implementation that takes the first positive-
        parameter intersection point.
        
        The default behaviour is to take the first intersection not behind the
        ray's vertex (positive prm).
        
        Arguments:
        coords - a 2x3 array whose each row is the global coordinates of one
            intersection point of a ray with the sphere.
        prm - the corresponding parametric location on the ray where the 
            intersection occurs.
        
        Returns:
        The index of the selected intersection, or None if neither will do.
        """
        is_positive = prm > 0.
        select = N.empty(prm.shape[1])
        select.fill(N.nan)

        # If both are positive, use the smaller one
        select[N.logical_and(*is_positive)] = 1

        # If either one is negative, use the positive one
        one_pos = N.logical_xor(*is_positive)
        select[one_pos] = N.nonzero(is_positive.T[one_pos,:])[1]
        
        return select
        
    def select_rays(self, idxs):
        """
        With this method, the ray tracer informs the surface that of the
        registered rays, only those with the given indexes will be used next.
        This is used here to trim the internal data structures and save memory.
        
        Arguments:
        idx - an array of indexes referring to the rays registered in
            register_incoming()
        """
        self._idxs = idxs
        self._vertices = self._vertices[:,idxs].copy()
        
        # Normals to the surface at the intersection points are calculated by
        # the subclass' _normals method.
        self._norm = self._normals(self._vertices.T,
                self._working_bundle.get_directions()[:,idxs].T)
    
    def get_normals(self):
        """
        Report the normal to the surface at the hit point of selected rays in
        the working bundle.
        """
        return self._norm
    
    def get_intersection_points_global(self):
        """
        Get the ray/surface intersection points in the global coordinates.

        Returns:
        A 3-by-n array for 3 spatial coordinates and n rays selected.
        """
        return self._vertices
    
    def done(self):
        """
        Discard internal data structures. This should be called after all
        information on the latest bundle's results have been extracted already.
        """
        if hasattr(self, '_vertices'):
            del self._vertices
        if hasattr(self, '_idxs'):
            del self._norm
            del self._idxs
        GeometryManager.done(self)

# vim: et:ts=4
