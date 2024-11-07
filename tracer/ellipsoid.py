import numpy as N
from .quadric import QuadricGM
from .spatial_geometry import rotx, roty, rotz

class Ellipsoid(QuadricGM):
	"""Implements the geometry of a circular paraboloid surface"""
	def __init__(self, a, b, c):
		"""			   
		Arguments: 
		a, b, c - describe the hyperboloid (x/a)**2 + (y/b)**2 + (z/c)**2 = 1
		
		Private attributes:																  
		a, b - describe the paraboloid as a*x**2 + b*y**2 +c*z**2 - 1 = 0
		""" 

		QuadricGM.__init__(self)
		self.a = 1./(a**2)
		self.b = 1./(b**2)
		self.c = 1./(c**2)
		
	def get_ABC(self, ray_bundle):
		"""
		Determines the variables forming the relevant quadric equation, [1]
		"""
		# Transform the the direction and position of the rays temporarily into the
		# frame of the paraboloid for calculations
		d = N.dot(self._working_frame[:3,:3].T, ray_bundle.get_directions())
		v = N.dot(N.linalg.inv(self._working_frame), 
			N.vstack((ray_bundle.get_vertices(), N.ones(d.shape[1]))))[:3]
		
		A = self.a*d[0]**2 + self.b*d[1]**2 + self.c*d[2]**2
		B = 2*self.a*d[0]*v[0] + 2*self.b*d[1]*v[1] + 2*self.c*d[2]*v[2]
		C = self.a*v[0]**2 + self.b*v[1]**2 + self.c*v[2]**2  - 1
		
		return A, B, C
		
	def _normals(self, hits, directs):
		"""
		Finds the normal to the parabola in a bunch of intersection points, by
		taking the derivative and rotating it. Used internally by quadric.
		
		Arguments:
		hits - the coordinates of intersections, as an n by 3 array.
		directs - directions of the corresponding rays, n by 3 array.
		"""
		hit = N.dot(N.linalg.inv(self._working_frame), N.vstack((hits.T, N.ones(hits.shape[0]))))
		dir_loc = N.dot(self._working_frame[:3,:3].T, directs.T)

		partial_x = 2*hit[0]*self.a
		partial_y = 2*hit[1]*self.b
		partial_z = 2*hit[2]*self.c
		
		local_normal = N.vstack((partial_x, partial_y, partial_z))
		local_unit = local_normal/N.sqrt(N.sum(local_normal**2, axis=0))

		down = N.sum(dir_loc * local_unit, axis=0) > 0.
		local_unit[:,down] *= -1

		normals = N.dot(self._working_frame[:3,:3], local_unit)

		return normals
		
class EllipsoidGM(Ellipsoid):
	def __init__(self, a, b, c, xlim=None, ylim=None, zlim=None):
		'''
		Arguments:
		a, b, c - semi axes along x, y and z. (think of it as radii along each axes)
		xlim, ylim, zlim - lists of minimum and maximum value used to truncate the ellipsoid along specific axes. intersectiosn beyond the limots are ignored.	
		'''
		Ellipsoid.__init__(self, a, b, c)
		if (N.hstack([xlim, ylim, zlim]) == None).any():
			self.xlim, self.ylim, self.zlim = xlim, ylim, zlim
			self.truncated = True
		else:
			self.xlim, self.ylim, self.zlim = [-a,a], [-b,b], [-c,c]
			self.truncated = False

	def _select_coords(self, coords, prm):
		"""
		Choose between two intersection points on a quadric surface.
		This implementation extends QuadricGM's behaviour by not choosing
		intersections outside the circular aperture.
		
		Arguments:
		coords - a 2 by 3 by n array whose each column is the global coordinates
			of one intersection point of a ray with the sphere.
		prm - the corresponding parametric location on the ray where the
			intersection occurs.

		Returns:
		The index of the selected intersection, or None if neither will do.
		"""
		select = N.empty(prm.shape[1])
		select.fill(N.nan)

		positive = prm > 1e-7
		
		coords = N.concatenate((coords, N.ones((2,1,coords.shape[2]))), axis=1)
		local = N.sum(N.linalg.inv(self._working_frame)[None,:3,:,None] * \
			coords[:,None,:,:], axis=2)

		if self.truncated:
			in_shape = N.ones(prm.shape, dtype=bool)
			if self.xlim is not None:
				in_shape *= N.logical_and(local[:,0]>=self.xlim[0], local[:,0]<=self.xlim[1])
			if self.ylim is not None:
				in_shape *= N.logical_and(local[:,1]>=self.ylim[0], local[:,1]<=self.ylim[1])
			if self.zlim is not None:
				in_shape *= N.logical_and(local[:,2]>=self.zlim[0], local[:,2]<=self.zlim[1])
			hitting = positive & in_shape
		else:
			hitting = positive
		
		select[N.logical_and(*hitting)] = 1
		one_hitting = N.logical_xor(*hitting)
		select[one_hitting] = N.nonzero(hitting.T[one_hitting,:])[1]

		return select
		
	def mesh(self, resolution=None):
		"""
		Represent the surface as a mesh in local coordinates.
		
		Arguments:
		resolution - in points per unit length (so the number of points 
			returned is O(A*resolution**2) for area A)
		
		Returns:
		x, y, z - each a 2D array holding in its (i,j) cell the x, y, and z
			coordinate (respectively) of point (i,j) in the mesh.
		"""
		if resolution == None:
			resolution = 40
			
		# Phi is declared for all points and then clipped to remove potential x or y truncation
		phi = N.linspace(0., 2.*N.pi, 4*resolution)
		# Theta is directly declared within correct z bounds.
		if self.truncated:
			thmin, thmax = N.arccos(self.zlim[0]*N.sqrt(self.c)), N.arccos(self.zlim[1]*N.sqrt(self.c))
		else:
			thmin, thmax = 0., N.pi
		theta = N.linspace(thmin, thmax, 2*resolution)
		TH, PH = N.meshgrid(theta, phi)
		xs = 1./N.sqrt(self.a)*N.sin(TH)*N.cos(PH)
		ys = 1./N.sqrt(self.b)*N.sin(TH)*N.sin(PH)
		zs = 1./N.sqrt(self.c)*N.cos(TH)
		
		'''
		# find xs beyond xlims
		xlow, xhigh = xs<self.xlim[0], xs>self.xlim[1]
		# remove them
		xbad = N.logical_or(xlow, xhigh)
		# add points to finish the shape on the truncating planes

		if xlow.any():
			n = N.sum(xlow)
			xs.append(N.ones(n)*self.xlim[0])
			zadd = N.linspace(0., 2.*N.pi, n)
			yadd = 1.-
			ys.append()
			zs.append(zadd)

		# find ys beyond xlims
		ylow, yhigh = ys<self.ylim[0], ys>self.ylim[1]
		ybad = N.logical_or(ylow, yhigh)

		bad = N.logical_and(xbad, ybad)
		xs, ys, zs = xs[~bad], ys[~bad], zs[~bad]
		'''		
		return xs, ys, zs
