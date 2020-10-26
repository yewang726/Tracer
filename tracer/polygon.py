import numpy as N
from geometry_manager import GeometryManager
from flat_surface import FiniteFlatGM


class FlatSimplePolygonGM(FiniteFlatGM):
	"""
	Finds if the intersections lie in a polygon. The polygon is defined using the sequential x and y coordinates of its vertices.
	This approach uses the boundary crossing method: https://en.wikipedia.org/wiki/Point_in_polygon
	"""
	def __init__(self, profile):
		'''
		Argument:
		profile: an [[xs],[ys]] array with the sequential coordinates of the simple p0lygon vertices IN CLOCKWISE DIRECTION.
		'''	
		self.profile = profile
		FiniteFlatGM.__init__(self)

	def find_intersections(self, frame, ray_bundle):
		ray_prms = FiniteFlatGM.find_intersections(self, frame, ray_bundle)
		profile = N.concatenate((self.profile, self.profile[:,0,None]), axis=1)
		inside = self.in_poly(points=self._local[:2], profile=profile)

		ray_prms[~inside] = N.inf

		del self._local
		return ray_prms

	def in_poly(self, points, profile):
		# Trim obvious cases: 
		# check local_coords gainst profile nodes
		x_pos = (points[0] <= profile[0,:,None]).T #(profile[0]-self._local[0])>=0.
		y_pos = (points[1] <= profile[1,:,None]).T #(profile[1]-self._local[1])>=0.
		# For segments fully beyond the point, just check the y coordinates of the segment: it is an intersection if y is in between the y coords of the segment.
		beyond_x = N.logical_and(x_pos[:,:-1], x_pos[:,1:])
		across_y = N.logical_xor(y_pos[:,:-1], y_pos[:,1:])
		inters = N.logical_and(beyond_x, across_y)

		# For segments with x on both sides of the point: find the intersection parameter and the positive params are valid
		across_x = N.logical_xor(x_pos[:,:-1], x_pos[:,1:])
		doubt = N.array(N.nonzero(N.logical_and(across_x, across_y)))

		# If there is a doubt for an intersection
		for i in range(doubt.shape[1]):
			inters[doubt[0,i],doubt[1,i]] = self.intersect(points[:2,doubt[0,i]], profile[:,doubt[1,i]:doubt[1,i]+2])
		inside = N.array(N.sum(inters, axis=1)%2, dtype=bool)

		return inside

	def intersect(self, xy, segment):
		x0, y0 = segment[:,0]
		x1, y1 = segment[:,1]
		a = (y1-y0)/(x1-x0)
		b = y0-a*x0
		x_inter = (xy[1]-b)/a
		inter = x_inter>=xy[0]
		return inter

	def mesh(self, resolution):
		if resolution==None:
			resolution=2
		alpha, beta = N.meshgrid(
			N.linspace(0, 1, resolution), # parameter along two edges
			N.linspace(0, 1, resolution)) # parameter between points on edges
		# Mesh using ear clipping
		mesh = []
		profile = N.concatenate((self.profile, self.profile[:,0,None]), axis=1)

		while (profile.shape[1]-1)>3:
			for i in range(profile.shape[1]-1):
				triangle = profile[:,i:i+3]
				# profile defined clockwise rotation so we can find reflex angles:
				reflex = N.linalg.det([profile[:,i+1]-profile[:,i], profile[:,i+2]-profile[:,i+1]])>0
				ear = False
				if ~reflex:
					rest = N.concatenate((profile[:,:i], profile[:,i+3:]), axis=1)
					inside = self.in_poly(points=rest, profile=triangle)

					if inside.any():	
						# Not an ear
						break
					xs = profile[0,i:i+3]
					ys = profile[1,i:i+3]
					verts = N.array([xs[1:]-xs[0], ys[1:]-ys[0], N.zeros(2)])
					
					x, y, z = alpha*verts[:,1,None,None]*(1 - beta) + \
						alpha*verts[:,0,None,None]*beta
					mesh.append(x+xs[0])
					mesh.append(y+ys[0])
					mesh.append(z)
					profile = N.delete(profile, i+1, axis=1)
					ear = True
					break
				if ear:
					break

		xs = profile[0,:-1]
		ys = profile[1,:-1]
		verts = N.array([xs[1:]-xs[0], ys[1:]-ys[0], N.zeros(2)])
		alpha, beta = N.meshgrid(
			N.linspace(0, 1, resolution), # parameter along two edges
			N.linspace(0, 1, resolution)) # parameter between points on edges
		
		x, y, z = alpha*verts[:,1,None,None]*(1 - beta) + \
			alpha*verts[:,0,None,None]*beta
		mesh.append(x+xs[0])
		mesh.append(y+ys[0])
		mesh.append(z)
		return mesh
		
