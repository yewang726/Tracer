# Define some basic surfaces for use with the ray tracer. From this minimal 
# hierarchy other surfaces should be derived to implement actual geometric
# operations.
#
# References:
# [1] John J. Craig, Introduction to Robotics, 3rd ed., 2005. 

import numpy as N
from has_frame import HasFrame


class Surface(HasFrame):
	"""
	Defines the base of surfaces that interact with rays.
	"""
	def __init__(self, geometry, optics, location=None, rotation=None, fixed_color=False):
		"""
		Arguments:
		geometry - a GeometryManager object responsible for finding ray 
			intersections with the surface.
		optics - a callable that gets the geometry manageri, bundle and
			selector, and returns the outgoing ray bundle generated by the
			geometry and bundle.
		location, rotation - passed directly to the HasFrame constructor.
		fixed_color - For rendering purposed. If a tuple of normalised RGB or RGBa is given, 
			the geometry will be of that color in the rendering.
		"""
		HasFrame.__init__(self, location, rotation)
		self._geom = geometry
		self._opt = optics
		self._fixed_color = fixed_color
		if fixed_color:
			self._fixed_color = fixed_color[:3]	
			if len(fixed_color) == 4:
				self._transparency = fixed_color[-1]
			else: 
				self._transparency = 0

		
	def get_optics_manager(self):
		"""
		Returns the optics-manager callable. May be useful for introspection.
		Note that it is a read-only attribute.
		"""
		return self._opt
	
	def get_geometry_manager(self):
		"""
		Returns the geometry-manager instance. May be useful for introspection.
		Note that it is a read-only attribute.
		"""
		return self._geom
	
	def register_incoming(self, ray_bundle):
		"""
		Records the incoming ray bundle, and uses the geometry manager to
		return the parametric positions of intersection with the surface along
		the ray.
		
		Arguments:
		ray_bundle - a RayBundle object with at-least its vertices and
			directions specified.
		
		Returns
		A 1D array with the parametric position of intersection along each of
			the rays. Rays that missed the surface return +infinity.
		"""	  
		self._current_bundle = ray_bundle
		return self._geom.find_intersections(self._temp_frame, ray_bundle)
	
	def select_rays(self, idxs):
		"""
		Informs the geometry manager that only the specified rays are to be
		used henceforth.
		
		Arguments:
		idxs - an array with indices into the last registered ray bundle,
			marking rays that will be used.
		"""
		self._selected = idxs
		self._geom.select_rays(idxs)
	
	def get_outgoing(self):
		"""
		Generates a new ray bundle, which is the reflections/refractions of the
		user-selected rays out of the incoming ray-bundle that was previously
		registered.
		
		Returns: 
		a RayBundle object with the new bundle, with vertices on the surface
			and directions according to optics laws.
		"""
		return self._opt(self._geom, self._current_bundle, self._selected)
	
	def done(self):
		"""
		When this is called, the surface will no longer be queried on the
		results of the latest trace iteration, so it can discard internal
		data to relieve memory pressure.
		"""
		if hasattr(self, '_current_bundle'):
			del self._current_bundle
		self._geom.done()
		
	def global_to_local(self, points):
		"""
		Transform a set of points in the global coordinates back into the frame
		used during tracing.
		
		Arguments:
		points - a 3 x n array for n 3D points
		
		returns:
		local - a 3 x n array with the respective points in local coordinates.
		"""
		proj = N.round(N.linalg.inv(self._temp_frame), decimals=9)
		return N.dot(proj, N.vstack((points, N.ones(points.shape[1]))))
	
	def mesh(self, resolution):
		"""
		Represent the surface as a mesh in global coordinates.
		
		Arguments:
		resolution - in points per unit length (so the number of points 
			returned is O(A*resolution**2) for area A)
		
		Returns:
		x, y, z - each a 2D array holding in its (i,j) cell the x, y, and z
			coordinate (respectively) of point (i,j) in the mesh.
		"""
		# The geometry manager has the local-coordinates mesh.
		x, y, z = self._geom.mesh(resolution)
		local = N.array((x, y, z, N.ones_like(x)))
		glob = N.tensordot(self._temp_frame, local, axes=([1], [0]))
		return glob[:3]

	def get_scene_graph(self, resolution, fluxmap, trans, vmin, vmax):
		"""
		Any object that provides a nice QuadMesh from the previous code should be able to render in Coin3D with with the following...
		"""
		from pivy import coin
		import matplotlib.cm as cm
		from matplotlib import colors

		n0 = self.get_scene_graph_transform()
		o = self.get_optics_manager()

		if self._fixed_color:
			mat = coin.SoMaterial()
			mat.diffuseColor = self._fixed_color
			mat.specularColor = self._fixed_color
			mat.transparency = (self._transparency)
			n0.addChild(mat)
			fluxmap = False
		else:
			if o.__class__.__name__[-10:] == 'Reflective':
				mat = coin.SoMaterial()
				mat.diffuseColor = (.5,.5,.5)
				mat.specularColor = (.6,.6,.6)
				mat.shininess = o._abs
				n0.addChild(mat)
				fluxmap = False

			elif o.__class__.__name__ == 'PeriodicBoundary':
				mat = coin.SoMaterial()
				mat.ambientColor = (.0,.5,.5)
				mat.transparency = (0.8)
				n0.addChild(mat)
				fluxmap = False

			elif fluxmap != None:
				if hasattr(o,'get_all_hits'):
					hitdata = o.get_all_hits()
					xyz = self.global_to_local(hitdata[1])[:3]
					# plot the histogram into the scenegraph
					g = self.get_geometry_manager()
					if hasattr(g, 'get_fluxmap'):
						flux = g.get_fluxmap(hitdata[0], xyz, resolution)
						if not(hasattr(flux[0],'__len__')):
							flux = [flux]
					else:
						fluxmap = False
			else: 
				mat = coin.SoMaterial()
				mat.diffuseColor = (0.2,0.2,0.2)
				mat.specularColor = (0.2,0.2,0.2)
				n0.addChild(mat)
				fluxmap = False

		meshes = self._geom.get_scene_graph(resolution)
		for m in xrange(len(meshes)/3):
			n = coin.SoSeparator()

			X,Y,Z = meshes[3*m:3*m+3]
			nr,nc = X.shape
			A = [(X.flat[i],Y.flat[i],Z.flat[i]) for i in range(len(X.flat))]
			coor = coin.SoCoordinate3()
			coor.point.setValues(0, len(A), A)
			n.addChild(coor)

			qm = coin.SoQuadMesh()
			qm.verticesPerRow = nc
			qm.verticesPerColumn = nr
			n.addChild(qm)

			sh = coin.SoShapeHints()
			sh.shapeType = coin.SoShapeHintsElement.UNKNOWN_SHAPE_TYPE
			sh.vertexOrdering = coin.SoShapeHintsElement.COUNTERCLOCKWISE
			sh.faceType = coin.SoShapeHintsElement.UNKNOWN_FACE_TYPE
			n.addChild(sh)

			if fluxmap:

				# It works using n0 instead of n here but I have absolutely not clue why.
				norm = colors.Normalize(vmin=vmin, vmax=vmax)
				M = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
				colormap = M.to_rgba(flux[m])
				mat = coin.SoMaterial()
				mat.ambientColor = (1,1,1)
				mat.diffuseColor.setValues(0, colormap.shape[0], colormap)
				if trans==True:
					mat.transparency.setValues(0,colormap.shape[0], 1.-flux[m]/N.amax(flux[m]))
				n0.addChild(mat)

				mymatbind = coin.SoMaterialBinding()
				mymatbind.value = coin.SoMaterialBinding.PER_FACE
				n0.addChild(mymatbind)
			
			n0.addChild(n)
			
		return n0


