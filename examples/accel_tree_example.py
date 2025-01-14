import numpy as N
from tracer.assembly import Assembly
from tracer.object import AssembledObject
from tracer.surface import Surface
from tracer.flat_surface import RectPlateGM
from tracer.boundary_shape import BoundaryBox
from tracer.optics_callables import LambertianReceiver
from tracer.CoIn_rendering.rendering import *
from tracer.tracer_engine import *
from tracer.ray_bundle import *
from tracer.sources import oblique_solar_rect_bundle
import logging
import time
import cProfile

'''
Example of usage of BoundaryBox with a scene for 3D acceleration
'''
#logging.basicConfig(level=logging.INFO)
# n**2 rectangular plates on a flat lambertian ground
n = 10
t0 = time.time()
# List of objects
objects = []
# Ground:
h, w = (n+1)*1., (n+1)*1.
geom = RectPlateGM(h, w)
opt = LambertianReceiver(0.6)
bound = BoundaryBox([[-w/2., -h/2., 0.], [w/2., h/2.,0.]])
surf = Surface(geometry=geom, optics=opt)
ground = AssembledObject(surf, bounds=bound)
objects.append(ground)

geom = RectPlateGM(h, w)
opt = LambertianReceiver(0.6)
bound = BoundaryBox([[-w/2., -h/2., 0.], [w/2., h/2.,0.]])
surf = Surface(geometry=geom, optics=opt)
ground2 = AssembledObject(surf, bounds=bound)
ground.set_location(N.array([0.,0.,-1]))
objects.append(ground2)

# Plates:
for k in range(n):
	for i in range(n):
		for j in range(n):		
			hp, wp = .8,.8#(i+1)/float(n), (j+1)/float(n)
			geom = RectPlateGM(wp, hp)
			opt = LambertianReceiver(0.9)
			bounds = BoundaryBox([[-wp/2., -hp/2., 0.], [wp/2., hp/2., 0.]]) # This is the boundary object with teh axis aligned bounding box defined with teh minimum point and maximum point.
			surf = Surface(geometry=geom, optics=opt)
			plate = AssembledObject(surf, bounds=bounds)
			plate.set_location(N.array([i+0.5-n/2., j+0.5-n/2., k+1.])) # here we set the location of the object afetr we gave it bounds, and thus they move with it (and are resized if there ws a rotation).
			objects.append(plate)

for i in range(3):	
	assembly = Assembly(objects=objects)
	engine = TracerEngine(assembly, loglevel=logging.INFO)
	print('Scene setup:', time.time()-t0,'s')

	assembly.reset_all_optics()

	t0 = time.time()
	source = oblique_solar_rect_bundle(num_rays=int(1000), center=N.vstack([0,0,k+2]), source_direction=N.hstack([0,0,-1]), rays_direction=N.hstack([0,0,-1]), x=w, y=h, ang_range=4.65e-3, flux=1000.)
	print('Source setup:', time.time()-t0,'s')

	def lightweight():
		t0 = time.time()
		engine.ray_tracer(source, accel='lightweight')
		surfs = engine._asm.get_surfaces()
		ener = 0
		for s in surfs:
			ener += N.sum(s.get_optics_manager().get_all_hits()[0])
		print ('ACCEL', time.time()-t0,'s', ener, 'W')
	lightweight()
	#cProfile.run('lightweight()', sort='time')
	#viewer = Renderer(engine)
	#viewer.show_rays(max_rays=100)

	assembly.reset_all_optics()
	def true():
		t0 = time.time()
		engine.ray_tracer(source, accel=True)
		surfs = engine._asm.get_surfaces()
		ener = 0
		for s in surfs:
			ener += N.sum(s.get_optics_manager().get_all_hits()[0])
		print ('True', time.time()-t0, ener, 'W')
	true()	
	#cProfile.run('true()', sort='time')

	assembly.reset_all_optics()
	def normal():
		t0 = time.time()
		engine.ray_tracer(source)#, accel=True)
		surfs = engine._asm.get_surfaces()
		ener = 0
		for s in surfs:
			ener += N.sum(s.get_optics_manager().get_all_hits()[0])
		print ('Normal', time.time()-t0, ener, 'W')
	normal()
	#cProfile.run('normal()', sort='time')

