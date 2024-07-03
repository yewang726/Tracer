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

'''
Example of usage of BoundaryBox with a scene for 3D acceleration
'''
logging.basicConfig(level=logging.INFO)
# n**2 rectangular plates on a flat lambertian ground
n = 100

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

# Plates:
for i in range(n):
	for j in range(n):
		hp, wp = (i+1)/float(n), (j+1)/float(n)
		geom = RectPlateGM(wp, hp)
		opt = LambertianReceiver(0.9)
		bounds = BoundaryBox([[-wp/2., -hp/2., 0.], [wp/2., hp/2., 0.]]) # This is the boundary object with teh axis aligned bounding box defined with teh minimum point and maximum point.
		surf = Surface(geometry=geom, optics=opt)
		plate = AssembledObject(surf, bounds=bounds)
		plate.set_location(N.array([i+0.5-n/2., j+0.5-n/2., 1.])) # here we set the location of the object afetr we gave it bounds, and thus they move with it (and are resized if there ws a rotation).
		objects.append(plate)

assembly = Assembly(objects=objects)
engine = TracerEngine(assembly)

source = oblique_solar_rect_bundle(num_rays=int(1e5), center=N.vstack([0,0,2]), source_direction=N.hstack([0,0,-1]), rays_direction=N.hstack([0,0,-1]), x=w, y=h, ang_range=4.65e-3, flux=1000.)

import time
t0 = time.time()
engine.ray_tracer(source, accel=True)
print (time.time()-t0)

viewer = Renderer(engine)
viewer.show_rays(max_rays=1000, bounding_boxes=True)

