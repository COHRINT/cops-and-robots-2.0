#!/usr/bin/env python

""" Summary:
	Creates a map object from an inputed 'map.yaml' file (in models dir)
		with softmax LIKELIHOODs
	Map includes:
		1) General info: name, bounds.max_x_y, bounds.min_x_y, origin
		2) Object hash: 'self.objects', each member is a Map_Object
		3) Rooms : self.rooms['room_name']['lower_l' OR 'upper_r' OR 'likelihood']
			access the room's lower left coordinate and upper right coord
	Map_Object includes:
		name, color, centroid[x, y], major axis, minor axis,
		orientation from the object's major axis to the map's positive x axis,
		shape (available shapes: oval and rectangle),
		softmax likelihood
"""

__author__ = "LT"
__copyright__ = "Copyright 2017, COHRINT"
__credits__ = ["Luke Babier", "Ian Loefgren", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "2.0.0" # Likelihoods added
__maintainer__ = "LT"
__email__ = "luba6098@colorado.edu"
__status__ = "Development"

import yaml
import os 			# path capabilities
from collections import OrderedDict
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import patches
from softmaxModels import *

class Map(object):
	""" Map Object from map.yaml file (located in models dir)

	Map includes:
		1) General info: self.name (str),
						self.size[max_x, max_y] (float list),
						self.origin[x, y] (float list)
		2) Object hash: 'self.objects', each member is a Map_Object

	Parameters
	----------
	yaml_file : map1.yaml, map2.yaml, etc

	"""

	def __init__(self, yaml_file):

		# load yaml file as a dictionary
		cfg = self._find_yaml(yaml_file)
		# cfg = yaml.load(open('../models/' + yaml_file, 'r'));

		if cfg is not None:

			# Get map's general info
			self.name = cfg['info']['name']
			self.bounds = [cfg['info']['bounds']['min_x'],cfg['info']['bounds']['min_y'],
							cfg['info']['bounds']['max_x'],cfg['info']['bounds']['max_y']]
			self.origin = [cfg['info']['origin']['x_coord'], cfg['info']['origin']['y_coord']]

			# Add room boundaries to the map
			self.rooms = {}
			lower_l = list()
			upper_r = list()
			for room in cfg['info']['rooms']:
				lower_l = (cfg['info']['rooms'][room]['min_x'], cfg['info']['rooms'][room]['min_y'])
				upper_r = (cfg['info']['rooms'][room]['max_x'], cfg['info']['rooms'][room]['max_y'])
				self.rooms[room] = {}
				self.rooms[room]['lower_l'] = lower_l
				self.rooms[room]['upper_r'] = upper_r
				length = upper_r[0] - lower_l[0]
				width = upper_r[1] - lower_l[1]
				cent = [lower_l[0] + length/2,lower_l[1]+width/2]
				self.rooms[room]['softmax'] = Softmax()
				self.rooms[room]['softmax'].buildOrientedRecModel(cent, 0.0, length, width,steepness=3)
				for i in range(0,len(self.rooms[room]['softmax'].weights)):
					self.rooms[room]['softmax'].weights[i] = [0,0,self.rooms[room]['softmax'].weights[i][0],self.rooms[room]['softmax'].weights[i][1]];
				self.rooms[room]['objects'] = cfg['info']['rooms'][room]['objects']

			# Store map's objects in self.objects hash
			self.objects = {}
			for item in cfg:
				if item != 'info':	# if not general info => object on map
					map_obj = Map_Object(cfg[item]['name'],
										cfg[item]['color'],
										[cfg[item]['centroid_x'], cfg[item]['centroid_y']],
										cfg[item]['length'],
										cfg[item]['width'],
										cfg[item]['orientation'],
										cfg[item]['shape']
										)
					self.objects[map_obj.name] = map_obj


	# Searches the yaml_dir for the given yaml_file
	# Returns a python dictionary if successful
	# Returns 'None' for failure
	def _find_yaml(self, yaml_file):
		yaml_dir = 'models'

		try:
			# navigate to yaml_dir
			cfg_file = os.path.dirname(__file__) \
				+ '/../' + yaml_dir + '/' + yaml_file
			# return dictionary of yaml file
			with open(cfg_file, 'r') as file:
				 return yaml.load(file)
		except IOError as ioerr:
			print str(ioerr)
			return None

	def make_occupancy_grid(self,res):
		"""
		Occupancy grid creation from a yaml file.
			- Uses the map associated with the instance of the Map class.
			- Saves occupancy grid as a png with only black and white coloring.

		Inputs
		-------
			- res - desired resolution for occupancy grid in [m/px]

		Outputs
		-------
			- returns nothing
			- saves occupancy grid
		"""
		#<>TODO: refactor into a sperate module?
		# create matplotlib figure to plot map
		fig = Figure()
		canvas = FigureCanvas(fig)
		ax = fig.add_subplot(111)

		# get dpi of figure
		dpi = float(fig.get_dpi())
		print("DPI: {}".format(dpi))
		# calculate required size in pixels of occupancy grid
		x_size_px = (self.bounds[2]-self.bounds[0]) / res
		y_size_px = (self.bounds[3]-self.bounds[1]) / res
		# calculate required size in inches
		x_size_in = x_size_px / dpi
		y_size_in = y_size_px / dpi
		#specify size of figure in inches
		fig.set_size_inches(x_size_in,y_size_in)

		# add patches for all objects in yaml file
		for obj in self.objects:
		    cent = self.objects[obj].centroid;
		    x = self.objects[obj].length;
		    y = self.objects[obj].width;
		    theta = self.objects[obj].orient;
		    col = self.objects[obj].color
		    if(self.objects[obj].shape == 'oval'):
		        tmp = patches.Ellipse((cent[0] - x/2,cent[1]-y/2),width = x, height=y,angle=theta,fc='black',ec='black');
		    else:
				# skip plotting posters as they aren't actually protruding into the space
				if 'poster' in obj:
					continue
				else:
					# find the location of the lower left corner of the object for plotting
					length = x
					width = y
					theta1 = theta*math.pi/180;
					h = math.sqrt((width/2)*(width/2) + (length/2)*(length/2));
					theta2 = math.asin((width/2)/h);
					s1 = h*math.sin(theta1+theta2);
					s2 = h*math.cos(theta1+theta2)
					xL = cent[0]-s2
					yL = cent[1]-s1

					tmp = patches.Rectangle((xL,yL),width = x, height=y,angle=theta,fc='black',ec='black');

		    ax.add_patch(tmp)

		# save the matplotlib figure
		ax.set_xlim(self.bounds[0],self.bounds[2])
		ax.set_ylim(self.bounds[1],self.bounds[3])
		ax.axis('image')
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		print('about to save plot')
		canvas.print_figure(os.path.dirname(__file__) + '/%s_occupancy.png'%self.name.lower(),bbox_inches='tight',pad_inches=0)

	def make_occupancy_yaml(self,res,occ_thresh=0.2,free_thresh=0.65):
		yaml_content = {'image': self.name.lower()+'_occupancy.png',
						'resolution': res,
						'origin': [self.bounds[0],self.bounds[1],0.0],
						'occupied_thresh': occ_thresh,
						'free_thresh': free_thresh,
						'negate': 0}
		
		file_name = os.path.dirname(__file__) + '/' + self.name.lower() + '_occupancy.yaml'

		with open(file_name,'w') as yaml_file:
			yaml.safe_dump(yaml_content,yaml_file,allow_unicode=False)

class Map_Object(object):
	"""
	Objects like chairs, bookcases, etc to be included in the map object
	-Derived from a map.yaml file (in models dir)

	Map_Object includes:
		name (str), color (str), centroid[x, y] (float list), major axis (float),
		minor axis (float),
		orientation from the object's major axis to the map's positive x axis (float)
		shape (str) (available shapes: oval and rectangle)
		softmax likelihood

	Parameters
	----------
	name: str
		Name of obj
	color: str
		Color of obj
	centroid : 2x1 list
		Centroid location [x, y] [m]
	length: float
		x axis length of obj [m] (along direction object is facing)
	width: float
		y axis width of obj [m] (normal to direction object is facing)
	orient : float
		Radians between obj's major axis and the map's pos-x axis
	shape : str
		Values accepted: 'rectangle' or 'oval'
	"""
	def __init__(self,
				name='wall',
				color='darkblue',
				centroid=[0.0,0.0],
				length = 0.0,
				width = 0.0,
				orient=0.0,
				shape = 'rectangle'
				):
		self.name = name
		self.color = color
		self.centroid = centroid
		self.length = length
		self.width = width
		self.orient = orient

		self._pick_shape(shape)

		# create the objects likelihood
		self.softmax = Softmax()
		self.get_likelihood()

	def get_likelihood(self):
		"""
		Create and store corresponding likelihood.
		Approximate all shapes as rectangles
		"""
		self.softmax.buildOrientedRecModel(self.centroid,
			self.orient, self.length, self.width, steepness=3)
		for i in range(0,len(self.softmax.weights)):
			self.softmax.weights[i] = [0,0,self.softmax.weights[i][0],self.softmax.weights[i][1]];

	# Selects the shape of the obj
	# Default = 'rectangle' --- 'oval' also accepted
	def _pick_shape(self, shape):

		if shape == 'oval':
			self.shape = 'oval'
		else:
			self.shape = 'rectangle'

def test_map_obj():
	map1 = Map('map2.yaml')

	if hasattr(map1, 'name'): # check if init was successful
		print map1.name
		print map1.objects['dining table'].color
		print map1.rooms['dining room']['lower_l']
		print map1.rooms['kitchen']['upper_r']

	else:
		print 'fail'

def test_likelihood():
	map2 = Map('map2.yaml')
	if hasattr(map2, 'name'):
		for obj in map2.objects:
			print obj
		print("Dining table:")
		print (map2.objects['dining table'].softmax.weights)
		print (map2.objects['dining table'].softmax.bias)
		print (map2.objects['dining table'].softmax.size)
		print("Mars Poster:")
		print(map2.objects['mars poster'].softmax.weights)
		print("Dining Room: ")
		print(map2.rooms['dining room']['softmax'].weights)
	else:
		print("Failed to initialize Map Object.")
		raise

def test_occ_grid_gen():
	map_= Map('map2.yaml')
	res = 0.01
	map_.make_occupancy_grid(res)
	map_.make_occupancy_yaml(res)

if __name__ == "__main__":
	#test_map_obj()
	# test_likelihood()
	test_occ_grid_gen()
