import os, math
import numpy as np
import bpy

def angle2rad( theta, unit ):
	return math.radians(theta) if unit=='DEG' else theta

def rad2angle( theta, unit ):
	return math.degrees(theta) if unit=='DEG' else theta

def expand_rgb( rgb, alpha=True ):
	if np.isscalar(rgb):
		rgb = 3*(rgb,)
	if alpha and len(rgb)==3:
		rgb = tuple(rgb) + (1.0,)
	return rgb
		
def cleanup( skiptype=[ 'CAMERA', 'LIGHT' ] ):
	# add argument 'skipname' that is a list of name patterns to not delete, e.g., cube*

	for x in bpy.data.objects:
		if not x.type in skiptype:
			bpy.data.objects.remove( x )

	for block in [ bpy.data.curves, bpy.data.meshes, bpy.data.materials, bpy.data.textures, bpy.data.images ]:
		for x in block:
			if x.users == 0:
				block.remove( x )

class Model:

	def __init__( self, filename=None ):
		if not filename is None:
			self.load(filename)

	def load( self, filename ):
		bpy.ops.object.select_all(action='DESELECT')
		bpy.ops.import_scene.obj(filepath=filename)
		self.model = bpy.context.selected_objects[0]
		bpy.context.view_layer.objects.active = self.model

	def centre( self, pos=(0,0,0) ):
		bpy.context.view_layer.objects.active = self.model
		bpy.ops.object.origin_set( type='ORIGIN_GEOMETRY', center='BOUNDS' )
		self.model.location = pos

	def scale( self, size=1 ):
		self.model.scale = 3*(size/max(self.model.dimensions),)

class Transform:

	def __init__( self, child, unit='DEG' ):
		bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0.0,0.0,0.0), scale=(1.0,1.0,1.0))
		self.transform = bpy.context.object
		child.parent = self.transform
		self.transform.name = child.name + 'Transform'
		self.transform.rotation_mode = 'XYZ'
		self.__unit = unit.upper()
		self.dir = 0.0, 0.0
	
	@property
	def azimuth( self ):
		return rad2angle( self.transform.rotation_euler[2], self.__unit )
	
	@azimuth.setter
	def azimuth( self, az ):
		self.transform.rotation_euler[2] = angle2rad( az, self.__unit )

	@property
	def elevation( self ):
		return rad2angle( -self.transform.rotation_euler[1], self.__unit )
	
	@elevation.setter
	def elevation( self, el ):
		self.transform.rotation_euler[1] = angle2rad( -el, self.__unit )

	@property
	def dir( self ):
		return self.azimuth, self.elevation
	
	@dir.setter
	def dir( self, d ):
		self.azimuth = d[0]
		self.elevation = d[1]

	@property
	def location( self ):
		return self.transform.location

	@location.setter
	def location( self, v ):
		self.transform.location = v
	
class Light( Transform ):

	def __init__( self, light=bpy.data.objects['Light'], world=bpy.data.worlds['World'], unit='DEG' ):
		
		super().__init__( light, unit )
		
		self.light = light
		self.world = world

		self.light.data.type='SUN'
		self.light.location = 5.0, 0.0, 0.0
		self.light.rotation_mode = 'XYZ'
		self.light.rotation_euler = 0.0, math.radians(90.0), 0.0
		self.point = 1.0, 1000.0
		self.ambient = 1.0, 100.0
		self.angle = 1.0

	@property
	def point( self ):
		return self.light.data.color[:], self.light.data.energy

	@point.setter
	def point( self, v ):
		rgb, mag = v
		if not rgb is None:
			self.light.data.color = expand_rgb( rgb, alpha=False )
		if not mag is None:
			self.light.data.energy = mag

	@property
	def ambient( self ):
		b = self.world.node_tree.nodes['Background'].inputs
		return b[0].default_value[0:3], b[1].default_value

	@ambient.setter
	def ambient( self, v ):
		rgb, mag = v
		b = self.world.node_tree.nodes['Background'].inputs
		if not rgb is None:
			b[0].default_value = expand_rgb( rgb )
		if not mag is None:
			b[1].default_value = mag

	@property
	def angle( self ):
		return rad2angle( self.light.data.angle, unit='DEG' )

	@angle.setter
	def angle( self, v ):
		self.light.data.angle = angle2rad( v, unit='DEG' )

class Camera( Transform ):

	def __init__( self, camera=bpy.data.objects['Camera'], distance=1.0, fov=40.0, unit='DEG', bg=None ):

		super().__init__( camera, unit )

		self.camera = camera
		self.__distance = distance
		self.__unit = unit.upper()
		self.camera.location = ( distance, 0.0, 0.0 )
		self.camera.rotation_mode = 'ZYX'
		self.camera.rotation_euler = (0.0,math.radians(90.0),math.radians(90.0))
		self.camera.data.lens_unit = 'FOV'
		self.fov = fov

		# create background plane
		if not bg is None:
			bpy.ops.mesh.primitive_plane_add( location = -5 * self.camera.location, rotation = ( 0, math.pi/2, 0 ), size = 10 * max( self.camera.location ) )
			plane = bpy.context.active_object
			plane.name = 'Background'
			m = bpy.data.materials.new( 'BackgroundMaterial' )
			m.diffuse_color = expand_rgb( bg )
			m.specular_intensity = 0.0
			plane.data.materials.append( m )
			plane.parent = self.camera.parent

	@property
	def fov( self ):
		return rad2angle( self.camera.data.angle, self.__unit )
	
	@fov.setter
	def fov( self, d ):
		self.camera.data.angle = angle2rad( d, self.__unit )

class Intrinsic:

	def __init__(self, hdr=True):

		# use nodes
		bpy.context.scene.use_nodes = True

		# delete existing nodes
		nodes, _ = Intrinsic.parts()
		for n in nodes:
			nodes.remove(n)

		# create render layers node
		render_layers = nodes.new('CompositorNodeRLayers')
		render_layers.name = 'MainRenderLayers'

		# set default state
		self.hdr = hdr
		self.reflectance = True
		self.normals = True
		self.depth = False
		self.objectid = False

	# reflectance property

	@property
	def reflectance(self):
		return bpy.context.scene.view_layers['View Layer'].use_pass_diffuse_color

	@reflectance.setter
	def reflectance(self,b):

		Intrinsic.delnodes( [ 'RefAlpha', 'RefFile' ] )
		bpy.context.scene.view_layers['View Layer'].use_pass_diffuse_color = b

		if b:
			nodes, links = Intrinsic.parts()
			main = nodes[ 'MainRenderLayers' ]

			ref_alpha = nodes.new(type='CompositorNodeSetAlpha')
			ref_alpha.name = 'RefAlpha'
			links.new(main.outputs['DiffCol'], ref_alpha.inputs['Image'])
			links.new(main.outputs['Alpha'], ref_alpha.inputs['Alpha'])

			f = self.filenode( nodes, 'RefFile' )
			links.new(ref_alpha.outputs['Image'], f.inputs[0])

	# normal property

	@property
	def normals(self):
		return bpy.context.scene.view_layers['View Layer'].use_pass_normal

	@normals.setter
	def normals(self, b):

		Intrinsic.delnodes( [ 'NrmScale', 'NrmBias', 'NrmFile' ] )
		bpy.context.scene.view_layers['View Layer'].use_pass_normal = b

		if b:
			nodes, links = Intrinsic.parts()
			main = nodes[ 'MainRenderLayers' ]

			scale_node = nodes.new(type='CompositorNodeMixRGB')
			scale_node.name = 'NrmScale'
			scale_node.blend_type = 'MULTIPLY'
			scale_node.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
			links.new(main.outputs['Normal'], scale_node.inputs[1])

			bias_node = nodes.new(type='CompositorNodeMixRGB')
			bias_node.name = 'NrmBias'
			bias_node.blend_type = 'ADD'
			bias_node.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
			links.new(scale_node.outputs[0], bias_node.inputs[1])

			f = self.filenode( nodes, name='NrmFile' )
			links.new(bias_node.outputs[0], f.inputs[0])

	# depth property

	@property
	def depth(self):
		return bpy.context.scene.view_layers['View Layer'].use_pass_z

	@depth.setter
	def depth(self, b):

		Intrinsic.delnodes( [ 'DepthFile' ] )
		bpy.context.scene.view_layers['View Layer'].use_pass_z = b

		if b:
			nodes, links = Intrinsic.parts()
			main = nodes[ 'MainRenderLayers' ]
			f = self.filenode( nodes, name='DepthFile' )
			links.new(main.outputs['Depth'], f.inputs[0])

	# object id property

	@property
	def objectid(self):
		return bpy.context.scene.view_layers['View Layer'].use_pass_object_index

	@objectid.setter
	def objectid(self, b):

		Intrinsic.delnodes( [ 'ObjectIDFile' ] )
		bpy.context.scene.view_layers['View Layer'].use_pass_object_index = b

		if b:
			nodes, links = Intrinsic.parts()
			main = nodes[ 'MainRenderLayers' ]
			f = self.filenode( nodes, name='ObjectIDFile' )
			links.new(main.outputs['IndexOB'], f.inputs[0])

	# filename property

	@property
	def filename(self):
		return bpy.context.scene.render.filepath

	@filename.setter
	def filename(self, fname):
		bpy.context.scene.render.filepath = fname
		nodes, _ = Intrinsic.parts()
		if self.reflectance:
			nodes[ 'RefFile' ].file_slots[0].path = fname + '_ref'
		if self.normals:
			nodes[ 'NrmFile' ].file_slots[0].path = fname + '_nrm'
		if self.depth:
			nodes[ 'DepthFile' ].file_slots[0].path = fname + '_z'
		if self.objectid:
			nodes[ 'ObjectIDFile' ].file_slots[0].path = fname + '_id'

	# utility functions

	@staticmethod
	def delnodes( names ):
		nodes, _ = Intrinsic.parts()
		for name in names:
			n = nodes.get( name )
			if not n == None:
				nodes.remove( n )

	@staticmethod
	def parts():
		return bpy.context.scene.node_tree.nodes, bpy.context.scene.node_tree.links

	@staticmethod
	def unpad( node ):
		fname_pad = os.path.join( node.base_path, node.file_slots[0].path + '0001' + '.' + node.format.file_format.lower() )
		fname     = os.path.join( node.base_path, node.file_slots[0].path          + '.' + node.format.file_format.lower() )
		os.rename( fname_pad, fname )

	def filenode( self, nodes, name='File' ):
		f = nodes.new(type='CompositorNodeOutputFile')
		f.name = name
		f.label = name
		f.base_path = ''
		f.file_slots[0].use_node_format = True
		self.setformat( f.format )
		return f

	def setformat( self, f ):
		f.file_format = 'HDR' if self.hdr else 'PNG'
		f.color_depth = '32' if self.hdr else '8'
		f.color_mode = 'RGB'
		f.compression = 0

	# render function

	def render(self):

		bpy.data.scenes[ 'Scene' ].render.engine = 'CYCLES'
		bpy.context.preferences.addons[ 'cycles' ].preferences.compute_device_type = 'CUDA'
		bpy.context.scene.cycles.device = 'GPU'
		self.setformat( bpy.context.scene.render.image_settings )
		bpy.data.scenes[ 'Scene' ].cycles.filter_width = 0

		bpy.ops.render.render(write_still=True)
		nodes, _ = Intrinsic.parts()
		if self.reflectance:
			self.unpad( nodes[ 'RefFile' ] )
		if self.normals:
			self.unpad( nodes[ 'NrmFile' ] )
		if self.depth:
			self.unpad( nodes[ 'DepthFile' ] )
		if self.objectid:
			self.unpad( nodes[ 'ObjectIDFile' ] )

def voronoi( mat, scale=5.0, randomness=1.0 ):

	mat.use_nodes = True
	nodes = mat.node_tree.nodes
	links = mat.node_tree.links
	
	[ nodes.remove( x ) for x in nodes ]
	
	matout = nodes.new(type='ShaderNodeOutputMaterial')
	diff = nodes.new(type='ShaderNodeBsdfDiffuse')
	links.new( diff.outputs['BSDF'], matout.inputs['Surface'] )
	
	bw = nodes.new(type='ShaderNodeRGBToBW')
	links.new( bw.outputs['Val'], diff.inputs['Color'] )
	
	vor = nodes.new(type='ShaderNodeTexVoronoi')
	vor.inputs[2].default_value = scale
	vor.inputs[5].default_value = randomness
	links.new( vor.outputs['Color'], bw.inputs['Color'] )
	
	tex = nodes.new(type='ShaderNodeTexCoord')
	bpy.ops.object.empty_add(type='PLAIN_AXES')
	empty = bpy.context.active_object
	empty.name = mat.name + 'Empty'
	tex.object = empty
	links.new( tex.outputs['Object'], vor.inputs['Vector'] )

def noise( mat, scale=2.0, detail=2.0, roughness=0.5, distortion=0.0, maprange=(0.3,1.0) ):

	mat.use_nodes = True
	nodes = mat.node_tree.nodes
	links = mat.node_tree.links
	
	[ nodes.remove( x ) for x in nodes ]
	
	matout = nodes.new(type='ShaderNodeOutputMaterial')
	diff = nodes.new(type='ShaderNodeBsdfDiffuse')
	links.new( diff.outputs['BSDF'], matout.inputs['Surface'] )
	
	mapr = nodes.new(type='ShaderNodeMapRange')
	mapr.inputs[1].default_value = maprange[0]
	mapr.inputs[2].default_value = maprange[1]
	links.new( diff.inputs['Color'], mapr.outputs['Result'] )

	bw = nodes.new(type='ShaderNodeRGBToBW')
	links.new( bw.outputs['Val'], mapr.inputs['Value'] )
	
	noise = nodes.new(type='ShaderNodeTexNoise')
	noise.inputs[2].default_value = scale
	noise.inputs[3].default_value = detail
	noise.inputs[4].default_value = roughness
	noise.inputs[5].default_value = distortion
	links.new( noise.outputs['Color'], bw.inputs['Color'] )

	tex = nodes.new(type='ShaderNodeTexCoord')
	bpy.ops.object.empty_add(type='PLAIN_AXES')
	empty = bpy.context.active_object
	empty.name = mat.name + 'Empty'
	tex.object = empty
	links.new( tex.outputs['Object'], noise.inputs['Vector'] )

