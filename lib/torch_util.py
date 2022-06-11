import os, random, glob, numpy as np, torch, imageio

def device( request=None, verbose=True ):
	if request=='cpu':
		return 'cpu'
	if torch.cuda.is_available():
		d = 'cuda:0'
		if verbose:
			print( 'running on %s (%s)' % ( d, torch.cuda.get_device_name( d ) ) )
	else:
		d = 'cpu'
		if verbose:
			print( 'running on cpu' )
	return d

def load( model, device, filename='model.pt', require=True ):
	if len(filename.strip())==0:
		print( 'no model file specified for loading' )
		return None
	if os.path.exists( filename ):
		print( 'loading model file %s' % filename )
		model.load_state_dict( torch.load( filename, map_location=device ) )
	else:
		msg = 'unable to find model file %s' % filename
		if require:
			raise FileNotFoundError( msg )
		print( msg )

def save( model, filename='model.pt' ):
	if len(filename.strip())==0:
		print( 'no model file specified for saving' )
		return None
	print( 'saving model file %s' % filename )
	torch.save( model.state_dict(), filename )

def hdrread( fname, ftag=None ):
	if not ftag is None:
		fname = fname[:-4] + ftag + fname[-4:]
	im = np.array( imageio.imread(fname, format='HDR-FI'), dtype=np.float32 )
	# im = cv2.imread( fname, flags=cv2.IMREAD_ANYDEPTH )
	# im = im.astype( np.float32 )
	# im = np.flip( im, axis=2 )
	return im

# generator that returns intrinsic image data in random order
def gen( directory='/hdd/geometric', batchsize=10, tensor=True, device=None ):

	def flip( im ):
		imp = im[:,:,0]
		imq = imp[...,np.newaxis]
		return np.moveaxis(imq,(0,1,2),(1,2,0))

	# get list of files in random order
	filt = os.path.join( directory, '*[0-9].hdr' )
	flist = glob.glob( filt )
	random.shuffle( flist )

	nbatch = len(flist) // batchsize
	flist = flist[:(nbatch*batchsize)]

	# load one file to get image size
	im = hdrread( flist[0] )
	imshape = im.shape
	batchshape = ( batchsize, 1, imshape[0], imshape[1] )

	for i in range(0,len(flist),batchsize):

		lumbatch = np.empty( batchshape )
		refbatch = np.empty( batchshape )
		# nrmbatch = np.empty( batchshape )

		for j in range(batchsize):
			fname = flist[i+j]
			lumbatch[j,] = flip( hdrread( fname ) )
			refbatch[j,] = flip( hdrread( fname, '_ref' ) )
			# nrmbatch[j,] = flip( hdrread( fname, '_nrm' ) )
		# *** double check batch organization

		if tensor:
			lumbatch = torch.tensor( lumbatch, dtype=torch.float )
			refbatch = torch.tensor( refbatch, dtype=torch.float )
			# nrmbatch = torch.tensor( nrmbatch, dtype=torch.float )
			if device != None:
				lumbatch = lumbatch.to( device )
				refbatch = refbatch.to( device )
				# nrmbatch = nrmbatch.to( device )

		yield lumbatch, refbatch # , nrmbatch

