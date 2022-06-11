import os, glob

def fullpath( p ):
    return os.path.abspath( os.path.expanduser(p) ) if len(p.strip())>0 else ''

def fileparts( fname ):
	p, n = os.path.split( fname )
	n, e = os.path.splitext( n )
	return p, n, e

def dirsort( filt ):
	flist = glob.glob( filt )
	flist.sort()
	return flist

