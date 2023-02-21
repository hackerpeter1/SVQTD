import kaldi_io as kio
import sys
import numpy as np
import os
import multiprocessing

indir  = sys.argv[1]
oudir = sys.argv[2]

egs_scp =  indir + '/egs.scp'
lang_scp = indir + '/utt2spk'

if not os.path.exists(egs_scp) and os.path.exists(lang_scp):	
	sys.exit()

lang_dict = {}
with open(lang_scp) as fa:
	for line in fa.readlines():
		item = line.strip().split(' ')
		lang_dict[item[0]] = item[1]
		

if not os.path.isdir(oudir):
	os.mkdir(oudir)

def process():	
	for key, mat in kio.read_mat_scp(egs_scp):
		lang = lang_dict[key]
		lang_dir = oudir + '/' + lang
		if not os.path.isdir(lang_dir):
			os.mkdir(lang_dir)
		savename = lang_dir + "/" + key
		print(savename, mat.shape)
#		if mat.shape[0] >= 200:
		np.save(savename, mat[:,1:])
#		np.save(savename, mat)

def process1():
    for key, mat in kio.read_mat_scp(egs_scp):
        np.save(oudir + '/' + key, mat[:,1:])
process()
