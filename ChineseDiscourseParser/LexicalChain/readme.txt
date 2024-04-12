Please reference the paper if you are using any of its contents:

Multi-Sense Embeddings Through a Word Sense Disambiguation Process
Ruas, T.; Grosky, I.;Aizawa, A.


Contents:
Dataset:
	- original: all original datasets used in this paper
	- cosine parsed: similarity score normalized  [0,1]
	
Wikipedia Dump (April) 2010:
	- wd10_raw_combined - one wikipedia article per line
	- wd10_raw_sep - one wikipedia article per document
	- wikidump20100408_nbsd_synsets_single.tar - useed to generate synset embeddings (MSSA) - have the notation word#offset#pos
	- wikidump20100408_nbsd_synsets_refined.tar - useed to generate synset embeddings (MSSA1R) - have the notation word#offset#pos
	- wikidump20100408_nbsd_synsets_refined2.tar - useed to generate synset embeddings (MSSA2R) - have the notation word#offset#pos
	- wikidump20100408_nbsd_synsets_single.tar - output of MSSA Algorithm. One document per file - have the notation word \t synset \t offset \t pos
	- wikidump20100408_nbsd_synsets_refined.tar - output of MSSA1R Algorithm. One document per file - have the notation word \t synset \t offset \t pos
	- wikidump20100408_nbsd_synsets_refined2.tar - output of MSSA2R Algorithm. One document per file - have the notation word \t synset \t offset \t pos
	- wikipedia_cdump_20100408.tar - the actual dump (xml cleaned) from 2010
	- models -  contains of the synset embeddings models trained using word2vec (window: 15; minimum count: 10; hierarchical sampling; cbow; 300 and 1000 dimensions)
	
Wikipedia Dump (January) 2018:
	- wikidump20180120_nbsd_synsets_single.tar - useed to generate synset embeddings (MSSA) - has the notation word#offset#pos
	- wikidump20180120_dbsd_synsets_single.tar - useed to generate synset embeddings (MSSA-D) - has the notation word#offset#pos
	- wikidump20180120_nbsd_synsets_separate.tar - output of MSSA Algorithm. One document per file - have the notation word \t synset \t offset \t pos
	- wikidump20180120_dbsd_synsets_separate.tar - output of MSSA-D (Dijkistra) Algorithm. One document per file - have the notation word \t synset \t offset \t pos
	- wikipedia_cdump_20180120.tar - the actual dump (xml cleaned) from 2010
	- models -  contains of the synset embeddings models trained using word2vec (window: 15; minimum count: 10; hierarchical sampling; cbow; 300 and 1000 dimensions)