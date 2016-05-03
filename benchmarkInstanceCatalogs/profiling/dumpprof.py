# From http://cyrille.rossant.net/profiling-and-optimizing-python-code/
import pstats
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("fname", help="output of cProfile")
parser.add_argument("outfile", help="output file for statistics")
args = parser.parse_args()

fname = args.fname
outfile = args.outfile

with open(outfile, 'w') as fp:
    ps = pstats.Stats(fname, stream=fp)
    ps.strip_dirs().sort_stats("cumulative").print_stats()
