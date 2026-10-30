"""Run the complete experiment from the repository directory."""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(name, '1')
import argparse
from abalone_project import run_all
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-search', action='store_true',
        help='Use the archived best RF settings instead of repeating 50 CV fits.')
    args = parser.parse_args()
    run_all(search=not args.skip_search)
