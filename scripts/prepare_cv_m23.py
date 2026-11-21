"""Copy only existing approved M23 inputs into a new content-pinned snapshot."""
import argparse
from src.vision.intelligent23.inputs import prepare

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--base-inputs')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    prepare(a.output, a.base_inputs, a.resume)
