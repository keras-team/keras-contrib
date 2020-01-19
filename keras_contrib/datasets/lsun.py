from __future__ import print_function, division
import argparse
import json
from os.path import join

import subprocess
import urllib2
from ..utils.data_utils import get_file

def download(out_dir='lsun', category, set_name, tag):
    url = 'http://lsun.cs.princeton.edu/htbin/download.cgi?tag={tag}' \
          '&category={category}&set={set_name}'.format(**locals())
    if set_name == 'test':
        out_name = 'test_lmdb.zip'
    else:
        out_name = '{category}_{set_name}_lmdb.zip'.format(**locals())
    out_path = join(out_dir, out_name)
    get_file(path = get_file(path, origin=url)
