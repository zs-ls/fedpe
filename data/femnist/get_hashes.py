import hashlib
import os
import sys

sys.path.append("../../")

from utils import util

parent_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

cfd = os.path.join(parent_path, 'dataset', 'FEMINIST', 'intermediate', 'class_file_dirs')
wfd = os.path.join(parent_path, 'dataset', 'FEMINIST', 'intermediate', 'write_file_dirs')
class_file_dirs = util.load_obj(cfd)
write_file_dirs = util.load_obj(wfd)

class_file_hashes = []
write_file_hashes = []

count = 0
for tup in class_file_dirs:
    if (count % 100000 == 0):
        print('hashed %d class images' % count)

    (cclass, cfile) = tup
    file_path = os.path.join(parent_path, cfile)

    chash = hashlib.md5(open(file_path, 'rb').read()).hexdigest()

    class_file_hashes.append((cclass, cfile, chash))

    count += 1

cfhd = os.path.join(parent_path, 'dataset', 'FEMINIST', 'intermediate', 'class_file_hashes')
util.save_obj(class_file_hashes, cfhd)

count = 0
for tup in write_file_dirs:
    if (count % 100000 == 0):
        print('hashed %d write images' % count)

    (cclass, cfile) = tup
    file_path = os.path.join(parent_path, cfile)

    chash = hashlib.md5(open(file_path, 'rb').read()).hexdigest()

    write_file_hashes.append((cclass, cfile, chash))

    count += 1

wfhd = os.path.join(parent_path, 'dataset', 'FEMINIST', 'intermediate', 'write_file_hashes')
util.save_obj(write_file_hashes, wfhd)