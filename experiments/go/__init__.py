import os
import sys

ppath = os.path.dirname(__file__)
if not ppath in sys.path:
    sys.path.append(ppath)

from GO import MSADataset, MSAEncoder