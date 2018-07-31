#!/usr/bin/env python3
#
# Run parametric model to generate CSV output for the HPSOs.

import sys
sys.path.append('..')

from sdp_par_model import reports as iapi
from sdp_par_model.parameters.definitions import HPSOs

ofile = 'hpsos.csv'
parallel = 4

iapi.write_csv_hpsos(ofile, HPSOs.hpsos, parallel=parallel)
