# -*- coding: utf-8 -*-

"""
Copyright 2020 Pantelis Frangoudis

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
#################################################################
Generates a random input for the GA for video quality assignment.
It draws random UE CQIs which it translates to maximum achievable 
bitrates based on 3GPP Release 14 specifications.
#################################################################
"""

from random import *
import time 
import json 
import getopt
import sys
import random
from datetime import datetime

# defaults
NPRB_DEFAULT = 25
NUSERS_DEFAULT = 100

# Mapping of CQI to MCS index. Table 7.2.3-2 of 3gpp Rel. 14 links the CQI with the modulation order & coding rate.
# Each modulation order is mapped (manually) to an appropriate (i.e., with the same modulation order) MCS index (there are more than one per Qm)   
# CQI 1-3: QPSK (Qm=2)
# CQI 4-6: 16QAM (Qm=4)
# CQI 7-11: 64QAM (Qm=6)
# CQI 12-15: 256QAM (Qm=8)
cqi_to_mcs_index = [1, 3, 4, 5, 7, 10, 11, 13, 15, 17, 18, 19, 25, 26, 27]

# Mapping of MCS index to TBS (Table 7.1.7.1-1A of 3GPP Rel. 14)
mcs_to_tbs_index = [0, 2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33]

# Mapping of TBS index to TBS for 25 PRBs (Table 7.1.7.2.1-1 of 3GPP Rel. 14)
# tbs / subframe / antenna --> *1000*num_antennas == max throughput
tbs_index_to_tbs_25 = [
  680,
  904,
  1096,
  1416,
  1800,
  2216,
  2600,
  3112,
  3496,
  4008,
  4392,
  4968,
  5736,
  6456,
  7224,
  7736,
  7992,
  9144,
  9912,
  10680,
  11448,
  12576,
  13536,
  14112,
  15264,
  15840,
  16416,# #26: ignored
  16416,
  17568,
  18336,
  19848,
  20616,
  21384,
  24496
]

tbs_index_to_tbs_100 = [
  2792,
  3624,
  4584,
  5736,
  7224,
  8760,
  10296,
  12216,
  14112,
  15840,
  17568,
  19848,
  22920,
  25456,
  28336,
  30576,
  32856,
  36696,
  39232,
  43816,
  46888,
  51024,
  55056,
  57336,
  61664, 
  63766, 
  66592, # #26: ignored
  66592,
  71112,
  73712,
  78704,
  81176,
  84760,
  97896
]

# video representation characteristics
#b = [117301, 238080, 487065, 977972, 1955438, 3901437]
b = [117, 238, 487, 977, 1955, 3901] # scaled down for the DP algorithm to run faster.
mos = [1.0672, 1.43, 2.6878, 4.1773, 4.8103, 4.9589]

# Uncomment to allow only the better quality representations to be used
# b = [487, 977, 1955, 3901] 
# mos = [2.6878, 4.1773, 4.8103, 4.9589]

def generate_scenario(nprb, nusers, seed = None, outpath = None, randomize_video = False, allow_disconnected_ues = False, rate_unit_bps = False):
  """Generates a scenario.

  randomize_video: Set to True to create random video representations with no correlation 
  between bitrate and MOS. 
  allow_disconnected_ues: Set to True to allows to create UEs with CQI == 0 and prb_throughput == 0.
  rate_unit_bps: Report capacity values in bps instead of kbps which is the default.
  """
  global b
  global mos
  
  if rate_unit_bps:
    scale_by = 1000
  else:
    scale_by = 1

  if randomize_video:
    b = []
    mos = []
    nrep = 6
    # create random video representation characteristics
    for i in range(nrep):
      b.append(random.randint(100, 5000))
      mos.append(random.uniform(1, 5))

  # Network downlink capacity (max achievable throughput for perfect channel conditions) 
  capacity = 0
  if nprb == 25:
    #capacity = tbs_index_to_tbs_25[mcs_to_tbs_index[cqi_to_mcs_index[14]]]*1000
    capacity = tbs_index_to_tbs_25[mcs_to_tbs_index[cqi_to_mcs_index[14]]] # scaled down for the DP algorithm to run faster (same below)
  if nprb == 100:
    #capacity = tbs_index_to_tbs_100[mcs_to_tbs_index[cqi_to_mcs_index[14]]]*1000
    capacity = tbs_index_to_tbs_100[mcs_to_tbs_index[cqi_to_mcs_index[14]]]

  # seed PRNG
  if not seed:
    seed = int(datetime.now().timestamp()*1000)
  random.seed(seed)

  representations = []
  for i in range(0, len(b)):
    representations.append({"id": str(i), "bitrate": b[i], "mos": mos[i]})

  users = []
  for i in range(0, nusers):
    # draw random cqi value (0 means CQI==1, etc.)
    if allow_disconnected_ues:
      cqi = random.randint(-1, 14)
    else:
      cqi = random.randint(0, 14)

    r1prb = 0
    rmax = 0

    # if cqi == -1, user is disconnected
    if cqi == -1:
      r1prb = 0
      rmax = 0
    else:  
      # calculate max achievable throughput for user
      if nprb == 25:
        rmax = tbs_index_to_tbs_25[mcs_to_tbs_index[cqi_to_mcs_index[cqi]]]*scale_by 
      else:
        # 100 prbs (20 MHz bandwidth)
        rmax = tbs_index_to_tbs_100[mcs_to_tbs_index[cqi_to_mcs_index[cqi]]]*scale_by
      
      # estimate throughput achievable in 1 PRB
      r1prb = rmax/float(nprb)
    
    users.append({"id": str(i), "cqi": cqi+1, "max_throughput": rmax, "prb_throughput": r1prb})

  # create scenario
  scenario = {"nprb": nprb, "capacity": capacity, "seed": seed, "users": users, "representations": representations}

  if outpath:
    fp = open(outpath, "w+")
    json.dump(scenario, fp, indent=2)
    fp.close()
  else:
    print(json.dumps(scenario, indent=2))

if __name__ == "__main__":
  # Get configuration from command line. If a seed is not provided, the current time is used
  # n: nusers
  # s: seed
  # p: nprb
  # o: output file
  seed = None
  nprb = None
  nusers = None
  outpath = None
  myopts, args = getopt.getopt(sys.argv[1:], "n:p:s:o:")
  for o, a in myopts:
    if o == "-n":
      nusers = int(a)
    if o == "-p":
      nprb = int(a)
    if o == "-s":
      seed = int(a)
    if o == "-o":
      outpath = a

  if not seed:
    seed = int(time.time())
  if not nprb or (nprb != 25 and nprb != 100):
    nprb = NPRB_DEFAULT
  if not nusers:
    nusers = NUSERS_DEFAULT

  generate_scenario(nprb, nusers, seed, outpath, allow_disconnected_ues=False, rate_unit_bps=False)

