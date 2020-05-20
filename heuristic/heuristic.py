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
A simple heuristic for video quality assignment. It works as follows:
- Sorts the users in decreasing CQI order
- Starts with a minimum bitrate representation and assigns it to as many users as possible
- If there's available capacity, starts goes over the users and tries to increase their bitrate
- Continues in the same spirit until there's no capacity left or all users have taken the best representation possible. 

#################################################################
"""

from operator import itemgetter, attrgetter
import sys
import time 
import string
import math
import json 
import os
import copy
import getopt
from datetime import datetime

def _copy_users(users):
  """ Creates a copy of the list given as argument (to avoid calling deepcopy in crossover())
  """
  retval = []
  for u in users:
    data = {}
    for key, val in u.items():
      data[key] = val
    retval.append(data)
  return retval

class Heuristic(object):
  """Heuristic algorithm.
  """
  
  def __init__(self, scenario):
    """Constructor function.
    """     
    self.scenario = scenario

  def get_used_capacity(self, s):
    """Get the used capacity in terms of prbs
    """
    used_capacity = 0
    for u in s["users"]:
      if "bitrate" in u:
        used_capacity += u["bitrate"]/u["prb_throughput"]
    return used_capacity
 
  def execute(self):
    """ Run the heuristic algorithm. 
    """
    tstart = time.time()
    solution = copy.deepcopy(self.scenario)

    #sort representations by bitrate
    sorted_reps = sorted(solution["representations"], key=itemgetter('bitrate'))

    #sort users by channel quality
    sorted_users = sorted(solution["users"], key=itemgetter('cqi'), reverse = True)

    # available capacity in terms of prbs
    available_capacity = self.scenario["nprb"]

    # objective function value
    objective = 0.0

    for r in sorted_reps:
      # go over all users and assign representation r to as many as possible
      for u in sorted_users:
        if u["cqi"] == 0:
          # users are sorted--no need to go past the first user with CQI==0
          break

        if available_capacity < r["bitrate"]/u["prb_throughput"]:
          # ran out of resources--no more users can be "updated"
          # we might get here multiple times in a row, after capacity is over
          break

        if u["max_throughput"] >= r["bitrate"]:
          # there's capacity, so if the user can accommodate the bitrate of this representation, "upgrade" user
          if "mos" in u:
            objective -= u["mos"] # subtract current representation MOS from the objective, since it's going to be updated
          u["rid"] = r["id"]
          u["bitrate"] = r["bitrate"]
          u["mos"] = r["mos"]
          available_capacity -= r["bitrate"]/u["prb_throughput"]
          objective += r["mos"]
    tend = time.time()
    info = {}
    info["execution_time"] = tend-tstart
    info["used_capacity"] = self.get_used_capacity(solution)
    solution["objective"] = objective
    solution["solution_performance"] = info

    return solution


if __name__ == "__main__":
  inpath = None
  outpath = None
  silent = False # do not print solution to stdout
  myopts, args = getopt.getopt(sys.argv[1:], "i:o:s")
  for o, a in myopts:
    if o == "-i":
      inpath = a
    if o == "-o":
      outpath = a
    if o == "-s":
      silent = True

  # open 
  try:
    with open(inpath) as fp:
      scenario = json.load(fp)
  except:
    print("Error loading scenario file. Missing or malformed.")
    sys.exit(1)

  h = Heuristic(scenario)
  solution = h.execute()

  if outpath:
    try:
      with open(outpath, "w+") as fp:
        json.dump(solution, fp, indent=2)
    except:
      print("Output file could not be opened.")
      sys.exit(2)

  if not silent:
    print(json.dumps(solution, indent=2))
  else:
    # only print objective function value
    print(solution["objective"])

