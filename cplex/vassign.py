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


import cplex
import numpy
import time
import sys
import getopt
import yaml
import json
import copy
import multiprocessing

from operator import itemgetter, attrgetter

def vassign(configuration, threads = 0):
  """
  Loads network configuration (user characteristics, network capacity, representations, etc. from a scenario file and solves the ILP.
  """
  # num users
  N = len(configuration["users"])

  # Capacity in terms of pRBs
  capacity = configuration["nprb"]

  # num representations
  Q = len(configuration["representations"]) + 1 #+1 for dummy zero representation

  b = [0]
  mos = [0]
  for r in sorted(configuration["representations"], key=itemgetter('bitrate')):
    b.append(r["bitrate"])
    mos.append(r["mos"])

  # load link capacities
  l = [u["max_throughput"] for u in configuration["users"]]
  l1 = [u["prb_throughput"] for u in configuration["users"]]

  # x_{i,j} variable names
  varnames = []
  vartypes = ""
  pairs = []
  # generate variable names
  for i in range(1, N+1):
    for j in range(1, Q+1):
      varnames.append("x-" + str(i) + "-" + str(j))
      vartypes += "B"
      pairs.append(["x-" + str(i) + "-" + str(j), mos[j-1]])

  # setup cplex
  c = cplex.Cplex()
  c.parameters.threads.set(threads)
  c.parameters.parallel.set(-1)
  c.set_results_stream(None)
  c.set_log_stream(None)
  c.set_problem_name("vassign")

  # maximization problem
  c.objective.set_sense(c.objective.sense.maximize)

  # create variables
  c.variables.add(names = varnames, types=vartypes)
  c.objective.set_linear(pairs)

  name2idx = { n : j for j, n in enumerate(c.variables.get_names()) }

  # create 1 overall capacity constraint: Sum Sum x_ij <= Cj
  vars = []
  coef = []
  for i in range(1,N+1):
    for j in range(1,Q+1):
      vars.append("x-" + str(i) + "-" + str(j))
      if l1[i-1] == 0:
        # for users with zero CQI, the necessary # of prbs -> inf
        coef.append(100000)
      else:
        coef.append(b[j-1]/l1[i-1])

  c.linear_constraints.add(
                           lin_expr=[cplex.SparsePair(ind = vars, val = coef)],
                           senses=["L"],
                           rhs=[capacity]
                           )

  # Create individual link constraints
  LHS = []
  senses = ""
  RHS = []
  for i in range(1,N+1):
    for j in range(1,Q+1):
      LHS.append(cplex.SparsePair(ind=[name2idx["x-" + str(i) + "-" + str(j)]], val = [b[j-1]]))
      senses += "L"
      RHS.append(l[i-1])

  c.linear_constraints.add(
                       lin_expr=LHS,
                       senses=senses,
                       rhs=RHS
                       )

  # Verify that each user can only receive at most 1 representation
  # [one constraint per user]
  LHS = []
  senses = ""
  RHS = []
  for i in range(1, N+1):
    vars = []
    coef = [1]*Q
    for j in range(1, Q+1):
      vars.append(name2idx["x-" + str(i) + "-" + str(j)])
    LHS.append(cplex.SparsePair(ind = vars, val = coef))
    senses += "L"
    RHS.append(1)

  c.linear_constraints.add(
                         lin_expr=LHS,
                         senses=senses,
                         rhs=RHS
                         )

  start = time.time()
  c.solve()
  end = time.time()
  sol = c.solution

  # Update the configuration structure with the solution 
  # (assignment per user)  
  capacity_used = 0
  for i in range(1, N+1):
    vars = []
    for j in range(1, Q+1):
      vars.append("x-" + str(i) + "-" + str(j))
    values = sol.get_values(vars)
    for j in range(len(values)):
      if int(values[j]) == 1:
        #print("User " + str(i) + ": " + str(j) + "/" + str(mos[j]))
        configuration["users"][i-1]["mos"] = mos[j]
        configuration["users"][i-1]["rid"] = str(j - 1)
        configuration["users"][i-1]["bitrate"] = b[j]
        capacity_used += b[j]
  configuration["objective"] = sol.get_objective_value() 
  return configuration
  
if __name__ == '__main__':    
  # open configuration file
  myopts, args = getopt.getopt(sys.argv[1:], "c:")
  configfile = None
  for o, a in myopts:
    if o == "-c":
      configfile = a
  if configfile == None:
    print("Missing configuration file. Usage: python vassign.py -c /path/to/configfile")
    exit(1)

  try:
    with open(configfile) as fp:
      configuration = json.load(fp)
  except:
    print("Error loading scenario file. Possibly malformed...")

  vassign(configuration)

