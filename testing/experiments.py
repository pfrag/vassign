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

from ga import *
from vassign import *
from generate import *
from heuristic import *
import itertools
from scipy.stats import sem, t, norm
from scipy import mean

def get_average_mos(solution, ignore_non_served=False):
  """Returns the average MOS of a solution.
  """
  smos = 0
  nserved = 0
  for u in solution["users"]:
    if "mos" in u:
      smos += u["mos"]
      nserved += 1
  if ignore_non_served:
    # only take into account users that get some video
    return smos/nserved
  else:
    # calculate avg QoE across all users
    return smos/len(solution["users"])
    
def get_ratio_served(solution):
  """Returns the ratio of users that have been assigned a video representation.
  """
  nserved = 0
  for u in solution["users"]:
    if "mos" in u:
      nserved += 1
  return nserved/len(solution["users"])

#######################################################################
#######################################################################

def exp_compare_algorithms(randomize = False):
  """
  Compares the performance of the b&c (cplex), GA, and baseline heuristic.
  """

  iterations = 200
  confidence = 0.95

  R = itertools.chain(range(10, 100, 10), range(100, 2000, 100), range(2000, 5001, 1000))

  # generate ga configuration object
  settings = {
    "crossover_rate": 0.5,
    "mutation_rate": 0.0,
    "solution_pool_size": 40,
    "generations": 30,
    "convergence_check": True,
    "delta": 0.01,
    "stop_after": 15,
    "scenario_file": "in",
    "solution_file": "./solution.json",
    "generations_file": "./generations.dat",
    "loglevel": "NONE",
  }

  print("# Number of iterations: ", iterations)
  print("# Crossover rate: ", settings["crossover_rate"])
  print("# Mutation rate: ", settings["mutation_rate"])
  print("# Solution pool size: ", settings["solution_pool_size"])
  print("# Generations: ", settings["generations"])
  print("# Convergence check: ", settings["convergence_check"])
  print("# Delta: ", settings["delta"])
  print("# Stop after: ", settings["stop_after"])
  print("###########################################")

  for nusers in R:
    # The results of all executions are held here, 1 element per iteration
    results = []

    for i in range(0, iterations):
      # generate scenario file
      generate_scenario(100, nusers, outpath = "in", randomize_video = randomize)

      # run ga
      g = GA(settings)
      start = time.time()
      solution = g.execute()
      end = time.time()
      gaval = solution["fitness"]
      tga = end - start 
      # get average MOS
      gamos = get_average_mos(solution)
      gamos_assigned = get_average_mos(solution, True) # Measure also avg MOS without taking into account users that are not assigned video
      # get ratio of users which receive video
      gaserved = get_ratio_served(solution)

      #################################################

      # run cplex
      with open("in") as fp:
        configuration = json.load(fp)

      start = time.time()
      solution = vassign(configuration)
      optval = solution["objective"]
      end = time.time()
      topt = end - start
      optmos = get_average_mos(solution)
      optmos_assigned = get_average_mos(solution, True)
      optserved = get_ratio_served(solution)

      #################################################

      # run heuristic
      with open("in") as fp:
        configuration = json.load(fp)
      h = Heuristic(configuration)
      start = time.time()
      solution = h.execute()
      end = time.time()
      hval = solution["objective"]
      theur = end - start 
      hmos = get_average_mos(solution)
      hmos_assigned = get_average_mos(solution, True)
      hserved = get_ratio_served(solution)

      hratio = hval/optval
      gratio = gaval/optval
      
      results.append({"optval": optval, "gaval": gaval, "hval": hval, "gratio": gratio, "hratio": hratio, "topt": topt, "tga": tga, "theur": theur, "optmos": optmos, "gamos": gamos, "hmos": hmos, "optserved": optserved, "gaserved": gaserved, "hserved": hserved, "gamos_assigned": gamos_assigned, "optmos_assigned": optmos_assigned, "hmos_assigned": hmos_assigned})


    #################################################
    # Calculate statistics (TODO: cleanup code, too ugly)
    #################################################

    # Calculation of statistics for obj function values
    optval = [r["optval"] for r in results]
    moptval = mean(optval)
    std_err = sem(optval)
    ciopt = std_err * t.ppf((1 + confidence) / 2, iterations - 1)
    
    gaval = [r["gaval"] for r in results]
    mgaval = mean(gaval)
    std_err = sem(gaval)
    ciga = std_err * t.ppf((1 + confidence) / 2, iterations - 1)

    hval = [r["hval"] for r in results]
    mhval = mean(hval)
    std_err = sem(hval)
    cih = std_err * t.ppf((1 + confidence) / 2, iterations - 1)

    # Calculation of statistics for optimality gaps
    gratio = [r["gratio"] for r in results]
    mgratio = mean(gratio)
    std_err = sem(gratio)
    cigratio = std_err * t.ppf((1 + confidence) / 2, iterations - 1)

    hratio = [r["hratio"] for r in results]
    mhratio = mean(hratio)
    std_err = sem(hratio)
    cihratio = std_err * t.ppf((1 + confidence) / 2, iterations - 1)

    # Calculation of statistics for execution times
    topt = [r["topt"] for r in results]
    mtopt = mean(topt)
    std_err = sem(topt)
    citopt = std_err * t.ppf((1 + confidence) / 2, iterations - 1)    

    tga = [r["tga"] for r in results]
    mtga = mean(tga)
    std_err = sem(tga)
    citga = std_err * t.ppf((1 + confidence) / 2, iterations - 1)    

    theur = [r["theur"] for r in results]
    mtheur = mean(theur)
    std_err = sem(theur)
    citheur = std_err * t.ppf((1 + confidence) / 2, iterations - 1)    

    # Calculation of statistics for avg mos
    optmos = [r["optmos"] for r in results]
    moptmos = mean(optmos)
    std_err = sem(optmos)
    ci_optmos = std_err * t.ppf((1 + confidence) / 2, iterations - 1)    

    gamos = [r["gamos"] for r in results]
    mgamos = mean(gamos)
    std_err = sem(gamos)
    ci_gamos = std_err * t.ppf((1 + confidence) / 2, iterations - 1)    
    
    hmos = [r["hmos"] for r in results]
    mhmos = mean(hmos)
    std_err = sem(hmos)
    ci_hmos = std_err * t.ppf((1 + confidence) / 2, iterations - 1)    

    # Same for avg mos only when served users are considered
    optmos_assigned = [r["optmos_assigned"] for r in results]
    moptmos_assigned = mean(optmos_assigned)
    std_err = sem(optmos_assigned)
    ci_optmos_assigned = std_err * t.ppf((1 + confidence) / 2, iterations - 1)    

    gamos_assigned = [r["gamos_assigned"] for r in results]
    mgamos_assigned = mean(gamos_assigned)
    std_err = sem(gamos_assigned)
    ci_gamos_assigned = std_err * t.ppf((1 + confidence) / 2, iterations - 1)    
    
    hmos_assigned = [r["hmos_assigned"] for r in results]
    mhmos_assigned = mean(hmos_assigned)
    std_err = sem(hmos_assigned)
    ci_hmos_assigned = std_err * t.ppf((1 + confidence) / 2, iterations - 1)    

    # calculation of statistics for %users served
    optserved = [r["optserved"] for r in results]
    moptserved = mean(optserved)
    std_err = sem(optserved)
    ci_optserved = std_err * t.ppf((1 + confidence) / 2, iterations - 1)

    gaserved = [r["gaserved"] for r in results]
    mgaserved = mean(gaserved)
    std_err = sem(gaserved)
    ci_gaserved = std_err * t.ppf((1 + confidence) / 2, iterations - 1)

    hserved = [r["hserved"] for r in results]
    mhserved = mean(hserved)
    std_err = sem(hserved)
    ci_hserved = std_err * t.ppf((1 + confidence) / 2, iterations - 1)

    print(
      nusers, 
      moptval, moptval-ciopt, moptval+ciopt, 
      mgaval, mgaval-ciga, mgaval + ciga, 
      mhval, mhval-cih, mhval+cih, 
      mtopt, mtopt-citopt, mtopt+citopt,
      mtga, mtga-citga, mtga+citga, 
      mtheur, mtheur-citheur, mtheur+citheur,

      mgratio, mgratio-cigratio, mgratio+cigratio,
      mhratio, mhratio-cihratio, mhratio+cihratio,

      moptmos, moptmos-ci_optmos, moptmos+ci_optmos,
      mgamos, mgamos-ci_gamos, mgamos+ci_gamos,
      mhmos, mhmos-ci_hmos, mhmos+ci_hmos,

      moptmos_assigned, moptmos_assigned-ci_optmos_assigned, moptmos_assigned+ci_optmos_assigned, 
      mgamos_assigned, mgamos_assigned-ci_gamos_assigned, mgamos_assigned+ci_gamos_assigned, 
      mhmos_assigned, mhmos_assigned-ci_hmos_assigned, mhmos_assigned+ci_hmos_assigned,

      moptserved, moptserved-ci_optserved, moptserved+ci_optserved,
      mgaserved, mgaserved-ci_gaserved, mgaserved+ci_gaserved, 
      mhserved, mhserved-ci_hserved, mhserved+ci_hserved
    )

def exp_multithread(randomize = False):
  """Test CPLEX execution time vs. the number of threads/cores
  """

  iterations = 100
  confidence = 0.95

  R = itertools.chain(range(10, 100, 10), range(100, 2000, 100), range(2000, 10001, 1000), range(15000, 20001, 5000))
  #R = itertools.chain(range(1000, 20001, 1000))
  for nusers in R:
    results = []
    for i in range(iterations):
      # generate scenario file
      generate_scenario(100, nusers, outpath = "in", randomize_video = randomize)
      with open("in") as fp:
        configuration = json.load(fp)

      r = {}
      for j in range(1,5):
        start = time.time()
        optval = vassign(configuration, j)
        end = time.time()
        r["t"+str(j)] = end - start
      results.append(r)

    print(nusers, end = ' ')
    for j in range(1, 5):
      topt = [r["t" + str(j)] for r in results]
      mtopt = mean(topt)
      std_err = sem(topt)
      citopt = std_err * t.ppf((1 + confidence) / 2, iterations - 1)
      print(mtopt, mtopt - citopt, mtopt + citopt, end = ' ')
    print() 

exp_compare_algorithms(False)
#exp_multithread(True)

