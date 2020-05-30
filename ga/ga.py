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
A genetic algorithm for video quality assignment.

The relevant parameters can be tuned in the configuration file.
The scenario_file parameter must point to a valid JSON-formatted
file with all the scenario parameters.
#################################################################
"""

from random import *
from operator import itemgetter, attrgetter
import sys
import time 
import string
import math
import json 
import os
import copy
import logging
import random
from datetime import datetime
import msgpack
import ujson
import numpy

def randint(low, high):
  """ A faster (?) randint
  """
  # One way to do it is as follows:
  #k = numpy.frombuffer(numpy.random.bytes(4), dtype=numpy.uint32)
  #return numpy.mod(k, high - low + 1)[0] + low
  # Another way: just call random() and scale appropriately
  return int(low + (high+1)*random.random())

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

class chromosome(object):
  def __init__(self, genes, fitness = 0.0):
    self.genes = genes
    self.fitness = fitness

  def __str__(self):
    return str({"genes": self.genes, "fitness": self.fitness})

  def __repr__(self):
    return str({"genes": self.genes, "fitness": self.fitness})

  @staticmethod
  def create_empty(ngenes):
    genes = []
    for i in range(ngenes):
      genes.append(gene(i))
    return chromosome(genes)

  @staticmethod
  def copy(c):
    genes = []
    if c.genes:
      for i in range(len(c.genes)):
        genes.append(gene(c.genes[i].uid, c.genes[i].rid, c.genes[i].bitrate, c.genes[i].mos, c.genes[i].max_throughput, c.genes[i].prb_throughput))
    return chromosome(genes, c.fitness)

class gene(object):
  def __init__(self, uid, rid = None, bitrate = 0, mos = 0.0, max_throughput = 0.0, prb_throughput = 0.0):
    self.uid = uid
    self.rid = rid
    self.bitrate = bitrate
    self.mos = mos
    self.max_throughput = max_throughput
    self.prb_throughput = prb_throughput

  def __str__(self):
    return str({"user": self.uid, "rid": self.rid, "bitrate": self.bitrate, "mos": self.mos, "max_throughput": self.max_throughput})

  def __repr__(self):
    return str({"user": self.uid, "rid": self.rid, "bitrate": self.bitrate, "mos": self.mos, "max_throughput": self.max_throughput})

class GA(object):
  """Genetic Algorithm class.
  """
  
  ERR_MALFORMED_INPUT = 400
  ERR_NO_INPUT_PROVIDED = 410
  ERR_NO_FEASIBLE_SOLUTION = 420

  def __init__(self, configuration):
    """Constructor function.
    """
    
    # seed random number generator (if this option is set in the config file)
    if "seed" in configuration:
      self.seed = configuration["seed"]
    else:
      self.seed = int(datetime.now().timestamp()*1000)
    random.seed(self.seed)
    
    # Load GA-specific params (S, G, Rc, Rm)
    # Load VNFFG, host graph, constraints, etc.
    self.solution_pool_size = configuration['solution_pool_size']
    self.generations = configuration['generations']
    self.crossover_rate = configuration['crossover_rate']
    self.mutation_rate = configuration['mutation_rate']
    self.convergence_check = configuration['convergence_check']
    self.error = None
    self.error_string = None

    if self.convergence_check:
      self.delta = configuration['delta']
      self.stop_after = configuration['stop_after']

    if "generations_file" in configuration:
      self.generations_file = configuration['generations_file']
    else:
      self.generations_file = None
    
    loglevel = getattr(logging, configuration["loglevel"].upper(), None)
    logging.basicConfig(level=loglevel)
    
    if 'scenario' in configuration:
      self.scenario = configuration["scenario"]
      if "scenario_file" not in configuration:
        self.scenario_file = None
    elif 'scenario_file' in configuration:
      self.scenario_file = configuration["scenario_file"]
      try:
        with open(configuration['scenario_file']) as fp:
          self.scenario = json.load(fp)
      except:
        logging.error("Error loading scenario file. Possibly malformed...")
        self.error_string = "Error loading scenario file. Possibly malformed."
        self.error = GA.ERR_MALFORMED_INPUT
    else:
      logging.error("Scenario not provided.")
      self.error_string = "No input provided."
      self.error = GA.ERR_NO_INPUT_PROVIDED
    
    if configuration["loglevel"].upper() != "NONE":
      self.print_configuration()

  def print_configuration(self):
    """Print the configuration
    """
    
    print("Configuration:\n-------------------------------" +
          "\n- Solution pool size:\t\t" + str(self.solution_pool_size) + 
          "\n- Number of generations:\t" + str(self.generations) +
          "\n- Crossover rate:\t\t" + str(self.crossover_rate) +
          "\n- Mutation rate:\t\t" + str(self.mutation_rate) +
          "\n- Scenario file:\t\t" + str(self.scenario_file) +
          "\n- Log data per generation:\t" + str(self.generations_file) +
          "\n- Enable convergence check:\t" + str(self.convergence_check))
    if self.convergence_check:
      print("\t+ Delta:\t\t" + str(self.delta) +
            "\n\t+ Stop after:\t\t" + str(self.stop_after) + 
            "\n- RNG seed:\t\t\t" + str(self.seed) +
            "\n-------------------------------\n")


  def fitness(self, solution):
    """Calculate the fitness of chromosome C.
    
    The fitness of a solution is the sum of the MOS values of all users.
    """
    retval = 0
    for g in solution.genes:
      if g.mos:
        retval += g.mos
    return retval

  def get_used_capacity(self, s):
    """Get the used capacity in terms of prbs
    """
    used_capacity = 0.0
    for g in s.genes:
      if g.bitrate:
        used_capacity += g.bitrate/g.prb_throughput
    return used_capacity

  def check_capacity_constraint(self, solution):
    """Chech whether the capacity constraint of a solution is respected (i.e., the number of needed prbs are less than the maximum available).
    """
    if self.get_used_capacity(solution) > self.scenario["nprb"]:
      return False
    return True
 
  def init_solution_pool(self):
    """Initialize solution pool.
    
    Generate S feasible assignments as follows:
    - Pick a random user
    - Assign a random feasible representation
    - Loop until the global capacity is exceeded or all users have a representation assigned
    """
    self.solution_pool = []
    solutions_to_generate = self.solution_pool_size
  
    sorted_reps = sorted(self.scenario["representations"], key=itemgetter('bitrate'))

    while solutions_to_generate > 0:
      solution = chromosome.create_empty(len(self.scenario["users"]))
      fitness = 0

      # Create a vector of user indexes to keep track which users are remaining to be assigned a representation
      user_indexes = range(0, len(self.scenario["users"]))
      available_capacity = self.scenario["nprb"]

      while len(user_indexes) > 0:
        # while there are still users without a video, 
        # draw a random user
        ptr = randint(0, len(user_indexes) - 1)
        uindex = user_indexes[ptr]

        if self.scenario["users"][uindex]["cqi"] == 0:
          # user disconnected -> ignore
          user_indexes.remove(uindex)
          continue

        # Check if there's still capacity
        minbitrate = sorted_reps[0]["bitrate"]
        # prbs the user needs for the minimum bitrate representation
        minprb = minbitrate/self.scenario["users"][uindex]["prb_throughput"]

        if available_capacity < minprb:
          # ok, no more available capacity even for the lowest bitrate representation
          # so we're done
          break

        # find which representations can be supported by the user (and the network)
        admissible_reps = copy.copy(sorted_reps)
        admissible_reps[:] = [r for r in admissible_reps if r["bitrate"] <= self.scenario["users"][uindex]["max_throughput"] and r["bitrate"]/self.scenario["users"][uindex]["prb_throughput"] <= available_capacity]

        # if user has the capacity to host at least the minimal bitrate representation
        if len(admissible_reps) > 0:
        #if rexists:
          # pick a random (admissible) representation          
          r = choice(admissible_reps)
          #R = randint(0, rindex - 1)
          #r = sorted_reps[R]
          solution.genes[uindex].rid = r["id"]
          solution.genes[uindex].bitrate = r["bitrate"]
          solution.genes[uindex].mos = r["mos"]
          solution.genes[uindex].prb_throughput = self.scenario["users"][uindex]["prb_throughput"]
          solution.genes[uindex].max_throughput = self.scenario["users"][uindex]["max_throughput"]
          available_capacity -= r["bitrate"]/self.scenario["users"][uindex]["prb_throughput"]
          solution.fitness += r["mos"]
        user_indexes.remove(uindex) # done with this user (either gets the above representation or nothing)
      self.solution_pool.append(solution)
      solutions_to_generate -= 1

  def init_solution_pool_v2(self):
    """Initialize solution pool.
    
    Generate S feasible assignments as follows:
    - First create a number of solutions where all users take the same representation
    - Do a number of crossover operations to create the rest of the solution pool
    """
    self.solution_pool = []
    solutions_to_generate = self.solution_pool_size

    #sort representations by bitrate
    sorted_reps = sorted(self.scenario["representations"], key=itemgetter('bitrate'))

    #sort users by channel quality
    sorted_users = sorted(self.scenario["users"], key=itemgetter('cqi'), reverse = True)

    for r in sorted_reps:
      solution = chromosome.create_empty(len(self.scenario["users"]))
      available_capacity = self.scenario["nprb"]
      for u in sorted_users:
        uindex = int(u["id"])
        if u["cqi"] > 0 and available_capacity >= r["bitrate"]/u["prb_throughput"] and u["max_throughput"] >= r["bitrate"]:
          # assign r to user and move to next one
          solution.genes[uindex].rid = r["id"]
          solution.genes[uindex].bitrate = r["bitrate"]
          solution.genes[uindex].mos = r["mos"]
          solution.genes[uindex].prb_throughput = u["prb_throughput"]
          solution.genes[uindex].max_throughput = u["max_throughput"]
          available_capacity -= r["bitrate"]/u["prb_throughput"]
          solution.fitness += r["mos"]
        else:
          solution.genes[uindex].rid = None
          solution.genes[uindex].bitrate = 0
          solution.genes[uindex].mos = 0.0
          solution.genes[uindex].prb_throughput = 0
          solution.genes[uindex].max_throughput = 0
      self.solution_pool.append(solution)
      solutions_to_generate -= 1
  
    # now that we have some initial solutions, create the rest based on them via crossover         
    while solutions_to_generate > 0:
      offspring = self.crossover_multi()
      for o in offspring:
        solutions_to_generate -= 1
        self.solution_pool.append(o)

  def crossover(self):
    """Crossover operation.
    
    - Select two random chromosomes (solutions) from the pool
    - Draw a random split point
    - Create two new chromosomes joining one part from the one and one from the other parent
    - Return the one with the better quality or None if none is ok with the
    network capacity constraint
    """

    # Pick two random chromosomes (C1 and C2 could coincide)  
    C1 = choice(self.solution_pool)
    C2 = choice(self.solution_pool)
    # select split point
    split_point = randint(0, len(self.scenario["users"]) - 1)

    # each offspring is the crossed-over array of the users of the parents
    offspring1 = chromosome(C1.genes[:split_point] + C2.genes[split_point:])
    offspring2 = chromosome(C2.genes[:split_point] + C1.genes[split_point:])
    used_capacity1 = 0
    fitness1 = 0
    ok1 = False
    used_capacity2 = 0
    fitness2 = 0
    ok2 = False 
 
    # check capacity constraints and calculate fitness in one pass
    for i in range(0, len(offspring1.genes)):
      if offspring1.genes[i].rid and offspring1.genes[i].prb_throughput > 0:
        used_capacity1 += offspring1.genes[i].bitrate/offspring1.genes[i].prb_throughput
        fitness1 += offspring1.genes[i].mos
      if offspring2.genes[i].rid and offspring2.genes[i].prb_throughput > 0:
        used_capacity2 += offspring2.genes[i].bitrate/offspring2.genes[i].prb_throughput
        fitness2 += offspring2.genes[i].mos

    if used_capacity1 <= self.scenario["nprb"]:
        ok1 = True
    if used_capacity2 <= self.scenario["nprb"]:
        ok2 = True

    offspring1.fitness = fitness1
    offspring2.fitness = fitness2

    if not ok1 and not ok2:
      return None
    elif ok1 and not ok2:
      return offspring1
    elif ok2 and not ok1:
      return offspring2
    else:
      if fitness1 > fitness2:
        return offspring1
      else:
        return offspring2

  def crossover_multi(self):
    """Crossover operation.
    
    - Select two random chromosomes (solutions) from the pool
    - Draw a random split point
    - Create two new chromosomes joining one part from the one and one from the other parent
    - Return up to two chromosomes (depending on whether they're ok with the capacity constraint)
    """

    offsprings = []

    # Pick two random chromosomes (C1 and C2 could coincide)  
    C1 = choice(self.solution_pool)
    C2 = choice(self.solution_pool)
    # select split point
    split_point = randint(0, len(self.scenario["users"]) - 1)

    # each offspring is the crossed-over array of the users of the parents
    offspring1 = chromosome(C1.genes[:split_point] + C2.genes[split_point:])
    offspring2 = chromosome(C2.genes[:split_point] + C1.genes[split_point:])
    used_capacity1 = 0
    fitness1 = 0
    ok1 = False
    used_capacity2 = 0
    fitness2 = 0
    ok2 = False 

    # check capacity constraints and calculate fitness in one pass
    for i in range(0, len(offspring1.genes)):
      if offspring1.genes[i].rid and offspring1.genes[i].prb_throughput > 0:
        used_capacity1 += offspring1.genes[i].bitrate/offspring1.genes[i].prb_throughput
        fitness1 += offspring1.genes[i].mos
      if offspring2.genes[i].rid and offspring2.genes[i].prb_throughput > 0:
        used_capacity2 += offspring2.genes[i].bitrate/offspring2.genes[i].prb_throughput
        fitness2 += offspring2.genes[i].mos

    if used_capacity1 <= self.scenario["nprb"]:
        ok1 = True
    if used_capacity2 <= self.scenario["nprb"]:
        ok2 = True

    offspring1.fitness = fitness1
    offspring2.fitness = fitness2

    if ok1:
      offsprings.append(offspring1)
    if ok2:
      offsprings.append(offspring2)
    return offsprings

  def mutation(self):
    """Mutation operator.
    
    For each chromosome in the solution pool, decide according to the mutation
    rate if we'll modify it or not. If it's selected for mutation, we select a random
    user and change its video representation assignment.
    """

    for s in self.solution_pool:
      if uniform(0, 1) <= self.mutation_rate:
        logging.debug("Mutating solution: " + str(s))
        #u = choice(s["users"])
        u = choice(s.genes)

        # find which representations can be supported by the user
        admissible_reps = copy.copy(sorted(self.scenario["representations"], key=itemgetter('bitrate')))
        admissible_reps[:] = [r for r in admissible_reps if r["bitrate"] <= u.max_throughput]

        # if the user cannot admit any video, do nothing
        if not admissible_reps:
          continue

        # Find a feasible *different* representation
        # pick a random representation from the ones admissible by the user   
        # TODO: looks ugly, refactor 
        r = randint(0, len(admissible_reps))
        # assumption/hack: the "virtual" representation #len corresponds to no video
        if r == len(admissible_reps):
          if u.rid is None:
            pass
          else:
            # otherwise, remove the assigned representation (aka switch user video off)
            s.fitness -= u.mos # update fitness value in place, to save time
            u.rid = None
            u.bitrate = None
            u.mos = 0.0
        else:
          # drew an existing representation. if it's different than the one the user already has, 
          # assign it. Otherwise, keep loopin.
          if u.rid is not None and u.rid == admissible_reps[r]["id"]:
            pass
          else:
            # Done with this guy. Move to next chromosome/solution
            # check if the new representation violates the constraint and if so undo
            if u.rid is not None:
              s.fitness -= u.mos
              old_rid = u.rid
              old_b = u.bitrate
              old_m = u.mos
            else:
              old_rid = -1
            u.rid = admissible_reps[r]["id"]
            u.bitrate = admissible_reps[r]["bitrate"]
            u.mos = admissible_reps[r]["mos"]
            s.fitness += u.mos

            if not self.check_capacity_constraint(s):
              # undo if the mutant is not a feasible solution
              if old_rid == -1:
                s.fitness -= u.mos
                u.rid = None
                u.bitrate = None
                u.mos = 0.0
              else:
                s.fitness = s.fitness - u.mos + old_m
                u.rid = old_rid
                u.bitrate = old_b
                u.mos = old_m
        s.fitness = self.fitness(s)

  def generation(self):
    """A GA generation to produce a new solution pool.
    
    This executes crossover and mutation operations to produce
    offspring and keeps the top-ranking solutions according to the
    fitness function.
    
    It also returns the fitness function of the best solution
    """ 
    # select number of offspring
    nOffspring = int(self.crossover_rate * self.solution_pool_size)
    
    # for each offspring
    while nOffspring > 0:
      # generate a new chromosome from crossover
      offspring = self.crossover_multi()
      for o in offspring:
        self.solution_pool.append(o)
      nOffspring -= 2

    # mutation
    self.mutation()
        
    # select top-S chromosomes according to their fitness value 
    # and create new solution pool
    #for s in self.solution_pool:
      ## update fitness values of all candidate solutions
      #s["fitness"] = self.fitness(s)
    
    # keep the top-S solutions
    self.solution_pool = sorted(self.solution_pool, key=attrgetter('fitness'), reverse = True)[0:self.solution_pool_size]
    
    return self.solution_pool[0].fitness


  def execute(self):
    """Run the genetic algorithm.
    
    Returns the scenario JSON object with placement decisions plus some 
    algorithm-specific information.
    """

    # check if we will store data per generation in a file
    log_generation_data = False
    if self.generations_file is not None:
      try:
        gen_fp = open(self.generations_file, "w+")
        log_generation_data = True
        gen_fp.write("# Scenario: " + str(self.scenario_file) + "\n")
        gen_fp.write("# Seed: " + str(self.seed) + "\n")
        gen_fp.write("#-----------------------------------------------\n")
        gen_fp.write("# Generation\tFitness\t\tTimestamp\n")
      except:
        logging.warn("Error opening/writing at " + self.generations_file)
        pass
        
    prev_obj_value = -1
    if self.convergence_check: 
      remaining_generations = self.stop_after

    # Keep track of the best solution we've seen, in case we arrive at a worse solution when the algorithm terminates
    current_best = None

    start_time = datetime.now()

    # Create initial solution pool    
    self.init_solution_pool_v2()

    for i in range(0, self.generations):
      obj_value = self.generation()

      # get a timestamp for this generation
      dt = datetime.now() - start_time
      # convert to seconds. dt.days should really not matter...
      time_taken = dt.days*24*3600 + dt.seconds + dt.microseconds/1000000.0

      logging.info("Generation/fitness/timestamp: " + str(i) + "\t" + str(obj_value) + "\t" + str(time_taken))
      if log_generation_data:
        gen_fp.write(str(i) + "\t\t" + str(obj_value) + "\t" + str(time_taken) + "\n")
      
      # maintain best solution we've seen thus far
      if current_best is None or current_best.fitness <= self.solution_pool[0].fitness:
          current_best = chromosome.copy(self.solution_pool[0]) #copy.deepcopy(self.solution_pool[0])

      # if we're checking for convergence to finish execution faster
      # we have to do some checks
      if self.convergence_check:
        if abs(obj_value - prev_obj_value) < self.delta:
          # the solution fitness did not significantly change
          remaining_generations -= 1
        else:
          remaining_generations = self.stop_after
        
        # the algorithm converged
        if remaining_generations < 0:
          break
        prev_obj_value = obj_value
    #final_solution = self.solution_pool[0]
    final_solution = current_best

    # convert from chromosome to json
    retval = self.from_chromosome(final_solution)
    # add extra information about solution performance (cost, availability, latency, time taken, # generations)
    # and indications about constraint violations
    info = {}
    info["generations"] = i + 1
    info["execution_time"] = time_taken
    info["used_capacity"] = self.get_used_capacity(final_solution)

    # some final checks
    retval["solution_performance"] = info

    return retval

  def from_chromosome(self, s):
    output = copy.deepcopy(self.scenario)
    for i in range(len(output["users"])):
      if s.genes[i] is not None and s.genes[i].rid is not None:
        output["users"][i]["rid"] = s.genes[i].rid
        output["users"][i]["mos"] = s.genes[i].mos
        output["users"][i]["bitrate"] = s.genes[i].bitrate
    output["fitness"] = s.fitness
    return output

