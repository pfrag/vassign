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
Invocation of the GA for video quality assignment.
#################################################################
"""

from ga import *
import sys
import getopt
import yaml
import json

def _load_configuration(configfile):
  """Loads configuration options from the configfile path (yaml-formatted).
  """

  try:
      f = open(configfile)
  except:
      print("Could not open configuration file " + configfile)
      exit(2)
      
  try:
      configuration = yaml.load(f)
  except:
      print("Error loading configuration file. Possibly malformed...")
      exit(3)
  return configuration


if __name__ == '__main__':    
  # open configuration file
  myopts, args = getopt.getopt(sys.argv[1:], "c:")
  configfile = None
  for o, a in myopts:
    if o == "-c":
      configfile = a

  if configfile == None:
    print("Missing configuration file. Usage: python main.py -c /path/to/configfile")
    exit(1)

  # load algorithm-specific configuration/settings
  configuration = _load_configuration(configfile)

  g = GA(configuration)
  solution = g.execute()
  #print(json.dumps(solution, indent=2))

