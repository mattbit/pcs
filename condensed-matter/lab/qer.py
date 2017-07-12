#/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import sys
import yaml
import string
from subprocess import Popen, PIPE, STDOUT
import logging

CONFIG_DIR = 'config'

logging.basicConfig(level=logging.DEBUG)

print("""
                    ___           ___
                   /\__\         /\  \
      ___         /:/ _/_       /::\  \
     /\  \       /:/ /\__\     /:/\:\__\
    /::\  \     /:/ /:/ _/_   /:/ /:/  /
   /:/\:\  \   /:/_/:/ /\__\ /:/_/:/__/___
  /:/ /::\  \  \:\/:/ /:/  / \:\/:::::/  /
 /:/_/:/\:\__\  \::/_/:/  /   \::/~~/~~~~
 \:\/:/  \/__/   \:\/:/  /     \:\~~\
  \::/  /         \::/  /       \:\__\
   \/__/           \/__/         \/__/

       ~ Quantum Espresso Runner ~
""")

#######################################
# Setup                               #
#######################################

# Create the output directory if it does not exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Parse the configuration files
config_file = os.path.join(CONFIG_DIR, 'config.yaml')

with open(config_file) as c:
    try:
        config = yaml.load(c)
        logging.debug("Config parsed correctly")
    except yaml.YAMLError:
        log.critical("There is some problem with config.yaml.")
        exit(1)

class Step(object):
    def __init__(self, name, command, input_file, vars={}):
        self.name = name
        self.command = command
        # Parse immediately the configuration so if there are errors
        # we can stop immediately.
        self._parse_input(input_file, vars)

    def _parse_input(self, input_file, vars):
        with open(os.path.join(CONFIG_DIR, input_file)) as f:
            t = string.Template(f.read())
            self.input = t.substitute(vars)


steps = []
for step in config['steps']:
    if not 'input_file' in step:
        step['input_file'] = "{}.in".format(step['name'])

    steps.append(Step(step['name'], step['command'], step['input_file'],
                      config['vars']))


class StepRunner(object):
    def __init__(self, bin_dir, out_dir):
        self.bin_dir = bin_dir
        self.out_dir = out_dir

    def run(self, step):
        logging.info("Running step {}".format(step.name))

        output_path = os.path.join(self.out_dir, "{}.out".format(step.name))
        output_file = open(output_path, "w")
        exec_path = os.path.join(self.bin_dir, step.command)

        logging.debug("Running command `{}`".format(exec_path))
        process = Popen(exec_path, stdin=PIPE, stdout=output_file, stderr=STDOUT)
        process.communicate(step.input)
        process.wait()

        if process.returncode != 0:
            log.critical("Run failed for step {}".format(step.name))
            exit(1)

    def _print_banner(self, text):
        print("""
============================================
= {:<40} =
============================================
""".format(text))

runner = StepRunner(config['qe_bin_dir'], config['output_dir'])

for step in steps:
    runner.run(step)
