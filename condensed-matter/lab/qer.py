#/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import sys
import yaml
import time
import string
import random
import itertools
from subprocess import Popen, PIPE, STDOUT
import logging

CONFIG_DIR = 'config'

logging.basicConfig(level=logging.INFO)

print('''
                    ___           ___
                   /\__\         /\  \\
      ___         /:/ _/_       /::\  \\
     /\  \       /:/ /\__\     /:/\:\__\\
    /::\  \     /:/ /:/ _/_   /:/ /:/  /
   /:/\:\  \   /:/_/:/ /\__\ /:/_/:/__/___
  /:/ /::\  \  \:\/:/ /:/  / \:\/:::::/  /
 /:/_/:/\:\__\  \::/_/:/  /   \::/~~/~~~~
 \:\/:/  \/__/   \:\/:/  /     \:\~~\\
  \::/  /         \::/  /       \:\__\\
   \/__/           \/__/         \/__/

       ~ Quantum Espresso Runner ~

''')

#######################################
# Setup                               #
#######################################

# Parse the configuration files
config_file = os.path.join(CONFIG_DIR, 'config.yaml')

with open(config_file) as c:
    try:
        config = yaml.load(c)
        logging.debug("Config parsed correctly")
    except yaml.YAMLError:
        log.critical("There is some problem with config.yaml.")
        exit(1)

# Create the output directory if it does not exist
if not os.path.exists(config['output_dir']):
    os.makedirs(config['output_dir'])

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
        self._print_banner("Running step {}".format(step.name))

        output_path = os.path.join(self.out_dir, "{}.out".format(step.name))
        output_file = open(output_path, "w")
        exec_path = os.path.join(self.bin_dir, step.command)

        logging.debug("Running command `{}`".format(exec_path))
        process = Popen(exec_path, stdin=PIPE, stdout=output_file)
        process.stdin.write(step.input)
        process.stdin.close()
        self._spinner(process.poll)

        if process.returncode != 0:
            logging.critical("Run failed for step {}".format(step.name))
            exit(1)

        print("Done!")

    def _print_banner(self, text):
        print("""
============================================
= {:<40} =
============================================
""".format(text))

    def _spinner(self, poll):
        spinner = itertools.cycle(['.  ', '.. ', '...'])

        messages = ["Thinking about spherical cows",
                    "What about spherical sheeps? ",
                    "Dreaming about spherical cows"]

        sys.stdout.write(random.choice(messages))

        while poll() is None:
            sys.stdout.write(spinner.next())
            sys.stdout.flush()
            sys.stdout.write('\b\b\b')
            time.sleep(0.5)



runner = StepRunner(config['qe_bin_dir'], config['output_dir'])

for step in steps:
    runner.run(step)
