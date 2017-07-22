#/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import sys
import yaml
import time
import string
import random
import getopt
import itertools
from subprocess import Popen, PIPE, STDOUT
import logging

CONFIG_DIR = 'config'
CONFIG_FILE = 'config/config.yaml'
DRY_RUN = False

options, remainder = getopt.getopt(sys.argv[1:], 'c:l:d', ['config=', 'log=', 'dry-run'])

for opt, arg in options:
    if opt in ('-c', '--config'):
        CONFIG_DIR = os.path.dirname(arg)
        CONFIG_FILE = arg
    elif opt in ('-l', '--log'):
        logging.basicConfig(level=getattr(logging, arg.upper()))
    elif opt in ('-d', '--dry-run'):
        DRY_RUN = True

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

# Default config from environment
config = {
    'qe_bin_dir': os.getenv('QUANTUM_ESPRESSO_BIN_DIR'),
    'qe_potentials_dir': os.getenv('QUANTUM_ESPRESSO_POTENTIALS_DIR'),
}

# Parse the configuration files
with open(CONFIG_FILE) as c:
    try:
        user_config = yaml.load(c)
        logging.debug("Config parsed correctly")
    except yaml.YAMLError:
        logging.exception("There is some problem with config.yaml.")
        exit(1)

# Merge the config
for k in user_config:
    config[k] = user_config[k]

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

    parsed = os.path.join(config['output_dir'], "{}.in.parsed".format(step.name))
    with open(parsed, "w") as f:
        f.write(step.input)

    if not DRY_RUN:
        runner.run(step)
