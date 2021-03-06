# Materials for the AECM lab

This folder contains resources and configurations used for the Advanced Experiments in Condensed Matter lab, concerning the study of a thin sample of diamond.


The material mainly consists in a python script called **qer** (Quantum Espresso Runner) that facilitates the management of multiple overlapping configurations for Quantum Espresso, and some scripts used for fitting and plotting the data obtained by resistivity measurements.

## Quantum Espresso Runner (QER)

The material is located in the `qer` folder. Usage is briefly outlined below.

```sh
# Run qer using the specified configuration file
python qer.py -c config/config.yaml

# Dry run: only parses the configuration files
python qer.py -d -c config/config.yaml

```

The YAML configuration files contain information about which Quantum Espresso commands to run, and may define variables that will be substituted in the various `[command].in` files (with python string templates syntax).


## Resistivity data

Located in the `resistivity` folder. Includes the raw measured data.

The script `resistivity.py` extracts the data and plots the resistivity and conductivity curves, while `fits.py` is used to calculate the appropriate fitting.
