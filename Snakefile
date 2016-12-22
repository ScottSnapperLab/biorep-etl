"""Snakemake file."""
import os

from pathlib import Path

import yaml

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

import munch




def pathify_by_key_ends(dictionary):
    """Return a dict that has had all values with keys marked as '*_PATH' or '*_DIR' converted to Path() instances."""
    for key, value in dictionary.items():
        if isinstance(value, dict):
            pathify_by_key_ends(value)
        elif key.endswith("_PATH") or key.endswith("_DIR"):
            dictionary[key] = Path(value)

    return dictionary


class MyRun(object):

    """Initialize and manage information common to the whole run."""

    def __init__(self, cfg):
        """Initialize common information for a run."""
        assert isinstance(cfg, dict)

        common = cfg["COMMON"]

        self.globals = munch.Munch()
        self.cfg = cfg
        self.name = common["RUN_NAME"]
        self.d = common["SHARED"]
        self.out_dir = Path("{base_dir}/{run_name}".format(base_dir=common["OUT_DIR"],
                                                           run_name=self.name
                                                           )
                            )
        self.log_dir = self.out_dir / "logs"

class MyRule(object):

    """Manage the initialization and deployment of rule-specific information."""

    def __init__(self, run, name):
        """Initialize logs, inputs, outputs, params, etc for a single rule."""
        assert isinstance(run, MyRun)

        self.run = run
        self.name = name.lower()
        self.log_dir = run.log_dir / self.name
        self.log = self.log_dir / "{name}.log".format(name=self.name)
        self.out_dir = run.out_dir / self.name
        self.i = munch.Munch() # inputs
        self.o = munch.Munch() # outputs
        self.p = munch.Munch() # params

        self._import_config_dict()

    def _import_config_dict(self):
        """Inport configuration values set for this rule so they are directly accessable as attributes."""
        try:
            for key, val in self.run.cfg[self.name.upper()].items():
                self.__setattr__(key, val)
            self.cfg = True
        except KeyError:
            self.cfg = False



#### COMMON RUN STUFF ####
ORIGINAL_CONFIG_AS_STRING = yaml.dump(config, default_flow_style=False)
config = pathify_by_key_ends(config)
config = munch.munchify(config)

RUN = MyRun(cfg=config)

PRE = []
ALL = []

# add specific useful stuff to RUN
# RUN.globals.fam_names = [vcf.stem for vcf in config.VALIDATE_INPUT_VCFS.IN.VCF_DIR.glob("*.vcf")]



############ BEGIN PIPELINE RULES ############
# ------------------------- #
#### SAVE_RUN_CONFIG ####
SAVE_RUN_CONFIG = MyRule(run=RUN, name="SAVE_RUN_CONFIG")
SAVE_RUN_CONFIG.o.file = RUN.out_dir / "{NAME}.yaml".format(NAME=RUN.name)



rule save_run_config:
    input:
    output:
        file=str(SAVE_RUN_CONFIG.o.file)

    run:
        with open(output.file, 'w') as cnf_out:
            cnf_out.write(ORIGINAL_CONFIG_AS_STRING)

PRE.append(rules.save_run_config.output)
ALL.append(rules.save_run_config.output)



############ BEGIN PIPELINE RULES ############


#### SAVE_RUN_CONFIG ####
SAVE_RUN_CONFIG_OUT = OUT_DIR+"/{RUN_NAME}.yaml".format(RUN_NAME=RUN_NAME)

rule save_run_config:
    input:
    output:
        file=SAVE_RUN_CONFIG_OUT

    run:
        with open(output.file, 'w') as cnf_out:
            cnf_out.write(ORIGINAL_CONFIG_AS_STRING)

ALL.append(rules.save_run_config.output)


# ------------------------- #
#### RULE_PYSCRIPT ####
RULE_PYSCRIPT = config["RULE_PYSCRIPT"]

# log
LOG_RULE_PYSCRIPT = LOGS_DIR+"/rule_pyscript.log"

# params
SETTING1 = RULE_PYSCRIPT["SETTING1"]

# inputs
IN_FILE1 = RULE_PYSCRIPT["IN_FILE1"]
IN_FILE2 = RULE_PYSCRIPT["IN_FILE2"]

# outputs
RULE_PYSCRIPT_OUT = OUT_DIR+"/RULE_PYSCRIPT"

OUT_FILE1 = RULE_PYSCRIPT_OUT+'/out_file1.csv'
OUT_FILE2 = RULE_PYSCRIPT_OUT+'/out_file2.txt'


# ---
rule rule_pyscript:
    params:
        setting1=SETTING1,

    input:
        in_file1=IN_FILE1,
        in_file2=IN_FILE2
    output:
        out_file1=OUT_FILE1,
        out_file2=OUT_FILE2,

    script:
        "python/scripts/rule_pyscript.py"
ALL.append(rules.rule_pyscript.output)


# ------------------------- #
#### RULE_RSCRIPT ####
RULE_RSCRIPT = config["RULE_RSCRIPT"]

# log
LOG_RULE_RSCRIPT = LOGS_DIR+"/rule_rscript.log"

# params
SETTINGA = RULE_RSCRIPT["SETTINGA"]

# inputs
IN_FILEA = RULE_RSCRIPT["IN_FILEA"]
IN_FILEB = RULE_RSCRIPT["IN_FILEB"]

# outputs
RULE_RSCRIPT_OUT = OUT_DIR+"/rule_rscript"

OUT_FILEA = RULE_RSCRIPT_OUT+'/out_filea.csv'
OUT_FILEB = RULE_RSCRIPT_OUT+'/out_fileb.txt'


# ---
rule rule_rscript:
    params:
        settinga=SETTINGA,

    input:
        in_filea=IN_FILEA,
        in_fileb=IN_FILEB
    output:
        out_filea=OUT_FILEA,
        out_fileb=OUT_FILEB,

    script:
        "r/scripts/rule_rscript.R"
ALL.append(rules.rule_rscript.output)

# ------------------------- #
#### RULE_SHELL_CMD ####
RULE_SHELL_CMD = config["RULE_SHELL_CMD"]

# log
LOG_RULE_SHELL_CMD = LOGS_DIR+"/rule_shell_cmd.log"

# params
SETTING_1 = RULE_SHELL_CMD["SETTING_1"]

# input
IN_FILE_1 = RULE_SHELL_CMD["IN_FILE_1"]
IN_FILE_2 = RULE_SHELL_CMD["IN_FILE_2"]

# output
RULE_SHELL_CMD_DIR = OUT_DIR+"/rule_shell_cmd"
RULE_SHELL_CMD_OUT = RULE_SHELL_CMD_DIR+"/rule_shell_cmd.txt"

# ---
rule rule_shell_cmd:
    log:
        path=LOG_RULE_SHELL_CMD

    params:
        setting_1=SETTING_1,

    input:
        in_file_1=IN_FILE_1,
        in_file_2=IN_FILE_2,

    output:
        rule_shell_cmd_out=RULE_SHELL_CMD_OUT,

    shell:
        """ \
        cat {params.setting_1} \
        {input.in_file_1} \
        {input.in_file_2} \
        > {output.rule_shell_cmd_out} \
        &> {log.path}
        """

ALL.append(rules.rule_shell_cmd.output)


# ------------------------- #


#### ALL ####
# ---
rule all:
    input: ALL
