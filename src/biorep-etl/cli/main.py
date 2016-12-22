#!/usr/bin/env python
"""Provide command line interface to biorep-etl."""

# Imports
import os
from pathlib import Path
import datetime as dt
import shutil
import pprint as pp


from munch import Munch, munchify, unmunchify
import ruamel.yaml as yaml

import click
from click import echo

import biorep-etl.cli.config as _config
import biorep-etl.errors as e


# Metadata
__author__ = "Gus Dunn"
__email__ = "w.gus.dunn@gmail.com"

HOME_DIR = (Path(os.path.realpath(__file__)).parent / '../../..').resolve()

def update_configs(directory, to_update=None):
    """Collect, combine, and return all *.yaml files in `directory`."""
    confs = Path(directory).glob('*.yaml')

    confs = {p.stem.upper(): p for p in confs}

    if to_update is None:
        to_update = Munch()


    for name, conf in confs.items():
        c = process_config(config=conf)
        to_update.update(Munch({name: c}))

    return to_update


def process_config(config=None):
    """Prepare single config file."""
    if config is None:
        return Munch()
    else:
        return munchify(yaml.load(config.open()))



@click.group(invoke_without_command=True)
@click.option('-c', '--config', default=None,
              help="Path to optional config directory. If `None`, configs/ is searched for *.yaml files.",
              type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--home', default=None,
              help="Print the home directory of the install and exit.",
              is_flag=True)
@click.pass_context
def run(ctx=None, config=None, home=None):
    """Command interface to the veoibd-synapse-manager.

    For command specific help text, call the specific
    command followed by the --help option.
    """
    ctx.obj = Munch()
    ctx.obj.CONFIG = Munch()

    top_lvl_confs = HOME_DIR / 'configs'

    ctx.obj.CONFIG = update_configs(directory=top_lvl_confs, to_update=ctx.obj.CONFIG)

    if config:
        ctx.obj.CONFIG = update_configs(directory=config, to_update=ctx.obj.CONFIG)

    if home:
        print(HOME_DIR)
        exit(0)




@run.command()
@click.option("-l", "--list", "list_",
              is_flag=True,
              default=False,
              help="Print the configuration values that will be used and exit.")
@click.option('-g', '--generate-config',
              is_flag=True,
              help="Copy one or more of the 'factory default' config files to the top-level "
              "config directory. Back ups will be made of any existing config files.",
              show_default=True,
              default=False)
@click.option('-k', '--kind',
              type=click.Choice(['all', 'main']),
              help="Which type of config should we replace?",
              show_default=True,
              default='all')
@click.pass_context
def configs(ctx, list_, generate_config, kind):
    """Manage configuration values and files."""
    if list_:
        conf_str = yaml.dump(unmunchify(ctx.obj.CONFIG), default_flow_style=False)
        echo(conf_str)
        exit(0)

    base_dir = HOME_DIR / 'configs'
    factory_resets = Path('configs/factory_resets')
    default_files = {"all": factory_resets.glob('*.yaml'),
                     "main": factory_resets / 'main.yaml',
                     }

    if kind == 'all':
        for p in default_files['all']:
            _config.replace_config(name=p.name, factory_resets=factory_resets)
    else:
        p = default_files[kind]
        _config.replace_config(name=p.name, factory_resets=factory_resets)









# Business
if __name__ == '__main__':
    run(obj=Munch())
