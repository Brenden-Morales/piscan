# -*- coding: utf-8 -*-
"""
* Checkbox question example
* run example by typing `python example/checkbox.py` in your console
"""
from pprint import pprint
from PyInquirer import prompt, Separator
from cli_prompts import CliPrompts


selected_hosts = prompt(CliPrompts.get_hosts_prompt("camera_configs.yaml"))
operation = prompt(CliPrompts.get_operation_prompt())
while operation["operation"] != "Quit":
    operation = prompt(CliPrompts.get_operation_prompt())