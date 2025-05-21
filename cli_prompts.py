from yaml import safe_load, YAMLError
from PyInquirer import prompt

class CliPrompts:
    @staticmethod
    def true_false_prompt():
        print('true / false prompt')
        true_false_prompt = [
            {
                'type': 'confirm',
                'message': 'On / Off?',
                'name': 'onOff',
                'default': True,
            }
        ]
        return prompt(true_false_prompt)["onOff"]

    @staticmethod
    def single_list_prompt(items):
        items_prompt = [
            {
                'type': 'list',
                'name': 'control',
                'message': 'Select one',
                'choices': items
            }
        ]
        return prompt(items_prompt)["control"]

    @staticmethod
    def value_range_prompt(range):
        range_prompt = [
            {
                'type': 'input',
                'name': 'value',
                'message': 'min:{}, max:{}, current:{}'.format(range[0], range[1], range[2]),
            }
        ]
        value = prompt(range_prompt)['value']
        print(range[-1])
        if isinstance(range[-1], int):
            return int(value)
        if isinstance(range[-1], float):
            return float(value)
        return prompt(range_prompt)['value']

    @staticmethod
    def get_hosts_prompt(config_path):
        camera_configs = None
        with open(config_path) as stream:
            try:
                camera_configs = safe_load(stream)
            except YAMLError as exc:
                print(exc)

        config_choices = []
        for config in camera_configs:
            config_choices.append({
                'name': config["host"]
            })

        return [
            {
                'type': 'checkbox',
                'qmark': '😃',
                'message': 'Select Hosts to control',
                'name': 'Hosts',
                'choices': config_choices,
                'validate': lambda answer: 'You must choose at least one Host.' \
                    if len(answer) == 0 else True
            }
        ]

    @staticmethod
    def list_controls_prompt(camera_controls):
        choices = []
        for control_name, control_values in camera_controls.items():
            choices.append("{}: {}".format(control_name, control_values))
        choices.append("Back")

        controls_prompt = [
            {
                'type': 'list',
                'name': 'control',
                'message': 'What control do you wish to change?',
                'choices': choices
            }
        ]
        control_to_edit = prompt(controls_prompt)
        control_name = control_to_edit["control"].split(":")[0]
        print(control_name)
        if control_name == "Back":
            return (None, None)
        elif control_name == "AeEnable":
            value = CliPrompts.true_false_prompt()
            return (control_name, value)
        else:
            value = CliPrompts.value_range_prompt(camera_controls[control_name])
            return (control_name, value)

    @staticmethod
    def get_operation_prompt():
        return [
            {
                'type': 'list',
                'name': 'operation',
                'message': 'What do you want to do?',
                'choices': [
                    'Take snapshot',
                    'Start project',
                    'Set Controls',
                    'List Controls',
                    'Select Hosts',
                    'MAGIC',
                    'Quit'
                ]
            }
        ]