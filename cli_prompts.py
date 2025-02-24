from yaml import safe_load, YAMLError

class CliPrompts:
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
    def get_operation_prompt():
        return [
            {
                'type': 'list',
                'name': 'operation',
                'message': 'What do you want to do?',
                'choices': [
                    'Take snapshot',
                    'Start project',
                    'Set configs',
                    'Select Hosts',
                    'Quit'
                ]
            }
        ]