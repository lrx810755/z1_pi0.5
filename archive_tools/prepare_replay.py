"""Instantiate an archived matched-seed evaluation without launching it."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
RECIPES = {
    'filtered-15': 'smolvla_eval/pi05_final_execute15_finger_table_filtered_30each_20261118',
    'filtered-20': 'smolvla_eval/pi05_final_filtered_execute20_30_30each_20261118/execute_20',
    'filtered-30': 'smolvla_eval/pi05_final_filtered_execute30_15each_20261118',
}
CODE = (
    'scene_filtered.py', 'entry_filtered.py', 'run_job.py', 'ros_env.sh',
    'runtime_rollout.py', 'penetration.py', 'baseline.json',
)


def prepare(recipe, output):
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    source = ROOT / RECIPES[recipe] / 'job'
    request = json.loads((source / 'request.json').read_text())
    former_output = request['output']
    command_files = ('scene_command.json', 'evaluation_command.json')
    commands = {name: json.loads((source / name).read_text()) for name in command_files}
    if former_output == str(output):
        raise ValueError('Replay output must have a new path')
    request['output'] = str(output)
    job = output / 'job'
    job.mkdir(parents=True, exist_ok=False)
    for filename in CODE:
        original = source / filename
        if filename == 'penetration.py' and not original.exists():
            continue
        shutil.copy2(original, job / filename)
    for name, command in commands.items():
        new_command = [arg.replace(former_output, str(output)) for arg in command]
        (job / name).write_text(json.dumps(new_command, ensure_ascii=False, indent=2) + '\n')
    (job / 'request.json').write_text(json.dumps(request, ensure_ascii=False, indent=2) + '\n')
    return job / 'run_job.py'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recipe', choices=RECIPES)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    path = prepare(args.recipe, args.output)
    print(path)
    print('Use the restored PI0.5 Python environment to run this file after the model and ROS checks.')


if __name__ == '__main__':
    main()
