"""Export joint/gripper traces and final camera views for the prior failed layout."""
import json
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
SEED = 882884764
COLORS = {'baseline': '#555555', 'uniform': '#2271b2', 'weighted': '#e69f00'}


def main():
    summary = json.loads((ROOT / 'comparison.json').read_text())
    figure, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    camera_rows = []
    for arm, color in COLORS.items():
        output = Path(summary['outputs'][arm])
        episodes = list((output / 'episodes').glob(f'*seed_{SEED}'))
        assert len(episodes) == 1
        episode = episodes[0]
        rows = [json.loads(line) for line in (episode / 'trace.jsonl').read_text().splitlines()]
        for ax, index, title in zip(axes, [3, 5, 6], ['Joint 4', 'Joint 6', 'Gripper']):
            time = [r['simulation_elapsed_s'] for r in rows]
            command = [r['command'][index] for r in rows]
            measured = [r['joint_state'][index] for r in rows]
            ax.plot(time, command, color=color, label=arm+' command')
            ax.plot(time, measured, color=color, linestyle=':', alpha=.7)
            ax.set_ylabel(title+' (rad)')
            ax.grid(alpha=.2)
        frames = []
        for camera in ['global_third_person', 'wrist']:
            capture = cv2.VideoCapture(str(episode / 'videos' / (camera+'.mp4')))
            assert capture.isOpened()
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, capture.get(cv2.CAP_PROP_FRAME_COUNT)-1))
            ok, frame = capture.read()
            capture.release()
            assert ok
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (640, int(h*640/w)))
            panel = np.zeros((540, 640, 3), np.uint8)
            panel[45:45+frame.shape[0]] = frame
            cv2.putText(panel, arm+' / '+camera, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 2)
            frames.append(panel)
        camera_rows.append(np.concatenate(frames, axis=1))
    axes[-1].axhline(-.75, color='#bb0000', linestyle='--', linewidth=1, label='close threshold')
    axes[0].legend(loc='best')
    axes[-1].legend(loc='best')
    axes[-1].set_xlabel('Simulation time since policy start (s)')
    figure.suptitle(f'Paired bottle layout {SEED}; dotted lines = measured joints')
    figure.tight_layout()
    figure.savefig(ROOT / 'paired_joint_traces.png', dpi=160)
    plt.close(figure)
    assert cv2.imwrite(str(ROOT / 'paired_final_camera_views.jpg'), np.concatenate(camera_rows, axis=0))
    print('Saved paired_joint_traces.png and paired_final_camera_views.jpg')


if __name__ == '__main__':
    main()
