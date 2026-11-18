"""Continuous demonstration windows; frame ends and anchor ends are inclusive."""
import numpy as np


def build_window(rows, meta, close, opening=None, fps=10, chunk_size=15):
    if [r['frame_index'] for r in rows] != list(range(len(rows))):
        raise ValueError('Frames are not contiguous')
    stable = next((i for i in range(close+4, len(rows))
                   if all(r.get('physical_grasp_ok') and r.get('grasp_success')
                          and not r.get('object_touching_table') for r in rows[i-4:i+1])), None)
    if stable is None:
        raise ValueError('No five-frame stable physical grasp')
    stages = [s for s in meta['stages'] if 'start_sim_time' in s]
    if stages:
        begin_time = min(s['start_sim_time'] for s in stages) - .5
        start = next((i for i,r in enumerate(rows) if r['simulation_time'] >= begin_time), len(rows))
    else:
        start = max(0, close-int(fps))
    requested_end = stable + int(1.5*fps)
    end = min(len(rows)-1, requested_end, opening-1 if opening is not None else len(rows)-1)
    if not (start < close <= stable <= end and end-start+1 >= chunk_size):
        raise ValueError('Invalid correction/grasp/lift window')
    return dict(frame_start=start, frame_end_inclusive=end, anchor_start=start,
                anchor_end_inclusive=end-(chunk_size-1), close_frame=close,
                stable_confirmed_frame=stable, open_frame=opening,
                hold_frames_after_confirmation=end-stable,
                hold_truncated=end<requested_end, chunk_size=chunk_size,
                frame_count=len(rows), kept_frames=end-start+1,
                eligible_anchor_count=end-(chunk_size-1)-start+1)


def anchor_mask(episode, frame, records, training=True):
    episode=np.asarray(episode);frame=np.asarray(frame)
    if episode.shape != frame.shape:
        raise ValueError('Episode/frame shapes differ')
    if len({r['episode_index'] for r in records}) != len(records):
        raise ValueError('Duplicate episode IDs')
    if set(np.unique(episode)) - {r['episode_index'] for r in records}:
        raise ValueError('Unmapped episode IDs')
    mask=np.zeros(episode.shape,dtype=bool)
    for r in records:
        if training and r['split'] != 'train':
            continue
        mask |= ((episode==r['episode_index']) & (frame>=r['anchor_start'])
                 & (frame<=r['anchor_end_inclusive']))
    return mask
