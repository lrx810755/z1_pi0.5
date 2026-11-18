"""Read immutable peach states/actions and exact predecoded float32 images."""
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import os
import json
import numpy as np
import torch
from common import RUN, CACHE, CAMERAS, HORIZON, read
from codec import unpack_frame


class PeachDataset(torch.utils.data.Dataset):
    def __init__(self, episodes=None, raw_images=False):
        self.records=read(RUN/'split_manifest.json')['source_records']
        self.episodes=list(range(len(self.records))) if episodes is None else list(episodes)
        self.raw_images=raw_images
        with np.load(RUN/'lowdim.npz') as archive:
            self.arrays={k:archive[k] for k in archive.files}
        with np.load(RUN/'sampling_arrays.npz') as archive:
            self.approved=archive['eligible_train']
        self.indices=np.flatnonzero(np.isin(self.arrays['episode'],self.episodes))
        self.absolute_to_relative_idx={int(x):i for i,x in enumerate(self.indices)}
        self.num_frames=len(self.indices);self.num_episodes=len(self.episodes);self.fps=10;self.root=RUN
        names=[f'joint{i}' for i in range(1,7)]+['jointGripper']
        features={k:dict(dtype='float32',shape=[7],names=names) for k in ['observation.state','action']}
        for cam in CAMERAS:
            features[cam]=dict(dtype='image',shape=[360 if cam.endswith('wrist') else 480,640,3],names=['height','width','channels'])
        stats={k:{name:np.asarray(value,dtype=np.float32) for name,value in item.items()} for k,item in read(RUN/'train_stats.json').items()}
        self.meta=SimpleNamespace(features=features,stats=stats,camera_keys=CAMERAS,has_language_columns=False,
                                  fps=10,robot_type='unitree_z1',repo_id='lp531/peach397_act_20261118',
                                  episodes={k:[r[k] for r in self.records] for k in ['dataset_from_index','dataset_to_index']})
        if not raw_images and not (CACHE/'completed.json').exists():
            raise ValueError('Full cache must be decoded before training')

    def __len__(self):return self.num_frames

    def __getitem__(self,index):
        absolute=int(self.indices[index]);ep=int(self.arrays['episode'][absolute]);frame=int(self.arrays['frame'][absolute]);record=self.records[ep]
        if not self.approved[absolute] or absolute+HORIZON>record['dataset_to_index']:
            raise ValueError('Require approved complete30-step expert window')
        if not np.all(self.arrays['episode'][absolute:absolute+HORIZON]==ep):
            raise ValueError('Action window crosses episode')
        result={'observation.state':torch.from_numpy(self.arrays['states'][absolute].copy()),
                'action':torch.from_numpy(self.arrays['actions'][absolute:absolute+HORIZON].copy()),
                'action_is_pad':torch.zeros(HORIZON,dtype=torch.bool),'episode_index':torch.tensor(ep),
                'frame_index':torch.tensor(frame),'index':torch.tensor(absolute),'timestamp':torch.tensor(frame/10.),
                'task':record['task'],'task_index':torch.tensor(0)}
        if getattr(self,'_pid',None)!=os.getpid():
            self._maps=OrderedDict();self._rows=OrderedDict();self._pid=os.getpid()
        if self.raw_images:
            if ep not in self._rows:
                if len(self._rows)>=8:self._rows.popitem(last=False)
                self._rows[ep]=[json.loads(line) for line in (RUN/'source_pins'/f'episode_{ep:06d}'/'frames.jsonl').read_text().splitlines()]
            self._rows.move_to_end(ep)
        for cam in CAMERAS:
            if self.raw_images:
                from PIL import Image
                with Image.open(Path(record['source_path'])/self._rows[ep][frame][cam]) as image:
                    pixels=np.asarray(image.convert('RGB')).copy()
                result[cam]=torch.from_numpy(pixels).permute(2,0,1).float()/255
            else:
                key=(ep,cam)
                if key not in self._maps:
                    if len(self._maps)>=24:self._maps.popitem(last=False)
                    data=CACHE/cam/f'episode_{ep:06d}.bin'
                    with np.load(data.with_suffix('.npz')) as archive:offsets=archive['offsets']
                    self._maps[key]=(np.memmap(data,dtype=np.uint8,mode='r'),offsets)
                self._maps.move_to_end(key)
                data,offsets=self._maps[key]
                result[cam]=torch.from_numpy(unpack_frame(memoryview(data)[int(offsets[frame]):int(offsets[frame+1])]).copy())
        return result
