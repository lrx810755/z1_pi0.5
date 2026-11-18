"""Compare RGB against robot-only rendering where depth proves the cup is hidden."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('candidate');a=p.parse_args();r=a.directory

def mask(name,label):
 labels=json.loads((r/(name+'_labels.json')).read_text())['idToLabels'];ids=[int(k) for k,v in labels.items() if label in v['class'].split(',')];return np.isin(np.load(r/(name+'_semantic.npy')),ids)
rd=np.load(r/'robot_only_depth.npy');cd=np.load(r/'cup_only_depth.npy');occluded=mask('robot_only','robot')&mask('cup_only','cup')&np.isfinite(rd)&np.isfinite(cd)&(rd<cd-.001)
robot=np.asarray(Image.open(r/'robot_only.png')).astype(float);im=np.asarray(Image.open(r/(a.candidate+'.png'))).astype(float);delta=np.max(abs(im-robot),axis=-1);n=int(occluded.sum());bad=int((occluded&(delta>30)).sum());report=dict(candidate=a.candidate,occluded_pixels=n,large_rgb_difference_pixels=bad,large_difference_fraction=bad/n if n else 0,median_rgb_difference=float(np.median(delta[occluded])) if n else 0)
print(json.dumps(report));assert n>100,'Probe must include meaningful cup/robot projection overlap';assert report['large_difference_fraction']<.03,report;assert report['median_rgb_difference']<5,report
