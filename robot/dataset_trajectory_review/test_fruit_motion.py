"""Regression checks for purposeful motion and independently flagged defects."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from screen_fruit_motion import Rules, command_labels, height_screen, reverse_segments


class FruitScreeningTests(unittest.TestCase):
    def segments(self, points, phases=None, closed=False, fragment=False):
        n=len(points)
        return reverse_segments(points,np.full(n,'TRANSPORT' if closed else 'READY') if phases is None else phases,
            np.full(n,int(closed)),np.ones(n,bool),np.arange(n)/10,10,fragment,Rules())

    def test_normal_loaded_lift_carry_descent_is_not_reverse(self):
        n=90
        points=np.c_[np.linspace(0,.4,n),np.zeros(n),
            np.r_[np.linspace(.2,.4,45),np.linspace(.4,.2,45)]]
        self.assertFalse(any(s['tier'] for s in self.segments(points,closed=True)))

    def test_real_reversal_is_retained(self):
        x=np.r_[np.linspace(0,.2,30),np.linspace(.2,.1,30),np.linspace(.1,.4,30)]
        self.assertTrue(any(s['tier']=='重点' for s in self.segments(np.c_[x,np.zeros((90,2))])))

    def test_documented_correction_retreat_is_not_combined_with_approach(self):
        x=np.r_[np.linspace(0,.2,30),np.linspace(.2,.1,30),np.linspace(.1,.4,30)]
        phases=np.array(['READY#retreat']*30+['READY#align']*30+['READY#approach']*30)
        self.assertFalse(any(s['tier'] for s in self.segments(np.c_[x,np.zeros((90,2))],phases,fragment=True)))

    def test_small_geometric_noise_does_not_flag(self):
        t=np.linspace(0,1,90)
        points=np.c_[.4*t,.0005*np.sin(100*t),.0005*np.cos(100*t)]
        self.assertFalse(any(s['tier'] for s in self.segments(points)))

    def test_large_reverse_is_not_diluted_by_long_detour(self):
        x=np.r_[np.linspace(0,.2,30),np.linspace(.2,.1,30),np.linspace(.1,.4,30)]
        points=np.c_[x,2*np.sin(np.linspace(0,np.pi,90)),np.zeros(90)]
        segments=self.segments(points)
        self.assertTrue(any(s['tier']=='重点' and s['backward_fraction']<.08 for s in segments))

    def test_apple_and_peach_have_separate_height_baselines(self):
        items=[];arrays={}
        for obj,base in [('apple',.30),('peach',.45)]:
            for i in range(10):
                path=obj+str(i);z=np.full(30,base)
                if obj=='apple' and i==9:z[10:20]=.45
                items.append(dict(path=path,object=obj,fragment=False,kind='basic',grid=[0,0],fps=10,
                    height_metrics=[dict(mode='空爪接近',peak_z_m=float(z.max()))],findings=[]))
                arrays[path]=dict(z=z,modes=np.full(30,'空爪接近'),expert=np.ones(30,bool))
        height_screen(items,arrays,Rules())
        self.assertEqual(sum(bool(e['findings']) for e in items),1)
        self.assertEqual(items[9]['findings'][0]['tier'],'重点')

    def test_bad_label_length_is_reported_and_source_is_not_modified(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);p=root/'gripper_labels.json'
            raw=json.dumps(dict(labels=[0,0]));p.write_text(raw)
            rows=[dict(action=[0]*6+[value]) for value in [-.9,-.8,-.3]]
            labels,source,warnings=command_labels(rows,root)
            self.assertEqual(labels.tolist(),[0,0,1])
            self.assertEqual(len(warnings),1)
            self.assertEqual(p.read_text(),raw)


if __name__=='__main__':unittest.main()
