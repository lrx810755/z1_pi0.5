import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review


class ReviewTests(unittest.TestCase):
    def test_discovery_skips_inflight_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name in ['dataset/episode_000001','.recording_000002','rejected_sources/episode_000003']:
                folder=root/name;folder.mkdir(parents=True);(folder/'frames.jsonl').write_text('{}\n')
            found,_=review.discover([root])
            self.assertEqual(len(found),1)
            self.assertEqual(found[0].parent.name,'episode_000001')
            alias=root/'selected'/'episode_000001';alias.parent.mkdir()
            alias.symlink_to(root/'dataset'/'episode_000001',target_is_directory=True)
            (root/'dataset'/'loop').symlink_to(root,target_is_directory=True)
            found,aliases=review.discover([root])
            self.assertEqual(len(found),1)
            self.assertEqual(len(aliases),1)

    def test_gripper_motion_and_changed_commands_are_not_full_robot_holds(self):
        n=100;times=np.arange(n)/10;cfg=review.Settings()
        q=np.zeros((n,7));command=q.copy()
        self.assertTrue(review.hold_mask(q,command,times,None,None,10,cfg).all())
        q[:,6]=times*.03
        self.assertFalse(review.hold_mask(q,command,times,None,None,10,cfg).any())
        q[:]=0;command[:,0]=times*.03
        self.assertFalse(review.hold_mask(q,command,times,None,None,10,cfg).any())

    def test_direction_statistics_and_duration_groups_do_not_mix_objects(self):
        items=[]
        for i in range(9):
            items.append(dict(group='apple',read_error='',expert_s=50. if i==8 else 10.,
                duration_s=50. if i==8 else 10.,frames=100,simulation_span_s=10.,
                phase_durations={},findings=[]))
        items.append(dict(group='bottle',read_error='',expert_s=100.,duration_s=100.,frames=100,
            simulation_span_s=100.,phase_durations={},findings=[]))
        review.duration_outliers(items,review.Settings())
        self.assertEqual(len(items[8]['findings']),1)
        self.assertEqual(len(items[9]['findings']),0)

    def test_cartesian_oscillation_is_flagged_but_monotonic_motion_is_not(self):
        chain=review.chain_from_urdf(review.DEFAULT_URDF)
        if chain is None:self.skipTest('Z1 URDF unavailable')
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'episode_000000';folder.mkdir()
            t=np.arange(80)/10
            for oscillate in [False,True]:
                q=np.repeat(np.array([[0.,2.2,-.8,-1.1,0.,-1.6,-.6]]),len(t),axis=0)
                q[:,0]=.04*np.sin(2*np.pi*t) if oscillate else -.1+.2*t/t[-1]
                rows=[dict(simulation_time=float(x),phase='TRANSPORT',target_object='test',
                           **{'observation.state':v.tolist(),'action':v.tolist()}) for x,v in zip(t,q)]
                p=folder/'frames.jsonl';p.write_text('\n'.join(json.dumps(r) for r in rows))
                result=review.analyze(p,chain,review.Settings())
                if oscillate:self.assertTrue(any('往返抖动' in f['type'] for f in result['findings']))
                else:self.assertEqual(result['findings'],[])

    def test_local_jitter_is_found_after_a_long_monotonic_carry(self):
        fps=10;t=np.arange(160)/fps
        positions=np.zeros((len(t),3))
        positions[:,0]=np.minimum(t,8)*.04
        positions[80:,0]=.32+.012*np.sin(2*np.pi*(t[80:]-8))
        positions[:,1]=.14
        found=review.jitter_windows(positions,0,len(t),fps,review.Settings())
        self.assertTrue(found)
        self.assertTrue(any(c['start']>=70 and c['turns']>=4 for c in found))

    def test_end_to_end_exact_double_speed_video_and_chapter_checklist(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ep=root/'source'/'episode_000000';ep.mkdir(parents=True)
            Image.new('RGB',(96,72),'blue').save(ep/'image.jpg')
            rows=[dict(frame_index=i,timestamp=i/10,simulation_time=100+i/10,
                **{'observation.state':[0.]*7,'action':[0.]*7,
                   'observation.images.wrist':'image.jpg','observation.images.global':'image.jpg'},
                target_object='test',phase='READY') for i in range(60)]
            (ep/'frames.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            (ep/'episode_summary.json').write_text(json.dumps(dict(fps=10,frame_count=60,target_object='test')))
            out=root/'result'
            subprocess.run([sys.executable,str(Path(review.__file__)),'--roots',str(root/'source'),
                '--output',str(out)],check=True,stdout=subprocess.DEVNULL)
            result=json.loads((out/'completed.json').read_text())
            self.assertEqual(result['scanned'],1)
            self.assertEqual(result['candidates'],1)
            self.assertEqual(result['video']['image_read_errors'],0)
            #6 source seconds /2 +0.8-second title card.
            self.assertAlmostEqual(result['video']['duration_s'],3.8,places=3)
            self.assertTrue((out/'异常候选清单.csv').exists())
            self.assertTrue((out/'异常轨迹汇总_2倍速.mp4').exists())
            chapter=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_chapters',
                '-of','json',str(out/'异常轨迹汇总_2倍速.mp4')],text=True))
            self.assertEqual(len(chapter['chapters']),1)
            # Analysis never modifies the source data or writes into its folder.
            self.assertEqual(sorted(p.name for p in ep.iterdir()),['episode_summary.json','frames.jsonl','image.jpg'])


if __name__=='__main__':
    unittest.main()
