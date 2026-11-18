import unittest
import numpy as np
from screen_fruit_repetition import RepeatRules,repeated_motion


class RepetitionTests(unittest.TestCase):
    def screen(self,points,joints=None,phases=None,loaded=False,expert=None):
        n=len(points)
        return repeated_motion(points,np.zeros((n,6)) if joints is None else joints,
            np.full(n,'TRANSPORT' if loaded else 'READY') if phases is None else phases,
            np.full(n,'带载抬升搬运' if loaded else '空爪接近'),
            np.ones(n,bool) if expert is None else expert,10,RepeatRules())[0]

    def test_sustained_shuttle_is_flagged(self):
        t=np.arange(100)/10
        p=np.c_[.025*np.sin(2*np.pi*.7*t),np.zeros((100,2))]
        self.assertTrue(any(f['type']=='多次往返或局部抖动候选' and f['tier']=='重点' for f in self.screen(p)))

    def test_normal_lift_carry_descent_is_not_repetition(self):
        p=np.c_[np.linspace(0,.3,100),np.zeros(100),np.r_[np.linspace(.2,.35,50),np.linspace(.35,.2,50)]]
        self.assertEqual(self.screen(p,loaded=True),[])

    def test_submillimetre_noise_is_ignored(self):
        t=np.arange(100)/10
        p=np.c_[.03*t,.0005*np.sin(20*t),.0005*np.cos(20*t)]
        self.assertEqual(self.screen(p),[])

    def test_stationary_tcp_with_shaking_wrist_is_flagged(self):
        t=np.arange(100)/10;q=np.zeros((100,6));q[:,4]=.07*np.sin(2*np.pi*.7*t)
        self.assertTrue(any(f['type']=='腕部关节反复调整候选' for f in self.screen(np.zeros((100,3)),q)))

    def test_deliberate_separate_steps_are_not_added_as_oscillations(self):
        t=np.arange(100)/10;p=np.c_[.025*np.sin(2*np.pi*.5*t),np.zeros((100,2))]
        phases=np.array(['READY#leg'+str(i//10) for i in range(100)])
        self.assertEqual(self.screen(p,phases=phases),[])

    def test_nonexpert_intervention_is_excluded(self):
        t=np.arange(100)/10;p=np.c_[.025*np.sin(2*np.pi*.7*t),np.zeros((100,2))]
        self.assertEqual(self.screen(p,expert=np.zeros(100,bool)),[])

    def test_many_stop_start_bouts_are_flagged(self):
        x=[0.]
        for i in range(6):
            x.extend(np.linspace(x[-1],x[-1]+.012,6)[1:]);x.extend([x[-1]]*10)
        p=np.c_[x,np.zeros((len(x),2))]
        self.assertTrue(any(f['type']=='搬运频繁启停候选' for f in self.screen(p,loaded=True)))


if __name__=='__main__':unittest.main()
