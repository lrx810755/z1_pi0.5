import tempfile
from pathlib import Path
import unittest
import numpy as np


class Contracts(unittest.TestCase):
    def test_cache_reserve_accounts_for_already_decoded_bytes(self):
        from decode import required_free_bytes
        reserve=20*2**30
        self.assertEqual(required_free_bytes(100,0),reserve+72000000)
        self.assertEqual(required_free_bytes(100,72000000),reserve)
        self.assertLess(required_free_bytes(117537,40*2**30),85*2**30)

    def test_launch_rejects_changed_cache_and_stale_smoke_weights(self):
        import json
        from common import require_training_evidence, sha, checkpoint_fingerprints
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp);(run/'cache').mkdir()
            def put(name,data):
                p=run/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data))
            put('prepared.json',{'passed':True})
            put('code_snapshot.json',{'files':{}})
            code=sha(run/'code_snapshot.json')
            put('cache/completed.json',dict(complete=True,episodes=397,frames=117537,images=352611,prepared_sha256=sha(run/'prepared.json')))
            prep=sha(run/'prepared.json');cache=sha(run/'cache/completed.json')
            put('cpu_verified.json',dict(passed=True,prepared_sha256=prep,cache_manifest_sha256=cache,code_snapshot_sha256=code))
            checkpoint=run/'smoke/checkpoints/000004/pretrained_model';checkpoint.mkdir(parents=True)
            (checkpoint/'model.safetensors').write_bytes(b'original')
            put('smoke_completed.json',dict(optimizer_updates=4,output=str(run/'smoke')))
            put('smoke_first_update_verified.json',dict(passed=True,actual_optimizer_steps=[1],action_head_changed=True))
            put('smoke_live_training.json',dict(optimizer_updates=4,actual_optimizer_steps=[4]))
            put('smoke_latest_checkpoint.json',dict(optimizer_updates=4,checkpoint=str(checkpoint)))
            put('smoke_reload_verified.json',dict(passed=True,prepared_sha256=prep,cache_manifest_sha256=cache,code_snapshot_sha256=code,checkpoint=str(checkpoint),checkpoint_files=checkpoint_fingerprints(checkpoint)))
            require_training_evidence(run,formal=True)
            (checkpoint/'model.safetensors').write_bytes(b'replaced')
            with self.assertRaisesRegex(ValueError,'checkpoint'):
                require_training_evidence(run,formal=True)
            put('cache/completed.json',dict(complete=True,episodes=397,frames=117537,images=352611,prepared_sha256=prep,changed=True))
            with self.assertRaisesRegex(ValueError,'cache'):
                require_training_evidence(run,formal=False)

    def test_native_optimizer_hook_keeps_backbone_rate_separate(self):
        from config import make_config
        from dataset import PeachDataset
        from train import install_hooks
        from lerobot.configs.types import FeatureType
        from lerobot.utils.feature_utils import dataset_to_policy_features
        from lerobot.policies.act.modeling_act import ACTPolicy
        cfg=make_config(validate=False);cfg.policy.device='cpu'
        features=dataset_to_policy_features(PeachDataset(raw_images=True).meta.features)
        cfg.policy.input_features={k:v for k,v in features.items() if v.type is not FeatureType.ACTION}
        cfg.policy.output_features={k:v for k,v in features.items() if v.type is FeatureType.ACTION}
        model=ACTPolicy(cfg.policy)
        trainer=install_hooks(cfg,True)
        optimizer,scheduler=trainer.make_optimizer_and_scheduler(cfg,model)
        self.assertEqual(len(optimizer.param_groups),2)
        self.assertAlmostEqual(optimizer.param_groups[1]['lr']/optimizer.param_groups[0]['lr'],.1)
        backbone_ids={id(p) for p in model.model.backbone.parameters() if p.requires_grad}
        self.assertEqual(backbone_ids,{id(p) for p in optimizer.param_groups[1]['params']})
        self.assertAlmostEqual(scheduler.lr_lambdas[0](60000),.1)

    def test_float_cache_roundtrip_is_bit_exact(self):
        from codec import pack_frame, unpack_frame
        pixels = np.random.default_rng(7).random((3, 224, 224), dtype=np.float32)
        pixels[0, 0, 0] = np.float32(-0.)
        restored = unpack_frame(pack_frame(pixels))
        self.assertEqual(restored.dtype, np.float32)
        self.assertEqual(restored.tobytes(), pixels.tobytes())

    def test_corrupt_cache_payload_fails(self):
        from codec import pack_frame, unpack_frame
        payload = pack_frame(np.zeros((3, 224, 224), dtype=np.float32))
        with self.assertRaises(Exception):
            unpack_frame(payload[:-8])

    def test_windows_cannot_cross_intervention_or_episode_end(self):
        from prepare import valid_anchors
        expert = np.array([True] * 35 + [False] + [True] * 35)
        expected = np.zeros(71, dtype=bool)
        expected[:6] = True
        expected[36:42] = True
        np.testing.assert_array_equal(valid_anchors(expert), expected)
        self.assertFalse(valid_anchors(np.ones(29, dtype=bool)).any())

    def test_batch_has_48_basic_16_correction_without_duplicates(self):
        from sampling import draw_block
        pools = [np.arange(128), np.arange(128, 192)]
        probabilities = [np.ones(128) / 128, np.ones(64) / 64]
        rng = np.random.default_rng(20261118)
        for _ in range(100):
            batch = draw_block(rng, pools, probabilities)
            self.assertEqual(len(batch), 64)
            self.assertEqual(len(set(batch)), 64)
            self.assertEqual(int((batch < 128).sum()), 48)
            self.assertEqual(int((batch >= 128).sum()), 16)


if __name__ == '__main__':
    unittest.main()
