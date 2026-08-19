import os, pytest, torch
def test_gpu_policy_env():
 if torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES') not in (None,'1'): pytest.fail('Exp3 must use physical GPU1')
