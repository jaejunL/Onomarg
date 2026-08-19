import os
def test_visible_gpu(): assert os.environ.get('CUDA_VISIBLE_DEVICES','1')=='1'
