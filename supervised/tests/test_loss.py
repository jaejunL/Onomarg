import torch
def test_label_smoothing_finite():
 f=torch.nn.CrossEntropyLoss(ignore_index=-100,label_smoothing=.1); logits=torch.randn(2,4,10).transpose(1,2); assert torch.isfinite(f(logits,torch.tensor([[1,2,-100,3],[2,3,4,-100]])))
