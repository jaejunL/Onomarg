import torch


def test_audio_projection_shape():
    projection = torch.nn.Linear(768, 768)
    assert projection(torch.zeros(2, 4, 768)).shape == (2, 4, 768)
