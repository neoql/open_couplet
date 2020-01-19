import torch

from open_couplet.models.seq2seq import CNN


def test_cnn():
    cnn = CNN(32, 64, 32)

    x = torch.randn(3, 7, 32)
    length = torch.tensor([7, 6, 5])
    mask = (torch.arange(0, x.size(1)) >= length.unsqueeze(-1)).unsqueeze(1)

    output, _ = cnn(x, mask=mask)

    for i in range(length.size(0)):
        x_i = x[i].unsqueeze(0)[:, :length[i], :]
        y_i, _ = cnn(x_i)
        assert torch.equal(output[i, :length[i], :], y_i.squeeze(0))
