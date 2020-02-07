from train.data import RandomBatchSampler


def test_random_batch_sampler():
    case = [False] * 8
    sampler = RandomBatchSampler(case, batch_size=3, drop_last=False)
    for batch in sampler:
        for i in batch:
            case[i] = True

    assert all(case)

    case = [False] * 8
    sampler = RandomBatchSampler(case, batch_size=3, drop_last=True)
    for batch in sampler:
        for i in batch:
            case[i] = True

    assert all(case[:6])
    assert not any(case[6:])

    case = [False] * 8
    sampler = RandomBatchSampler(case, batch_size=2, drop_last=False)
    for batch in sampler:
        for i in batch:
            case[i] = True

    assert all(case)
