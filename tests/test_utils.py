import torch

from open_couplet import utils


def test_sentence_pattern():
    pad = 0

    sentences = torch.tensor([
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 0, 0],

        [1, 2, 3, 4, 1, 2, 3, 4],
        [1, 2, 3, 4, 1, 2, 0, 0],

        [1, 1, 2, 3, 2, 3, 5, 0],
        [1, 1, 2, 3, 2, 3, 0, 0],
    ])

    max_len = sentences.size(1)
    pad_mask = sentences.eq(pad)

    # pattern: (batch_size, max_len, max_len)
    pattern = utils.sentence_pattern(sentences, pad_mask=pad_mask)

    golden_pattern_1 = torch.eye(max_len, dtype=torch.bool)

    golden_pattern_2 = torch.zeros(max_len, max_len, dtype=torch.bool)
    golden_pattern_2[range(6), range(6)] = 1

    golden_pattern_3 = torch.eye(max_len // 2, dtype=torch.bool)
    golden_pattern_3 = torch.cat([golden_pattern_3] * 2, dim=1)
    golden_pattern_3 = torch.cat([golden_pattern_3] * 2)

    golden_pattern_4 = torch.cat([golden_pattern_3[:-2], torch.zeros_like(golden_pattern_3[-2:])])
    golden_pattern_4 = torch.cat([golden_pattern_4[:, :-2], torch.zeros_like(golden_pattern_4[:, -2:])], dim=1)

    golden_pattern_5 = torch.tensor([
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=torch.bool)

    golden_pattern_6 = golden_pattern_5.clone()
    golden_pattern_6[max_len-2, max_len-2] = 0

    assert golden_pattern_1.equal(pattern[0])
    assert golden_pattern_2.equal(pattern[1])
    assert golden_pattern_3.equal(pattern[2])
    assert golden_pattern_4.equal(pattern[3])
    assert golden_pattern_5.equal(pattern[4])
    assert golden_pattern_6.equal(pattern[5])


def test_replace_en_punctuation():
    source = [
        r'李白(701年－762年) ,字太白,号青莲居士,又号"谪仙人",唐代伟大的浪漫主义诗人,被后人誉为"诗仙",与杜甫并称为"李杜",为了与另两位诗人'
        r'李商隐与杜牧即"小李杜"区别,杜甫与李白又合称"大李杜".据<新唐书>记载,李白为兴圣皇帝(凉武昭王李暠)九世孙,与李唐诸王同宗.其人爽朗大'
        r'方,爱饮酒作诗,喜交友.',
        r'李白深受黄老列庄思想影响,有<李太白集>传世,诗作中多以醉时写的,代表作有<望庐山瀑布>\<行路难>\<蜀道难>\<将进酒>\<明堂赋>\<早发'
        r'白帝城>等多首.',
        r'噫吁嚱~危乎高哉!蜀道之难,难于上青天![唐 李白]',
        r'李白是唐代诗人吗?',
        r'有的学会烤烟,自己做挺讲究的纸烟和雪茄;有的学会蔬菜加工,做的番茄酱能吃到冬天;有的学会蔬菜腌渍\窖藏,使秋菜接上春菜.'
    ]

    target = [
        '李白（701年－762年） ，字太白，号青莲居士，又号“谪仙人”，唐代伟大的浪漫主义诗人，被后人誉为“诗仙”，与杜甫并称为“李杜”，为了与另'
        '两位诗人李商隐与杜牧即“小李杜”区别，杜甫与李白又合称“大李杜”。据《新唐书》记载，李白为兴圣皇帝（凉武昭王李暠）九世孙，与李唐诸王'
        '同宗。其人爽朗大方，爱饮酒作诗，喜交友。',
        '李白深受黄老列庄思想影响，有《李太白集》传世，诗作中多以醉时写的，代表作有《望庐山瀑布》、《行路难》、《蜀道难》、《将进酒》、'
        '《明堂赋》、《早发白帝城》等多首。',
        '噫吁嚱～危乎高哉！蜀道之难，难于上青天！【唐 李白】',
        '李白是唐代诗人吗？',
        '有的学会烤烟，自己做挺讲究的纸烟和雪茄；有的学会蔬菜加工，做的番茄酱能吃到冬天；有的学会蔬菜腌渍、窖藏，使秋菜接上春菜。'
    ]

    for src, tgt in zip(source, target):
        assert utils.replace_en_punctuation(src) == tgt
