import logging
import torch
import itertools

from open_couplet import CoupletBot
from open_couplet.models import Seq2seqModel
from open_couplet.tokenizer import Seq2seqTokenizer
from open_couplet.config import Seq2seqConfig


logging.basicConfig(level=logging.DEBUG)


def test_cleaning_inputs():
    golden = '天连碧树春滋雨'

    before_cleaning = [
        '天连碧树春滋雨',
        ' 天 连 碧 树 春 滋 雨 ',
        '\t天连碧树春滋雨\t',
        '   天连碧树春滋雨  ',
        'a天连碧树b春滋A雨',
        'cd天ab连碧A树春kkk滋雨  ',
        ' \tcd天ab连   碧A\t树  春kkk滋123 雨  ',
        'cd天ab连D碧A树春kkk滋雨  \n',
    ]

    for s in before_cleaning:
        assert CoupletBot.cleaning_inputs(s) == golden
        logging.debug(f'"{s}" cleaned success')


def test_tokenize():
    up_part_case1 = ["园满香花红缀地"]

    up_part_case2 = [
        "园满香花红缀地",
        "院满红花香满径",
        "艳艳红花飞处处",
    ]

    up_part_case3 = [
        "东山鹤树，逢春风百里华亭鹤唳",
        "绿水青山，如歌如画",
        "院满红花香满径",
    ]

    up_part_case4 = [
        "绿水青山，如歌如画",
        "东山鹤树，逢春风百里华亭鹤唳",
        "院满红花香满径",
    ]

    tokenizer = Seq2seqTokenizer()
    for tokens in itertools.chain(up_part_case1 + up_part_case2 + up_part_case3 + up_part_case4):
        tokenizer.add_tokens(list(tokens))
    model = Seq2seqModel(Seq2seqConfig(vocab_size=tokenizer.vocab_size, hidden_size=128))

    bot = CoupletBot(model, tokenizer)

    out, length = bot.tokenize(up_part_case1)
    assert length.squeeze(0).equal(torch.tensor(len(up_part_case1[0]) + 2))
    assert out.size() == (1, len(up_part_case1[0]) + 2)

    assert tokenizer.convert_ids_to_tokens(out[0].tolist()) \
           == [tokenizer.bos_token] + list(up_part_case1[0]) + [tokenizer.eos_token]

    out, length = bot.tokenize(up_part_case2)
    assert length.equal(torch.tensor([len(s)+2 for s in up_part_case2]))
    for i in range(3):
        assert tokenizer.convert_ids_to_tokens(out[i].tolist()) \
               == [tokenizer.bos_token] + list(up_part_case2[i]) + [tokenizer.eos_token]

    out, length = bot.tokenize(up_part_case3)
    assert length.equal(torch.tensor([len(s) + 2 for s in up_part_case3]))
    for i in range(3):
        assert tokenizer.convert_ids_to_tokens(out[i].tolist()) \
               == [tokenizer.bos_token] + list(up_part_case3[i]) + [tokenizer.eos_token] + \
               [tokenizer.pad_token] * (length.max()-length[i]).item()


def test_reply():
    up_part_case1 = "园满香花红缀地"

    up_part_case2 = [
        "绿水青山，如歌如画",
        "东山鹤树，逢春风百里华亭鹤唳",
        "院满红花香满径",
    ]

    tokenizer = Seq2seqTokenizer()
    for tokens in itertools.chain([up_part_case1] + up_part_case2):
        tokenizer.add_tokens(list(tokens))
    model = Seq2seqModel(Seq2seqConfig(vocab_size=tokenizer.vocab_size, hidden_size=128))

    bot = CoupletBot(model, tokenizer)

    result = bot.reply(up_part_case1, topk=1)
    assert len(result) == 1
    assert isinstance(result[0], tuple)
    assert len(result[0][0]) == len(up_part_case1)
    assert isinstance(result[0][1], float)

    result = bot.reply(up_part_case1, topk=2)
    assert len(result) == 2
    for i in range(2):
        assert isinstance(result[i], tuple)
        assert len(result[i][0]) == len(up_part_case1)
        assert isinstance(result[i][1], float)

    results = bot.reply(up_part_case2, topk=1)

    for n, result in enumerate(results):
        assert len(result) == 1
        assert isinstance(result[0], tuple)
        print()
        assert len(result[0][0]) == len(up_part_case2[n])
        assert isinstance(result[0][1], float)

    results = bot.reply(up_part_case2, topk=2)

    for n, result in enumerate(results):
        for i in range(2):
            assert isinstance(result[i], tuple)
            assert len(result[i][0]) == len(up_part_case2[n])
            assert isinstance(result[i][1], float)
