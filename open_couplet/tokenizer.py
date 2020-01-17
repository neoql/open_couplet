class Tokenizer(object):
    SPECIAL_TOKENS_ATTRIBUTES = ['pad_token', 'bos_token', 'eos_token', 'sep_token',
                                 'unk_token', 'mask_token', 'cls_token']

    def __init__(self, token_iter=None, **kwargs):
        self._bos_token = '[BOS]'
        self._eos_token = '[EOS]'
        self._pad_token = '[PAD]'
        self._sep_token = '[SEP]'
        self._unk_token = '[UNK]'
        self._mask_token = '[MASK]'
        self._cls_token = '[CLS]'

        self._token2id = {}
        self._id2token = []

        init = token_iter is None

        special_tokens = []
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            val = kwargs.get(attr, getattr(self, attr))
            setattr(self, attr, val)
            special_tokens.append(val)
        special_tokens = special_tokens + self._unused_tokens(20)

        if init:
            token_iter = special_tokens
        self._special_tokens = special_tokens

        for token in token_iter:
            assert isinstance(token, str)
            token = token.strip()
            assert token not in self._token2id
            self._token2id[token] = len(self._id2token)
            self._id2token.append(token)

    # noinspection PyMethodMayBeStatic
    def _unused_tokens(self, number):
        tokens = []
        for i in range(number):
            tokens.append(f'[unused{i}]')
        return tokens

    @classmethod
    def from_vocab(cls, vocab_file, **kwargs):
        with open(vocab_file, 'r') as fp:
            return cls(fp, **kwargs)

    def save_vocab(self, vocab_file):
        with open(vocab_file, 'w') as fp:
            for token in self._token2id:
                fp.write(f'{token}\n')
            fp.flush()

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self._id2token[ids]

        return [self._id2token[_id] for _id in ids]

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self._token2id.get(tokens, self._token2id[self.unk_token])

        return [self._token2id.get(token, self.unk_token_id) for token in tokens]

    def add_tokens(self, tokens):
        if isinstance(tokens, str):
            tokens = [tokens]

        add_tokens = []
        for token in tokens:
            if token != self.unk_token and \
                    self.convert_tokens_to_ids(token) == self.unk_token_id and \
                    token not in add_tokens:
                add_tokens.append(token)

        offset = self.vocab_size
        self._id2token += add_tokens
        self._token2id.update({token: offset+i for i, token in enumerate(add_tokens)})

        return len(add_tokens)

    @property
    def special_tokens(self):
        return self._special_tokens

    @property
    def special_token_ids(self):
        return self.convert_tokens_to_ids(self.special_tokens)

    @property
    def vocab_size(self):
        return len(self._token2id)

    @property
    def bos_token(self):
        return self._bos_token

    @property
    def eos_token(self):
        return self._eos_token

    @property
    def pad_token(self):
        return self._pad_token

    @property
    def sep_token(self):
        return self._sep_token

    @property
    def unk_token(self):
        return self._unk_token

    @property
    def mask_token(self):
        return self._mask_token

    @property
    def cls_token(self):
        return self._cls_token

    @bos_token.setter
    def bos_token(self, bos_token):
        self._bos_token = bos_token

    @eos_token.setter
    def eos_token(self, eos_token):
        self._eos_token = eos_token

    @pad_token.setter
    def pad_token(self, pad_token):
        self._pad_token = pad_token

    @unk_token.setter
    def unk_token(self, unk_token):
        self._unk_token = unk_token

    @sep_token.setter
    def sep_token(self, sep_token):
        self._sep_token = sep_token

    @mask_token.setter
    def mask_token(self, mask_token):
        self._mask_token = mask_token

    @cls_token.setter
    def cls_token(self, cls_token):
        self._cls_token = cls_token

    @property
    def bos_token_id(self):
        return self.convert_tokens_to_ids(self.bos_token)

    @property
    def eos_token_id(self):
        return self.convert_tokens_to_ids(self.eos_token)

    @property
    def pad_token_id(self):
        return self.convert_tokens_to_ids(self.pad_token)

    @property
    def sep_token_id(self):
        return self.convert_tokens_to_ids(self.sep_token)

    @property
    def unk_token_id(self):
        return self.convert_tokens_to_ids(self.unk_token)

    @property
    def mask_token_id(self):
        return self.convert_tokens_to_ids(self.mask_token)

    @property
    def cls_token_id(self):
        return self.convert_tokens_to_ids(self.cls_token)
