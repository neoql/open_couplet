import json


class Seq2seqConfig(object):
    def __init__(self, **cfg):
        self.vocab_size = cfg.get('vocab_size', 10000)
        self.hidden_size = cfg.get('hidden_size', 700)
        self.rnn_layers = cfg.get('rnn_layers', 2)
        self.cnn_kernel_size = cfg.get('cnn_kernel_size', 3)
        self.dropout_p = cfg.get('dropout_p', 0.1)

    @classmethod
    def from_config(cls, filename):
        config = cls()
        with open(filename, 'r') as fp:
            config.__dict__ = json.load(fp)
        return config

    def save_config(self, filename):
        with open(filename, 'w') as fp:
            json.dump(self.__dict__, fp, indent=2)

    def __repr__(self):
        return f'Seq2seqConfig({self.__dict__})'
