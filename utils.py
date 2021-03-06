import re
import pickle
import torch
import numpy as np
from collections import Counter
from torch.utils.data import Dataset
from rdkit import Chem
from tqdm import tqdm


def split(smiles):
    pattern = r'\^|\s|#|=|-[0-9]*|\+[0-9]*|[0-9]|\[.{2,5}\]|%[0-9]{2}|\(|\)|\.|/|\\|:|@+|\{|\}|Cl|Ca|Cu|Br|Be|Ba|Bi|' \
              'Si|Se|Sr|Na|Ni|Rb|Ra|Xe|Li|Al|As|Ag|Au|Mg|Mn|Te|Zn|He|Kr|Fe|[BCFHIKNOPScnos]'
    return ' '.join(re.findall(pattern, smiles))


class TorchVocab(object):
    def __init__(self, counter, max_size=None, min_freq=1, specials=None,
                 vectors=None, unk_init=None, vectors_cache=None):

        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        if not specials:
            specials = [' ', '<oov>']
        self.itos = list(specials)
        for tok in specials:
            del counter[tok]

        max_size = None if max_size is None else max_size + len(self.itos)

        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

        self.vectors = None
        if vectors is not None:
            self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
        else:
            assert unk_init is None and vectors_cache is None

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        if self.vectors != other.vectors:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def vocab_rerank(self):
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1


class Vocab(TorchVocab):
    def __init__(self, counter, max_size=None, min_freq=1):
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4
        super().__init__(counter, specials=[" ", "<unk>", "<eos>", "^", "<mask>"], max_size=max_size,
                         min_freq=min_freq)

    def to_seq(self, sentece, seq_len, with_eos=False, with_sos=False) -> list:
        pass

    def from_seq(self, seq, join=False, with_pad=False):
        pass

    @staticmethod
    def load_vocab(vocab_path: str) -> 'Vocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


class WordVocab(Vocab):
    def __init__(self, texts, max_size=None, min_freq=1):
        print("\nBuilding vocab...")
        counter = Counter()
        for line in tqdm(texts):
            if isinstance(line, list):
                words = line
            else:
                words = line.replace("\n", "").replace("\t", "").split()

            for word in words:
                counter[word] += 1
        super().__init__(counter, max_size=max_size, min_freq=min_freq)

    def to_seq(self, sentence, seq_len=None, with_eos=False, with_sos=False, with_len=False):
        if isinstance(sentence, str):
            sentence = sentence.split()

        seq = [self.stoi.get(word, self.unk_index) for word in sentence]

        if with_eos:
            seq += [self.eos_index]  # this would be index 1
        if with_sos:
            seq = [self.sos_index] + seq

        origin_seq_len = len(seq)

        if seq_len is None:
            pass
        elif len(seq) <= seq_len:
            seq += [self.pad_index for _ in range(seq_len - len(seq))]
        else:
            seq = seq[:seq_len]

        return (seq, origin_seq_len) if with_len else seq

    def from_seq(self, seq, join=False, with_pad=False):
        words = [self.itos[idx] if idx < len(self.itos) else "<%d>" % idx for idx in seq if not with_pad or idx != self.pad_index]
        return " ".join(words) if join else words

    @staticmethod
    def load_vocab(vocab_path: str) -> 'WordVocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)


class SmilesEnumerator(object):
    def __init__(self, isomeric=True, canonical=False):
        self.isomericSmiles = isomeric
        self.canonical = canonical

    def randomize_smiles(self, smiles):
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m, ans)
        return Chem.MolToSmiles(nm, canonical=self.canonical, isomericSmiles=self.isomericSmiles)

    def __call__(self, smiles):
        sm_r = self.randomize_smiles(smiles)
        if sm_r is None:
            sm_spaced = split(smiles)
        else:
            sm_spaced = split(sm_r)
        return sm_spaced.split()


class Seq2seqDataset(Dataset):
    def __init__(self, smiles, vocab, seq_len=200):
        self.smiles = smiles
        self.vocab = vocab
        self.seq_len = seq_len
        self.transform = SmilesEnumerator()

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, item):
        sm = self.transform(self.smiles[item])  # List
        content = [self.vocab.stoi.get(token, self.vocab.unk_index) for token in sm]
        X = [self.vocab.sos_index] + content + [self.vocab.eos_index]
        padding = [self.vocab.pad_index] * (self.seq_len - len(X))
        X.extend(padding)
        return torch.tensor(X)
