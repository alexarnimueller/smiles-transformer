import torch
import numpy as np

from torch.utils.data import Dataset
from rdkit import Chem

from .utils import split

PAD = 0
MAX_LEN = 220


class SmilesEnumerator(object):
    """SMILES Enumerator, vectorizer and devectorizer

    #Arguments
        charset: string containing the characters for the vectorization
          can also be generated via the .fit() method
        pad: Length of the vectorization
        leftpad: Add spaces to the left of the SMILES
        isomericSmiles: Generate SMILES containing information about stereogenic centers
        enum: Enumerate the SMILES during transform
        canonical: use canonical SMILES during transform (overrides enum)
    """

    def __init__(self, charset=None, pad=120, leftpad=True, isomericSmiles=True, enum=True,
                 canonical=False):
        if not charset:
            charset = '@C)(=cOn1S2/H[N]\\'
        self.charset = charset
        self._charset = None
        self.pad = pad
        self.leftpad = leftpad
        self.isomericSmiles = isomericSmiles
        self.enumerate = enum
        self.canonical = canonical

    @property
    def charset(self):
        return self._charset

    @charset.setter
    def charset(self, charset):
        self._charset = charset
        self._charlen = len(charset)
        self._char_to_int = dict((c, i) for i, c in enumerate(charset))
        self._int_to_char = dict((i, c) for i, c in enumerate(charset))

    def fit(self, smiles, extra_chars=None, extra_pad=5):
        """Performs extraction of the charset and length of a SMILES datasets and sets self.pad and self.charset

        #Arguments
            smiles: Numpy array or Pandas series containing smiles as strings
            extra_chars: List of extra chars to add to the charset (e.g. "\\\\" when "/" is present)
            extra_pad: Extra padding to add before or after the SMILES vectorization
        """
        charset = set("".join(list(smiles)))
        if extra_chars:
            self.charset = "".join(charset.union(set(extra_chars)))
        else:
            self.charset = charset
        self.pad = max([len(smile) for smile in smiles]) + extra_pad

    def randomize_smiles(self, smiles):
        """Perform a randomization of a SMILES string must be RDKit sanitizable"""
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None  # Invalid SMILES
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m, ans)
        return Chem.MolToSmiles(nm, canonical=self.canonical, isomericSmiles=self.isomericSmiles)

    def transform(self, smiles):
        """Perform an enumeration (randomization) and vectorization of a Numpy array of smiles strings
        #Arguments
            smiles: Numpy array or Pandas series containing smiles as strings
        """
        one_hot = np.zeros((smiles.shape[0], self.pad, self._charlen), dtype=np.int8)

        if self.leftpad:
            for i, ss in enumerate(smiles):
                if self.enumerate:
                    ss = self.randomize_smiles(ss)
                l = len(ss)
                diff = self.pad - l
                for j, c in enumerate(ss):
                    one_hot[i, j + diff, self._char_to_int[c]] = 1
            return one_hot
        else:
            for i, ss in enumerate(smiles):
                if self.enumerate:
                    ss = self.randomize_smiles(ss)
                for j, c in enumerate(ss):
                    one_hot[i, j, self._char_to_int[c]] = 1
            return one_hot

    def reverse_transform(self, vect):
        """ Performs a conversion of a vectorized SMILES to a smiles strings
        charset must be the same as used for vectorization.
        #Arguments
            vect: Numpy array of vectorized SMILES.
        """
        smiles = []
        for v in vect:
            # Find one hot encoded index with argmax, translate to char and join to string
            v = v[v.sum(axis=1) == 1]
            smile = "".join(self._int_to_char[i] for i in v.argmax(axis=1))
            smiles.append(smile)
        return np.array(smiles)

    def __call__(self, smiles):
        """ Perform SMILES transformation """
        sm_r = self.randomize_smiles(smiles)
        if sm_r is None:
            sm_spaced = split(smiles)  # Spacing
        else:
            sm_spaced = split(sm_r)  # Spacing
        sm_split = sm_spaced.split()
        if len(sm_split) <= MAX_LEN - 2:
            return sm_split  # List
        else:
            return split(smiles).split()


class Seq2seqDataset(Dataset):
    def __init__(self, smiles, vocab, seq_len=220):
        self.smiles = smiles
        self.vocab = vocab
        self.seq_len = seq_len
        self.transform = SmilesEnumerator()

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, item):
        sm = self.smiles[item]
        sm = self.transform(sm)  # List
        content = [self.vocab.stoi.get(token, self.vocab.unk_index) for token in sm]
        X = [self.vocab.sos_index] + content + [self.vocab.eos_index]
        padding = [self.vocab.pad_index] * (self.seq_len - len(X))
        X.extend(padding)
        return torch.tensor(X)
