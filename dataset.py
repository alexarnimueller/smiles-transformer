import torch
import numpy as np

from torch.utils.data import Dataset
from rdkit import Chem

from utils import split

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

    def __init__(self, charset=None, pad=120, leftpad=True, isomeric=True, enum=True, canonical=False):
        if not charset:
            charset = '@C)(=cOn1S2/H[N]\\'
        self.charset = charset
        self._charset = None
        self.pad = pad
        self.leftpad = leftpad
        self.isomericSmiles = isomeric
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

    def randomize_smiles(self, smiles):
        """Perform a randomization of a SMILES string must be RDKit sanitizable"""
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None  # Invalid SMILES
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m, ans)
        return Chem.MolToSmiles(nm, canonical=self.canonical, isomericSmiles=self.isomericSmiles)

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
        sm = self.transform(self.smiles[item])  # List
        content = [self.vocab.stoi.get(token, self.vocab.unk_index) for token in sm]
        X = [self.vocab.sos_index] + content + [self.vocab.eos_index]
        padding = [self.vocab.pad_index] * (self.seq_len - len(X))
        X.extend(padding)
        return torch.tensor(X)
