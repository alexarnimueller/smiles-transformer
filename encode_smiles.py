import torch
from pretrain_trfm import TrfmSmiles
from utils import WordVocab, split


def get_inputs(sm, maxlen=200):
    sm = sm.split()
    if len(sm) > (maxlen-2):
        print('SMILES is too long ({:d})'.format(len(sm)))
        sm = sm[:(maxlen/2)-1] + sm[-(maxlen/2)-1:]
    ids = [vocab.stoi.get(token, 1) for token in sm]
    ids = [3] + ids + [2]
    seg = [1]*len(ids)
    padding = [0]*(maxlen - len(ids))
    ids.extend(padding)
    seg.extend(padding)
    return ids, seg


def get_array(smiles):
    x_id, x_seg = [], []
    for sm in smiles:
        a, b = get_inputs(sm)
        x_id.append(a)
        x_seg.append(b)
    return torch.tensor(x_id), torch.tensor(x_seg)


print("Loading vocab and trained model...")
vocab = WordVocab.load_vocab('.data/vocab.pkl')
trfm = TrfmSmiles(len(vocab), 128, len(vocab), 3)
trfm.load_state_dict(torch.load('.save/trfm_4_10000.pkl'))
trfm.eval()
print('Total parameters:', sum(p.numel() for p in trfm.parameters()))

print("Reading SMILES...")
with open('.data/chembl24.txt', 'r') as f:
    smls = list()
    for line in f:
        while len(smls) < 1000:
            smls.append(line)

print("Encoding SMILES...")
x_split = [split(sm) for sm in smls]
xid, _ = get_array(x_split)
X = trfm.encode(torch.t(xid))
print("Shape of encoding: (%d, %d)" % X.shape)
