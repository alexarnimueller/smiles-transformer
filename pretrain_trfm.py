import argparse
import math
import os

import numpy as np
import torch
from torch.nn import Module, Dropout, Embedding, Linear, Transformer
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import WordVocab, Seq2seqDataset, split


class PositionalEncoding(Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TrfmSmiles(Module):
    def __init__(self, in_size, hidden_size, out_size, n_layers, nhead=4, dropout=0.1):
        super(TrfmSmiles, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.embed = Embedding(in_size, hidden_size)
        self.pe = PositionalEncoding(hidden_size, dropout)
        self.trfm = Transformer(d_model=hidden_size, nhead=nhead, num_encoder_layers=n_layers,
                                num_decoder_layers=n_layers, dim_feedforward=hidden_size)
        self.out = Linear(hidden_size, out_size)

    def forward(self, src):
        # src: (T,B)
        embedded = self.embed(src)  # (T,B,H)
        embedded = self.pe(embedded)  # (T,B,H)
        hidden = self.trfm(embedded, embedded)  # (T,B,H)
        out = self.out(hidden)  # (T,B,V)
        out = F.log_softmax(out, dim=2)  # (T,B,V)
        return out  # (T,B,V)

    def _encode(self, src):
        # src: (T,B)
        embedded = self.embed(src)  # (T,B,H)
        embedded = self.pe(embedded)  # (T,B,H)
        output = embedded
        for i in range(self.trfm.encoder.num_layers - 1):
            output = self.trfm.encoder.layers[i](output, None)  # (T,B,H)
        penul = output.detach().numpy()
        output = self.trfm.encoder.layers[-1](output, None)  # (T,B,H)
        if self.trfm.encoder.norm:
            output = self.trfm.encoder.norm(output)  # (T,B,H)
        output = output.detach().numpy()
        # mean, max, first*2
        return np.hstack([np.mean(output, axis=0), np.max(output, axis=0), output[0, :, :], penul[0, :, :]])  # (B,4H)

    def encode(self, src):
        # src: (T,B)
        batch_size = src.shape[1]
        if batch_size <= 100:
            return self._encode(src)
        else:  # Batch is too large to load
            print('There are {:d} molecules. It will take a little time.'.format(batch_size))
            st, ed = 0, 100
            out = self._encode(src[:, st:ed])  # (B,4H)
            while ed < batch_size:
                st += 100
                ed += 100
                out = np.concatenate([out, self._encode(src[:, st:ed])], axis=0)
            return out


def evaluate(model, test_loader, vocab):
    model.eval()
    total_loss = 0
    for b, sm in enumerate(test_loader):
        sm = torch.t(sm.cuda())  # (T,B)
        with torch.no_grad():
            output = model(sm)  # (T,B,V)
        loss = F.nll_loss(output.view(-1, len(vocab)), sm.contiguous().view(-1), ignore_index=0)
        total_loss += loss.item()
    return float(total_loss) / float(len(test_loader))


def parse_arguments():
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--n_epoch', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--data', '-d', type=str, default='.data/chembl24.txt', help='train corpus')
    parser.add_argument('--out-dir', '-o', type=str, default='.save', help='output directory')
    parser.add_argument('--name', '-n', type=str, default='ST', help='model name')
    parser.add_argument('--seq_len', type=int, default=220, help='maximum length of the paired seqence')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='batch size')
    parser.add_argument('--n_worker', '-w', type=int, default=8, help='number of workers')
    parser.add_argument('--hidden', type=int, default=128, help='length of hidden vector')
    parser.add_argument('--n_layer', '-l', type=int, default=3, help='number of layers')
    parser.add_argument('--n_head', type=int, default=4, help='number of attention heads')
    parser.add_argument('--lr', type=float, default=1e-4, help='Adam learning rate')
    parser.add_argument('--test_size', '-t', default=0.1, help='size of test set')
    parser.add_argument('--gpu', metavar='N', type=int, nargs='+', help='list of GPU IDs to use')
    return parser.parse_args()


def main():
    args = parse_arguments()
    assert torch.cuda.is_available()

    print('\nLoading and preparing dataset...')
    with open(args.data, "r") as f:
        text = list()
        smls = list()
        for line in tqdm(f):
            sl = split(line)
            text.append(sl)
            if len(sl) <= (args.seq_len - 2):
                smls.append(line)
        vocab = WordVocab(text, min_freq=5)
    vocab.save_vocab('.data/vocab.pkl')
    print("\nVocab size: %d" % len(vocab))
    print(vocab.stoi)
    print("\n%d examples are comply with seq_len of %d" % (len(smls), args.seq_len))

    print("\nSplitting dataset into test and train...")
    dataset = Seq2seqDataset(smls, vocab, seq_len=args.seq_len)
    train, test = torch.utils.data.random_split(dataset, [len(dataset) - int(len(dataset) * args.test_size),
                                                          int(len(dataset) * args.test_size)])
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.n_worker)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.n_worker)
    print('\nTrain size:', len(train))
    print('Test size:', len(test))
    del dataset, train, test

    print("\nBuilding model...")
    model = TrfmSmiles(len(vocab), args.hidden, len(vocab), args.n_layer, nhead=args.n_head).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(model)
    print('\nTotal parameters:', sum(p.numel() for p in model.parameters()))

    print("\nTraining...")
    best_loss = None
    for epoch in range(1, 5):
        print("\n----- Epoch %d -----" % epoch)
        for batch, sm in tqdm(enumerate(train_loader)):
            sm = torch.t(sm.cuda())  # (T,B)
            optimizer.zero_grad()
            output = model(sm)  # (T,B,V)
            loss = F.nll_loss(output.view(-1, len(vocab)), sm.contiguous().view(-1), ignore_index=0)
            loss.backward()
            optimizer.step()
            if batch % 1000 == 0:
                print(' Train Ep.{:3d}: iter {:5d} | loss {:.4f} | ppl {:.4f}'.format(epoch, batch, loss.item(),
                                                                                      math.exp(loss.item())))
            if batch % 10000 == 0:
                loss = evaluate(model, test_loader, vocab)
                print(' Val Ep.{:3d}: iter {:5d} | loss {:.4f} | ppl {:.4f}'.format(epoch, batch, loss, math.exp(loss)))
                # Save the model if the validation loss is the best we've seen so far.
                if not best_loss or loss < best_loss:
                    print("[!] saving model...")
                    if not os.path.isdir("./.save"):
                        os.makedirs("./.save")
                    torch.save(model.state_dict(), './.save/trfm_%d_%d.pkl' % (epoch, batch))
                    best_loss = loss


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
