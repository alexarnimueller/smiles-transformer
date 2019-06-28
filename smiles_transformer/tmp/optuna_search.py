import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import optuna
from adabound import AdaBound

from bert import BERT, BERTLM
from dataset import STDataset
from build_vocab import WordVocab
import numpy as np
import utils
PAD = 0

class STTrainer:
    def __init__(self, optim, bert, vocab_size, train_dataloader, test_dataloader,
                 log_freq=10, gpu_ids=[], vocab=None):
        """
        :param bert: BERT model
        :param vocab_size: vocabに含まれるトータルの単語数
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: 学習率
        :param betas: Adam optimizer betas
        :param with_cuda: traning with cuda
        :param log_freq: logを表示するiterationの頻度
        """

        # GPU環境において、GPUを指定しているかのフラグ
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert = bert
        self.model = BERTLM(bert, vocab_size).to(self.device)

        if self.device == 'cuda':
            self.model = nn.DataParallel(self.model, gpu_ids)

        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.optim = optim
        self.criterion = nn.NLLLoss()

        self.log_freq = log_freq
        self.vocab = vocab

    def iteration(self, data, train=True):
        """
        :param epoch: 現在のepoch
        :param data_loader: torch.utils.data.DataLoader
        :param train: trainかtestかのbool値
        """
        data = {key: value.to(self.device) for key, value in data.items()}
        tsm, msm = self.model.forward(data["bert_input"], data["segment_embd"])
        loss_tsm = self.criterion(tsm, data["is_same"])
        loss_msm = self.criterion(msm.transpose(1, 2), data["bert_label"])
        filleds = utils.sample(msm)
        smiles = []
        for filled in filleds:
            s1, s2 = self.num2str(filled)
            smiles.append(s1)
            smiles.append(s2)
        validity = utils.validity(smiles) * 100
        loss = loss_tsm + loss_msm
        if train:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        # TSM prediction accuracy
        n = data["is_same"].nelement()
        acc_tsm = tsm.argmax(dim=-1).eq(data["is_same"]).sum().item()  / n * 100
        acc_msm = filleds.eq(data['bert_label']).sum().item() / 220  / n * 100
    
        return loss.item(), loss_tsm.item(), loss_msm.item(), acc_tsm, acc_msm, validity
    
    def num2str(self, nums):
        s = [self.vocab.itos[num] for num in nums]
        s = ''.join(s).replace('<pad>', '')
        ss = s.split('<eos>')
        if len(ss)>=2:
            return ss[0], s[1]
        else:
            sep = len(s)//2
            return s[:sep], s[sep:]

    
def get_trainer(trial, args, vocab, train_data_loader, test_data_loader):
    hidden = 256
    n_layers = [4, 5, 6, 7, 8]
    n_layer = trial.suggest_categorical('n_layer', n_layers)
    n_heads = [2, 4, 8]
    n_head = trial.suggest_categorical('n_head', n_heads)

    vocab_size = len(vocab)
    bert = BERT(vocab_size, hidden=hidden, n_layers=n_layer, attn_heads=n_head, dropout=args.dropout)
    bert.cuda()

    lr = trial.suggest_loguniform('lr', 1e-6, 1e-3)
    final_lr = trial.suggest_loguniform('final_lr', 1e-4, 1e-1)
    optim = AdaBound(BERTLM(bert, vocab_size).parameters(), lr=lr, final_lr=final_lr)

    trainer = STTrainer(optim, bert, vocab_size, train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                        gpu_ids=args.gpu, vocab=vocab)
    return trainer


def main():
    parser = argparse.ArgumentParser(description='Pretrain SMILES Transformer')
    parser.add_argument('--n_epoch', '-e', type=int, default=300, help='number of epochs')
    parser.add_argument('--n_trial', '-t', type=int, default=100, help='number of optuna trials')
    parser.add_argument('--vocab', '-v', type=str, default='data/vocab.pkl', help='vocabulary (.pkl)')
    parser.add_argument('--train_data', type=str, default='data/chembl24_bert_train.csv', help='train corpus (.csv)')
    parser.add_argument('--test_data', type=str, default='data/chembl24_bert_test.csv', help='test corpus (.csv)')
    parser.add_argument('--name', '-n', type=str, default='ST', help='model name')
    parser.add_argument('--seq_len', type=int, default=220, help='maximum length of the paired seqence')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='batch size')
    parser.add_argument('--n_worker', '-w', type=int, default=16, help='number of workers')
    parser.add_argument('--dropout', '-d', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta2')
    parser.add_argument('--gpu', metavar='N', type=int, nargs='+', help='list of GPU IDs to use')
    args = parser.parse_args()

    vocab = WordVocab.load_vocab(args.vocab)
    train_dataset = STDataset(args.train_data, vocab, seq_len=args.seq_len)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_worker)

    def objective(trial):
        trainer = get_trainer(trial, args, vocab, train_data_loader, None)
        data_iter = tqdm(enumerate(train_data_loader), total=len(train_data_loader), bar_format="{l_bar}{r_bar}")
        l, a1, a2, v = 0, 0, 0, 0
        for iter, data in data_iter:
            if iter>10000:
                break
            loss, _, __, acc_tsm, acc_msm, validity = trainer.iteration(data)
            l += loss
            a1 += acc_tsm
            a2 += acc_msm
            v += validity
        print('2SM: {:.3f}, MSM: {:.3f}, VAL: {:.3f}'.format(a1/10000, a2/10000, v/10000))
        return l/100

    study = optuna.create_study()
    study.optimize(objective, n_trials=args.n_trial)

if __name__=='__main__':
    main()