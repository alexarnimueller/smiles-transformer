import math
import time
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.2):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self, initrange=0.1):
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
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


def batchify(data, btchsz):
    data = tokenizer(data, mode='encode')
    # Divide the dataset into btchsz parts.
    nbatch = data.size(0) // btchsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * btchsz)
    # Evenly divide the data across the btchsz batches.
    data = data.view(btchsz, -1).t().contiguous()
    return data.to(device)


def get_batch(source, i, bptt=35):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


def pad(sequences, maxlen=0):
    if maxlen == 0:
        length = max(map(len, sequences))
    else:
        sequences = filter(lambda x: len(x) < maxlen+1, sequences)
        length = maxlen
    return ['^' + seq + ' ' * (length - len(seq)) for seq in sequences]


def tokenizer(arr, mode='encode'):
    itos = {"0": 'H', "1": '9', "2": 'D', "3": 'r', "4": 'T', "5": 'R', "6": 'V', "7": '4', "8": 'c', "9": 'l',
            "10": 'b', "11": '.', "12": 'C', "13": 'Y', "14": 's', "15": 'B', "16": 'k', "17": '+', "18": 'p',
            "19": '2', "20": '7', "21": '8', "22": 'O', "23": '%', "24": 'o', "25": '6', "26": 'N', "27": 'A',
            "28": 't', "29": '$', "30": '(', "31": 'u', "32": 'Z', "33": '#', "34": 'M', "35": 'P', "36": 'G',
            "37": 'I', "38": '=', "39": '-', "40": 'X', "41": '@', "42": 'E', "43": ':', "44": '\\', "45": ')',
            "46": 'i', "47": 'K', "48": '/', "49": '{', "50": 'h', "51": 'L', "52": 'n', "53": 'U', "54": '[',
            "55": '0', "56": 'y', "57": 'e', "58": '3', "59": 'g', "60": 'f', "61": '}', "62": '1', "63": 'd',
            "64": 'W', "65": '5', "66": 'S', "67": 'F', "68": ']', "69": 'a', "70": 'm', "71": '^', "72": ' '}
    stoi = {v: k for k, v in itos.items()}
    if mode == 'encode':
        return torch.tensor([[stoi[x] for x in ex] for ex in arr], dtype=torch.long, device=device).contiguous()
    elif mode == 'decode':
        return [''.join([x for x in ex]).strip().replace('^', '') for ex in arr]
    else:
        raise NotImplementedError("Only 'encode' and 'decode are available as modes!")


def train(epoch, model, optimizer, train_data, scheduler, criterion=nn.CrossEntropyLoss(), bptt=35, ntok=73):
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, ntok), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'
                  .format(epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0], elapsed * 1000 / log_interval,
                          cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def evaluate(eval_model, data_source, criterion=nn.CrossEntropyLoss(), bptt=35, ntokens=73):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)




# batch_size = 20
# eval_batch_size = 10
# train_data = batchify(train_txt, batch_size)
# val_data = batchify(val_txt, eval_batch_size)
# test_data = batchify(test_txt, eval_batch_size)
#
# emsize = 200  # embedding dimension
# nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
# nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
# nhead = 2  # the number of heads in the multiheadattention models
# dropout = 0.2  # the dropout value
# model = TransformerModel(73, emsize, nhead, nhid, nlayers, dropout).to(device)
#
# lr = 5.0  # learning rate
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
#
# best_val_loss = float("inf")
# epochs = 3  # The number of epochs
# best_model = None
#
# for epoch in range(1, epochs + 1):
#     epoch_start_time = time.time()
#     train()
#     val_loss = evaluate(model, val_data)
#     print('-' * 89)
#     print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
#           'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
#     print('-' * 89)
#
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         best_model = model
#
#     scheduler.step()


