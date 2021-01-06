import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pdb

# Using gpu if available else cpu
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


class seq2seq(nn.Module):
    def __init__(self, embed_size, nHidden, en_vocab, hi_vocab, device):
        super(seq2seq, self).__init__()
        self.device = device
        self.embed_size = embed_size

        self.encoder_embeds = nn.Embedding(en_vocab, embed_size)
        self.decoder_embeds = nn.Embedding(hi_vocab, embed_size)

        self.encoder_lstm = nn.LSTM(embed_size, nHidden, num_layers=4)
        self.decoder_lstm = nn.LSTM(embed_size, nHidden, num_layers=4)

        self.lin = nn.Linear(nHidden, hi_vocab)

    def forward(self, in_seq, out_seq):
        encoder_seq = self.encoder_embeds(in_seq).view(-1, 1, self.embed_size)
        recurrent_encoder, (hidden_n, c_n) = self.encoder_lstm(encoder_seq)

        decoder_seq = self.decoder_embeds(out_seq).view(-1, 1, self.embed_size)
        recurrent_decoder, (hidden, c) = self.decoder_lstm(decoder_seq, (hidden_n, c_n))

        out = self.lin(recurrent_decoder)

        
        s,b,o = out.size()
        out = out.view(s*b,o)
        return out

# For debugging
# seq2seq_model = seq2seq(4, 5, 1000, 1000, device)
# inp = torch.tensor([3,6,23,65])
# target = torch.tensor([89,800,123])
# output = seq2seq_model(inp, target)


class attention(nn.Module):
    def __init__(self, embed_size, nHidden, en_vocab, hi_vocab, device):
        super(attention, self).__init__()
        self.embed_size = embed_size
        self.nHidden = nHidden
        self.en_vocab = en_vocab
        self.hi_vocab = hi_vocab
        self.device = device

        self.encoder_embeds = nn.Embedding(en_vocab, embed_size)
        self.decoder_embeds = nn.Embedding(hi_vocab, embed_size)

        self.encoder_gru = nn.GRU(embed_size, nHidden, bidirectional = True)
        self.decoder_gru = nn.GRU(embed_size + 2 * nHidden, nHidden)

        # Attention weights
        self.va = nn.Parameter(torch.randn(nHidden, 1), requires_grad = True)
        self.wa = nn.Parameter(torch.randn(nHidden, nHidden), requires_grad = True)
        self.ua = nn.Parameter(torch.randn(nHidden, 2*nHidden), requires_grad = True)

        # Linear layer for out probability
        self.lin = nn.Linear(2*nHidden + nHidden, hi_vocab)


    def forward(self, in_seq, out_seq):
        encoder_seq = self.encoder_embeds(in_seq).view(-1, 1, self.embed_size)
        recurrent_encoder, hidden = self.encoder_gru(encoder_seq)

        outputs = torch.autograd.Variable(torch.zeros(out_seq.size()[0], self.hi_vocab)).to(self.device)
        last_hidden = hidden[:self.decoder_gru.num_layers]

        # iterating though each decoder step
        for i in range(out_seq.size()[0]):
            out = out_seq[i].unsqueeze(0)
            out_embeds = self.decoder_embeds(out)

            # Calculating attention weights and getting context vector
            context = self.get_context_vector(last_hidden, recurrent_encoder)

            # Concatinating embed and context and giving input to gru
            rnn_inp = torch.cat((context, out_embeds[0]))
            rnn_inp = rnn_inp.view(-1, 1, self.embed_size + 2 * self.nHidden)

            output, hidden = self.decoder_gru(rnn_inp, last_hidden)

            # getting the probability of output word
            t = torch.cat((output.view(-1), context))
            output = self.lin(t)

            # current hidden becomes last hidden for next iteration
            last_hidden = hidden

            # record the output
            outputs[i] = output

        return outputs

    def get_context_vector(self, last_hidden, recurrent_encoder):
        # housekeeping
        lh = last_hidden[-1].transpose(0,1)
        re = recurrent_encoder.squeeze(1)
        re = torch.transpose(re, 0 ,1)

        first_term = torch.mm(self.wa, lh)
        second_term = torch.mm(self.ua, re)
        inter = torch.tanh(first_term.expand_as(second_term) + second_term)
        e_ij = torch.mm(self.va.transpose(0,1), inter)
        alpha_ij = e_ij.softmax(1)
        c_ij = torch.mul(alpha_ij.expand_as(re),re)
        c_ij = torch.sum(c_ij, 1)
        return c_ij

# attention_model = attention(4, 5, 1000, 1000, device)
# inp = torch.tensor([3,6,23,65])
# target = torch.tensor([89,800,123])
# output = attention_model(inp, target)


class global_attention_1(nn.Module):
    def __init__(self, embed_size, nHidden, en_vocab, hi_vocab, device):
        super(global_attention_1, self).__init__()
        self.embed_size = embed_size
        self.nHidden = nHidden
        self.en_vocab = en_vocab
        self.hi_vocab = hi_vocab
        self.device = device

        self.encoder_embeds = nn.Embedding(en_vocab, embed_size)
        self.decoder_embeds = nn.Embedding(hi_vocab, embed_size)

        self.encoder_gru = nn.GRU(embed_size, nHidden)
        self.decoder_gru = nn.GRU(embed_size + nHidden, nHidden)

        # Attention weights
        self.wc = nn.Parameter(torch.randn(nHidden, 2*nHidden), requires_grad = True)

        # Linear layer for out probability
        self.lin = nn.Linear(nHidden, hi_vocab)


    def forward(self, in_seq, out_seq):
        encoder_seq = self.encoder_embeds(in_seq).view(-1, 1, self.embed_size)
        recurrent_encoder, hidden = self.encoder_gru(encoder_seq)

        outputs = torch.autograd.Variable(torch.zeros(out_seq.size()[0], self.hi_vocab)).to(self.device)

        # Initial decoding
        sos = out_seq[0].unsqueeze(0)
        out_embeds = self.decoder_embeds(sos)
        rnn_inp = torch.cat((out_embeds[0], torch.zeros(self.nHidden).to(device)))
        rnn_inp = rnn_inp.view(1, 1, -1)

        output, hidden = self.decoder_gru(rnn_inp)
        # Calculating attention weights and getting context vector
        context = self.get_context_vector(hidden, recurrent_encoder)
        # dropping axis
        hidden_initial = hidden[0][0]

        ht_bar = torch.mm(self.wc, torch.cat((context, hidden_initial)).unsqueeze(1)).tanh()
        output = self.lin(ht_bar.transpose(0,1))
        outputs[0] = output
 
        out_seq = out_seq[1:]

        # iterating though each decoder step
        for i in range(out_seq.size()[0]):
            out = out_seq[i].unsqueeze(0)
            out_embeds = self.decoder_embeds(out)

            rnn_inp = torch.cat((out_embeds[0], ht_bar.transpose(0,1)[0]))
            rnn_inp = rnn_inp.view(1, 1, -1)

            output, hidden = self.decoder_gru(rnn_inp)
            
            # Calculating attention weights and getting context vector
            context = self.get_context_vector(hidden, recurrent_encoder)
            # dropping axis
            hidden = hidden[0][0]

            ht_bar = torch.mm(self.wc, torch.cat((context, hidden)).unsqueeze(1)).tanh()
            output = self.lin(ht_bar.transpose(0,1))

            # record the output
            outputs[i+1] = output

        return outputs

    def get_context_vector(self, hidden, recurrent_encoder):
        # housekeeping
        re = recurrent_encoder.squeeze(1)
        re = torch.transpose(re, 0 ,1)

        eij = torch.mm(hidden[-1], re)
        alpha_ij = eij.softmax(1)
        c_ij = torch.mul(alpha_ij.expand_as(re),re)
        c_ij = torch.sum(c_ij, 1)
        return c_ij

# global_attention_model = global_attention_1(4, 5, 1000, 1000, device)
# inp = torch.tensor([3,6,23,65])
# target = torch.tensor([89,800,123])
# output = global_attention_model(inp, target)
# pdb.set_trace()

class global_attention_2(nn.Module):
    def __init__(self, embed_size, nHidden, en_vocab, hi_vocab, device):
        super(global_attention_2, self).__init__()
        self.embed_size = embed_size
        self.nHidden = nHidden
        self.en_vocab = en_vocab
        self.hi_vocab = hi_vocab
        self.device = device

        self.encoder_embeds = nn.Embedding(en_vocab, embed_size)
        self.decoder_embeds = nn.Embedding(hi_vocab, embed_size)

        self.encoder_gru = nn.GRU(embed_size, nHidden)
        self.decoder_gru = nn.GRU(embed_size + nHidden, nHidden)

        # Attention weights
        self.wc = nn.Parameter(torch.randn(nHidden, 2*nHidden), requires_grad = True)
        self.wa = nn.Parameter(torch.randn(nHidden, nHidden), requires_grad = True)

        # Linear layer for out probability
        self.lin = nn.Linear(nHidden, hi_vocab)


    def forward(self, in_seq, out_seq):
        encoder_seq = self.encoder_embeds(in_seq).view(-1, 1, self.embed_size)
        recurrent_encoder, hidden = self.encoder_gru(encoder_seq)

        outputs = torch.autograd.Variable(torch.zeros(out_seq.size()[0], self.hi_vocab)).to(self.device)

        # Initial decoding
        sos = out_seq[0].unsqueeze(0)
        out_embeds = self.decoder_embeds(sos)
        rnn_inp = torch.cat((out_embeds[0], torch.zeros(self.nHidden).to(device)))
        rnn_inp = rnn_inp.view(1, 1, -1)

        output, hidden = self.decoder_gru(rnn_inp)
        # Calculating attention weights and getting context vector
        context = self.get_context_vector(hidden, recurrent_encoder)
        # dropping axis
        hidden_initial = hidden[0][0]

        ht_bar = torch.mm(self.wc, torch.cat((context, hidden_initial)).unsqueeze(1)).tanh()
        output = self.lin(ht_bar.transpose(0,1))
        outputs[0] = output
        
        out_seq = out_seq[1:]

        # iterating though each decoder step
        for i in range(out_seq.size()[0]):
            out = out_seq[i].unsqueeze(0)
            out_embeds = self.decoder_embeds(out)

            rnn_inp = torch.cat((out_embeds[0], ht_bar.transpose(0,1)[0]))
            rnn_inp = rnn_inp.view(1, 1, -1)

            output, hidden = self.decoder_gru(rnn_inp)
            
            # Calculating attention weights and getting context vector
            context = self.get_context_vector(hidden, recurrent_encoder)
            # dropping axis
            hidden = hidden[0][0]

            ht_bar = torch.mm(self.wc, torch.cat((context, hidden)).unsqueeze(1)).tanh()
            output = self.lin(ht_bar.transpose(0,1))

            # record the output
            outputs[i+1] = output

        return outputs

    def get_context_vector(self, hidden, recurrent_encoder):
        # housekeeping
        re = recurrent_encoder.squeeze(1)
        re = torch.transpose(re, 0 ,1)

        e_ij = torch.mm(torch.mm(hidden[-1], self.wa), re)

        alpha_ij = e_ij.softmax(1)
        c_ij = torch.mul(alpha_ij.expand_as(re),re)
        c_ij = torch.sum(c_ij, 1)
        return c_ij

# global_attention_model = global_attention_2(4, 5, 1000, 1000, device)
# inp = torch.tensor([3,6,23,65])
# target = torch.tensor([89,800,123])
# output = global_attention_model(inp, target)


class global_attention_3(nn.Module):
    def __init__(self, embed_size, nHidden, en_vocab, hi_vocab, device):
        super(global_attention_3, self).__init__()
        self.embed_size = embed_size
        self.nHidden = nHidden
        self.en_vocab = en_vocab
        self.hi_vocab = hi_vocab
        self.device = device

        self.encoder_embeds = nn.Embedding(en_vocab, embed_size)
        self.decoder_embeds = nn.Embedding(hi_vocab, embed_size)

        self.encoder_gru = nn.GRU(embed_size, nHidden)
        self.decoder_gru = nn.GRU(embed_size + nHidden, nHidden)

        # Attention weights
        self.wc = nn.Parameter(torch.randn(nHidden, 2*nHidden), requires_grad = True)
        self.wa = nn.Parameter(torch.randn(1, 2*nHidden), requires_grad = True)

        # Linear layer for out probability
        self.lin = nn.Linear(nHidden, hi_vocab)


    def forward(self, in_seq, out_seq):
        encoder_seq = self.encoder_embeds(in_seq).view(-1, 1, self.embed_size)
        recurrent_encoder, hidden = self.encoder_gru(encoder_seq)

        outputs = torch.autograd.Variable(torch.zeros(out_seq.size()[0], self.hi_vocab)).to(self.device)

        # Initial decoding
        sos = out_seq[0].unsqueeze(0)
        out_embeds = self.decoder_embeds(sos)
        rnn_inp = torch.cat((out_embeds[0], torch.zeros(self.nHidden).to(device)))
        rnn_inp = rnn_inp.view(1, 1, -1)

        output, hidden = self.decoder_gru(rnn_inp)
        # Calculating attention weights and getting context vector
        context = self.get_context_vector(hidden, recurrent_encoder)
        # dropping axis
        hidden_initial = hidden[0][0]

        ht_bar = torch.mm(self.wc, torch.cat((context, hidden_initial)).unsqueeze(1)).tanh()
        output = self.lin(ht_bar.transpose(0,1))
        outputs[0] = output
        
        out_seq = out_seq[1:]

        # iterating though each decoder step
        for i in range(out_seq.size()[0]):
            out = out_seq[i].unsqueeze(0)
            out_embeds = self.decoder_embeds(out)

            rnn_inp = torch.cat((out_embeds[0], ht_bar.transpose(0,1)[0]))
            rnn_inp = rnn_inp.view(1, 1, -1)

            output, hidden = self.decoder_gru(rnn_inp)
            
            # Calculating attention weights and getting context vector
            context = self.get_context_vector(hidden, recurrent_encoder)
            # dropping axis
            hidden = hidden[0][0]

            ht_bar = torch.mm(self.wc, torch.cat((context, hidden)).unsqueeze(1)).tanh()
            output = self.lin(ht_bar.transpose(0,1))

            # record the output
            outputs[i+1] = output

        return outputs

    def get_context_vector(self, hidden, recurrent_encoder):
        # housekeeping
        re = recurrent_encoder.squeeze(1)
        re_trans = torch.transpose(re, 0 ,1)

        hidden = hidden[-1].expand_as(re)
        concatenated = torch.cat((hidden, re), 1)
        concatenated = torch.transpose(concatenated, 0, 1)

        e_ij = torch.mm(self.wa, concatenated)
        alpha_ij = e_ij.softmax(1)

        c_ij = torch.mul(alpha_ij.expand_as(re_trans),re_trans)
        c_ij = torch.sum(c_ij, 1)
        return c_ij

# global_attention_model = global_attention_3(4, 5, 1000, 1000, device)
# inp = torch.tensor([3,6,23,65])
# target = torch.tensor([89,800,123])
# output = global_attention_model(inp, target)

