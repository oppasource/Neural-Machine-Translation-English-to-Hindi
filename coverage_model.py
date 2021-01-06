import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pdb


# Using gpu if available else cpu
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


class coverage(nn.Module):
    def __init__(self, embed_size, nHidden, en_vocab, hi_vocab, device):
        super(coverage, self).__init__()
        self.embed_size = embed_size
        self.nHidden = nHidden
        self.en_vocab = en_vocab
        self.hi_vocab = hi_vocab
        self.device = device
        self.d = nHidden

        self.encoder_embeds = nn.Embedding(en_vocab, embed_size)
        self.decoder_embeds = nn.Embedding(hi_vocab, embed_size)

        self.encoder_gru = nn.GRU(embed_size, nHidden, bidirectional = True)
        self.decoder_gru = nn.GRU(embed_size + 2 * nHidden, nHidden)

        self.coverage_rnn = nn.RNN(3 * self.nHidden, self.d)

        # Attention weights
        self.va = nn.Parameter(torch.randn(nHidden, 1), requires_grad = True)
        self.wa = nn.Parameter(torch.randn(nHidden, nHidden), requires_grad = True)
        self.ua = nn.Parameter(torch.randn(nHidden, 2*nHidden), requires_grad = True)
        self.vc = nn.Parameter(torch.randn(nHidden, self.d), requires_grad = True)

        # Linear layer for out probability
        self.lin = nn.Linear(2*nHidden + nHidden, hi_vocab)


    def forward(self, in_seq, out_seq):
        encoder_seq = self.encoder_embeds(in_seq).view(-1, 1, self.embed_size)
        recurrent_encoder, hidden = self.encoder_gru(encoder_seq)

        outputs = torch.autograd.Variable(torch.zeros(out_seq.size()[0], self.hi_vocab)).to(self.device)
        last_hidden = hidden[:self.decoder_gru.num_layers]

        sos = out_seq[0].unsqueeze(0)
        out_embeds = self.decoder_embeds(sos)

        # Calculating attention weights and getting context vector
        context, alphas = self.get_initial_context_vector(last_hidden, recurrent_encoder)

        # Concatinating embed and context and giving input to gru
        rnn_inp = torch.cat((context, out_embeds[0]))
        rnn_inp = rnn_inp.view(-1, 1, self.embed_size + 2 * self.nHidden)

        output, hidden = self.decoder_gru(rnn_inp, last_hidden)

        # getting the probability of output word
        t = torch.cat((output.view(-1), context))
        output = self.lin(t)

        outputs[0] = output

        out_seq = out_seq[1:]

        # Calculate coverage vector for next step
        cov_vector, cov_last = self.cal_coverage_vector(recurrent_encoder, last_hidden, alphas, torch.zeros(1, 1, self.nHidden).to(self.device))


        # iterating though each decoder step
        for i in range(out_seq.size()[0]):
            out = out_seq[i].unsqueeze(0)
            out_embeds = self.decoder_embeds(out)

            # Calculating attention weights and getting context vector(coverage vector is used as well)
            context, alphas = self.get_context_vector(last_hidden, recurrent_encoder, cov_vector)

            # Concatinating embed and context and giving input to gru
            rnn_inp = torch.cat((context, out_embeds[0]))
            rnn_inp = rnn_inp.view(-1, 1, self.embed_size + 2 * self.nHidden)

            output, hidden = self.decoder_gru(rnn_inp, last_hidden)

            # getting the probability of output word
            t = torch.cat((output.view(-1), context))
            output = self.lin(t)

            # current hidden becomes last hidden for next iteration
            last_hidden = hidden

            # Calculating coverage vector using alphas for next step
            cov_vector, cov_last = self.cal_coverage_vector(recurrent_encoder, last_hidden, alphas, cov_last)

            # record the output
            outputs[i+1] = output

        return outputs

    def get_initial_context_vector(self, last_hidden, recurrent_encoder):
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
        return c_ij, alpha_ij

    def cal_coverage_vector(self, recurrent_encoder, decoder_hidden, alphas, last_coverage):
        re = recurrent_encoder.squeeze(1)
        re = torch.transpose(re, 0 ,1)

        hidden = decoder_hidden[-1].transpose(0,1)
        hidden = hidden.repeat(1, re.size()[1])

        crnn_inp = torch.cat((re, hidden))
        crnn_inp = torch.mul(alphas.expand_as(crnn_inp), crnn_inp)
        crnn_inp = crnn_inp.transpose(0,1).view(-1, 1, self.nHidden * 3)

        coverage_output, last_coverage_hidden = self.coverage_rnn(crnn_inp)
        coverage_output = coverage_output[:-1]
        coverage_output = torch.cat((last_coverage, coverage_output))
        coverage_output = coverage_output.tanh()
        return coverage_output, last_coverage_hidden

    def get_context_vector(self, last_hidden, recurrent_encoder, coverage_vectors):
        # housekeeping
        lh = last_hidden[-1].transpose(0,1)
        re = recurrent_encoder.squeeze(1)
        re = torch.transpose(re, 0 ,1)

        coverage_vectors = coverage_vectors.squeeze(1).transpose(0,1)

        first_term = torch.mm(self.wa, lh)
        second_term = torch.mm(self.ua, re)
        third_term = torch.mm(self.vc, coverage_vectors)
    
        inter = torch.tanh(first_term.expand_as(second_term) + second_term + third_term)
        e_ij = torch.mm(self.va.transpose(0,1), inter)
        alpha_ij = e_ij.softmax(1)
        c_ij = torch.mul(alpha_ij.expand_as(re),re)
        c_ij = torch.sum(c_ij, 1)
        return c_ij, alpha_ij
        
        


# For debug purposes
# coverage_model = coverage(4, 5, 1000, 1000, device)
# inp = torch.tensor([3,6,23,65])
# target = torch.tensor([89,800,123])
# output = coverage_model(inp, target)
# pdb.set_trace()