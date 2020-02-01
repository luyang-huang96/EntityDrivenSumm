import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from .rnn import MultiLayerLSTMCells
from .rnn import lstm_encoder
from .util import sequence_mean, len_mask
from .attention import prob_normalize, prob_normalize_sigmoid
from .myownutils import get_sinusoid_encoding_table
import math

INI = 1e-2

class ConvSentEncoder(nn.Module):
    """
    Convolutional word-level sentence encoder
    w/ max-over-time pooling, [3, 4, 5] kernel sizes, ReLU activation
    """
    def __init__(self, vocab_size, emb_dim, n_hidden, dropout, pe=False, petrainable=False, pembedding=None, embedding=None):
        super().__init__()
        if embedding is None:
            self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        else:
            self._embedding = embedding
        self._convs = nn.ModuleList([nn.Conv1d(emb_dim, n_hidden, i)
                                     for i in range(3, 6)])
        if pe and (not petrainable):
            # sentence level pos enc
            self.poisition_enc = nn.Embedding.from_pretrained(
                get_sinusoid_encoding_table(512, emb_dim, padding_idx=0),
                freeze=True)
        elif pe:
            assert pembedding is not None
            self.poisition_enc = pembedding
        self._dropout = dropout
        self._grad_handle = None
        self.position_encoder = pe



    def forward(self, input_):
        if self.position_encoder:
            mask_ = input_.gt(0)
            sent_len = mask_.sum(1)
            sent_num, max_sent_length = input_.size()
            src_pos = torch.zeros(sent_num, max_sent_length).long().to(input_.device)
            #total_word_num = 0
            for i in range(sent_num):
                #src_pos[i, :] = torch.arange(1, max_sent_length + 1) + total_word_num
                #total_word_num += sent_len[i]
                src_pos[i, :] = torch.arange(1, max_sent_length + 1)
            src_pos = src_pos * mask_.long()
            src_pos[ src_pos > 512] = 512 # in order for
            emb_input = self._embedding(input_) + self.poisition_enc(src_pos)
        else:
            emb_input = self._embedding(input_)

        conv_in = F.dropout(emb_input.transpose(1, 2),
                            self._dropout, training=self.training)
        output = torch.cat([F.relu(conv(conv_in)).max(dim=2)[0]
                            for conv in self._convs], dim=1)
        return output

    def set_embedding(self, embedding):
        """embedding is the weight matrix"""
        assert self._embedding.weight.size() == embedding.size()
        self._embedding.weight.data.copy_(embedding)


class ConvEntityEncoder(nn.Module):
    """
    Convolutional word-level sentence encoder
    w/ max-over-time pooling, [3, 4, 5] kernel sizes, ReLU activation
    """
    def __init__(self, vocab_size, emb_dim, n_hidden, dropout, pe=False, embedding=None, context=False,
                 context_hidden=0, context_input_dim=0):
        super().__init__()
        if embedding is None:
            self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        else:
            self._embedding = embedding
        if context:
            self.context_dim = context_hidden
            self._linear = nn.Linear(context_input_dim, context_hidden)
            self._convs = nn.ModuleList([nn.Conv1d(emb_dim + self.context_dim, n_hidden, i)
                                         for i in range(2, 5)])
        else:
            self._convs = nn.ModuleList([nn.Conv1d(emb_dim, n_hidden, i)
                                     for i in range(2, 5)])
        if pe:
            self.poisition_enc = nn.Embedding.from_pretrained(
                get_sinusoid_encoding_table(512, emb_dim, padding_idx=0),
                freeze=True)
        self._dropout = dropout
        self._grad_handle = None
        self.position_encoder = pe
        self._context = context


    def forward(self, input_, input_wpos, input_spos, context=None):
        if self.position_encoder:
            input_wpos[input_wpos > 512] = 512
            emb_input = self._embedding(input_) + self.poisition_enc(input_wpos)
        else:
            emb_input = self._embedding(input_)
        if self._context:
            context_dim = context.size(1)
            context = torch.cat(
                [
                    torch.zeros(1, context_dim).to(context.device),
                    context
                ],
                dim = 0
            )
            # print('embed_input:', emb_input.size())
            # print('input_spos:', input_spos.size())
            # print('context:', context.size())
            n_side, n_word = input_spos.size()
            spos = input_spos.view(-1)
            context_embed = context.index_select(0, spos).view(n_side, n_word, -1)
            context_embed = self._linear(context_embed)
            emb_input = torch.cat([emb_input, context_embed], dim=2)
            # print('check:', context)
            # print('check:', input_spos)
            # print('check:', context_embed)
            # print(emb_input.size())

        conv_in = F.dropout(emb_input.transpose(1, 2),
                            self._dropout, training=self.training)
        # conv_in = emb_input.transpose(1, 2)
        # output = torch.cat([F.tanh(conv(conv_in)).max(dim=2)[0]
        #                     for conv in self._convs], dim=1)
        output = torch.cat([F.relu(conv(conv_in)).max(dim=2)[0]
                            for conv in self._convs], dim=1)
        # print('output:', output)
        # print(output.size())
        return output

    def set_embedding(self, embedding):
        """embedding is the weight matrix"""
        assert self._embedding.weight.size() == embedding.size()
        self._embedding.weight.data.copy_(embedding)



class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, n_hidden, n_layer, dropout, bidirectional):
        super().__init__()
        self._init_h = nn.Parameter(
            torch.Tensor(n_layer*(2 if bidirectional else 1), n_hidden))
        self._init_c = nn.Parameter(
            torch.Tensor(n_layer*(2 if bidirectional else 1), n_hidden))
        init.uniform_(self._init_h, -INI, INI)
        init.uniform_(self._init_c, -INI, INI)
        self._lstm = nn.LSTM(input_dim, n_hidden, n_layer,
                             dropout=dropout, bidirectional=bidirectional)

    def forward(self, input_, in_lens=None, return_states=False):
        """ [batch_size, max_num_sent, input_dim] Tensor"""
        size = (self._init_h.size(0), input_.size(0), self._init_h.size(1))
        init_states = (self._init_h.unsqueeze(1).expand(*size),
                       self._init_c.unsqueeze(1).expand(*size))
        lstm_out, final_states = lstm_encoder(
            input_, self._lstm, in_lens, init_states)
        if return_states:
            return lstm_out.transpose(0, 1), final_states
        else:
            return lstm_out.transpose(0, 1)

    @property
    def input_size(self):
        return self._lstm.input_size

    @property
    def hidden_size(self):
        return self._lstm.hidden_size

    @property
    def num_layers(self):
        return self._lstm.num_layers

    @property
    def bidirectional(self):
        return self._lstm.bidirectional


class ExtractSumm(nn.Module):
    """ ff-ext """
    def __init__(self, vocab_size, emb_dim,
                 conv_hidden, lstm_hidden, lstm_layer,
                 bidirectional, dropout=0.0, pe=False, petrainable=False, stop=False):
        super().__init__()
        self._sent_enc = ConvSentEncoder(
            vocab_size, emb_dim, conv_hidden, dropout, pe, petrainable)
        self._art_enc = LSTMEncoder(
            3*conv_hidden, lstm_hidden, lstm_layer,
            dropout=dropout, bidirectional=bidirectional
        )

        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self._sent_linear = nn.Linear(lstm_out_dim, 1)
        self._art_linear = nn.Linear(lstm_out_dim, lstm_out_dim)

    def forward(self, article_sents, sent_nums):
        enc_sent, enc_art = self._encode(article_sents, sent_nums)
        saliency = torch.matmul(enc_sent, enc_art.unsqueeze(2))
        saliency = torch.cat(
            [s[:n] for s, n in zip(saliency, sent_nums)], dim=0)
        content = self._sent_linear(
            torch.cat([s[:n] for s, n in zip(enc_sent, sent_nums)], dim=0)
        )
        logit = (content + saliency).squeeze(1)
        return logit

    def extract(self, article_sents, sent_nums=None, k=4, force_ext=True):
        """ extract top-k scored sentences from article (eval only)"""
        enc_sent, enc_art = self._encode(article_sents, sent_nums)
        saliency = torch.matmul(enc_sent, enc_art.unsqueeze(2))
        content = self._sent_linear(enc_sent)
        logit = (content + saliency).squeeze(2)
        if force_ext:
            if sent_nums is None:  # test-time extract only
                assert len(article_sents) == 1
                n_sent = logit.size(1)
                extracted = logit[0].topk(
                    k if k < n_sent else n_sent, sorted=False  # original order
                )[1].tolist()
            else:
                extracted = [l[:n].topk(k if k < n else n)[1].tolist()
                             for n, l in zip(sent_nums, logit)]
        else:
            logit = F.sigmoid(logit)
            extracted = logit.gt(0.2)
            if extracted.sum() < 1:
                extracted = logit[0].topk(1)[1].tolist()
            else:
                extracted = [i for i, x in enumerate(extracted[0].tolist()) if x == 1]
        return extracted

    def _encode(self, article_sents, sent_nums):
        if sent_nums is None:  # test-time extract only
            enc_sent = self._sent_enc(article_sents[0]).unsqueeze(0)
        else:
            max_n = max(sent_nums)
            enc_sents = [self._sent_enc(art_sent)
                         for art_sent in article_sents]
            def zero(n, device):
                z = torch.zeros(n, self._art_enc.input_size).to(device)
                return z
            enc_sent = torch.stack(
                [torch.cat([s, zero(max_n-n, s.device)],
                           dim=0) if n != max_n
                 else s
                 for s, n in zip(enc_sents, sent_nums)],
                dim=0
            )
        lstm_out = self._art_enc(enc_sent, sent_nums)
        enc_art = F.tanh(
            self._art_linear(sequence_mean(lstm_out, sent_nums, dim=1)))
        return lstm_out, enc_art

    def set_embedding(self, embedding):
        self._sent_enc.set_embedding(embedding)



class NNSESumm(nn.Module):
    """ ff-ext """
    def __init__(self, vocab_size, emb_dim,
                 conv_hidden, lstm_hidden, lstm_layer,
                 bidirectional, dropout=0.0, pe=False, petrainable=False, stop=False):
        super().__init__()
        self._sent_enc = ConvSentEncoder(
            vocab_size, emb_dim, conv_hidden, dropout, pe=pe, petrainable=petrainable)
        self._art_enc = LSTMEncoder(
            3*conv_hidden, lstm_hidden, lstm_layer,
            dropout=dropout, bidirectional=bidirectional
        )

        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.LSTM = nn.LSTM(3*conv_hidden, lstm_out_dim, batch_first=True)
        self._sent_linear = nn.Linear(lstm_out_dim + lstm_out_dim, 1)
        self.step = 0

    @staticmethod
    def _fix_enc_hidden(hidden):
        # The encoder hidden is  (layers*directions) x batch x dim.
        # We need to convert it to layers x batch x (directions*dim).
        hidden = torch.cat([hidden[0:hidden.size(0):2],
                                hidden[1:hidden.size(0):2]], 2)
        return hidden

    def forward(self, article_sents, sent_nums, target):
        enc_sent, enc_art, hidden = self._encode(article_sents, sent_nums)
        batch_size, max_num_sent, input_dim = enc_sent.size()
        hidden = (self._fix_enc_hidden(hidden[0]), self._fix_enc_hidden(hidden[1]))
        all_decoder_output = torch.zeros(batch_size, max_num_sent, 1).to(enc_sent.device)
        j = 0
        teacher_forcing = torch.rand(1) > math.exp(- self.step / 10000)
        for i in range(max_num_sent):
            if i == 0:
                lstm_in = enc_art[:, i, :].unsqueeze(1)
            else:
                if teacher_forcing:
                    lstm_in = enc_art[:, i, :].unsqueeze(1) * F.sigmoid(output)
                else:
                    lstm_in = enc_art[:, i, :].unsqueeze(1) * target[:, i-1].unsqueeze(1).unsqueeze(1).float()
            output, hidden = self.LSTM(lstm_in, hidden)
            output = self._sent_linear(torch.cat([output, enc_sent[:, i, :].unsqueeze(1)], dim=2))
            all_decoder_output[:, i, :] = output.squeeze(1)
        #mask
        mask = target.ne(-1)
        final_output = torch.masked_select(all_decoder_output.squeeze(2), mask)
        self.step += 1

        return final_output

    def extract(self, article_sents, sent_nums=None, k=4, force_ext=True):
        """ extract top-k scored sentences from article (eval only)"""
        enc_sent, enc_art, hidden = self._encode(article_sents, sent_nums)
        batch_size, max_num_sent, input_dim = enc_sent.size()
        hidden = (self._fix_enc_hidden(hidden[0]), self._fix_enc_hidden(hidden[1]))
        all_decoder_output = torch.zeros(batch_size, max_num_sent, 1).to(enc_sent.device)
        for i in range(max_num_sent):
            if i == 0:
                lstm_in = enc_art[:, i, :].unsqueeze(1)
            else:
                lstm_in = enc_art[:, i, :].unsqueeze(1) * F.sigmoid(output)
            output, hidden = self.LSTM(lstm_in, hidden)
            output = self._sent_linear(torch.cat([output, enc_sent[:, i, :].unsqueeze(1)], dim=2))
            all_decoder_output[:, i, :] = output.squeeze(1)
        logit = all_decoder_output.squeeze(2)
        if force_ext:
            if sent_nums is None:
                assert len(article_sents) == 1
                n_sent = logit.size(1)
                extracted = logit[0].topk(
                    k if k < n_sent else n_sent, sorted=False  # original order
                )[1].tolist()
            else:
                extracted = [l[:n].topk(k if k < n else n)[1].tolist()
                             for n, l in zip(sent_nums, logit)]
        else:
            logit = F.sigmoid(logit)
            extracted = logit.gt(0.15)
            if extracted.sum() < 1:
                extracted = logit[0].topk(1)[1].tolist()
            else:
                extracted = [i for i, x in enumerate(extracted[0].tolist()) if x == 1]
        return extracted

    def _encode(self, article_sents, sent_nums):
        if sent_nums is None:  # test-time extract only
            enc_sent = self._sent_enc(article_sents[0]).unsqueeze(0)
        else:
            max_n = max(sent_nums)
            enc_sents = [self._sent_enc(art_sent)
                         for art_sent in article_sents]
            def zero(n, device):
                z = torch.zeros(n, self._art_enc.input_size).to(device)
                return z
            enc_sent = torch.stack(
                [torch.cat([s, zero(max_n-n, s.device)],
                           dim=0) if n != max_n
                 else s
                 for s, n in zip(enc_sents, sent_nums)],
                dim=0
            ) # [batch_size, max_num_sent, input_dim]
        lstm_out, final_hidden = self._art_enc(enc_sent, sent_nums, return_states=True)

        return lstm_out, enc_sent, final_hidden

    def set_embedding(self, embedding):
        self._sent_enc.set_embedding(embedding)


class LSTMPointerNet(nn.Module):
    """Pointer network as in Vinyals et al """
    def __init__(self, input_dim, n_hidden, n_layer,
                 dropout, n_hop, stop=False):
        super().__init__()
        self._init_h = nn.Parameter(torch.Tensor(n_layer, n_hidden))
        self._init_c = nn.Parameter(torch.Tensor(n_layer, n_hidden))
        self._init_i = nn.Parameter(torch.Tensor(input_dim))
        init.uniform_(self._init_h, -INI, INI)
        init.uniform_(self._init_c, -INI, INI)
        init.uniform_(self._init_i, -0.1, 0.1)
        self._lstm = nn.LSTM(
            input_dim, n_hidden, n_layer,
            bidirectional=False, dropout=dropout
        )
        self._lstm_cell = None

        # attention parameters
        self._attn_wm = nn.Parameter(torch.Tensor(input_dim, n_hidden))
        self._attn_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self._attn_v = nn.Parameter(torch.Tensor(n_hidden))
        init.xavier_normal_(self._attn_wm)
        init.xavier_normal_(self._attn_wq)
        init.uniform_(self._attn_v, -INI, INI)

        # hop parameters
        self._hop_wm = nn.Parameter(torch.Tensor(input_dim, n_hidden))
        self._hop_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self._hop_v = nn.Parameter(torch.Tensor(n_hidden))
        init.xavier_normal_(self._hop_wm)
        init.xavier_normal_(self._hop_wq)
        init.uniform_(self._hop_v, -INI, INI)
        self._n_hop = n_hop

        # stop token
        if stop:
            self._stop = nn.Parameter(torch.Tensor(input_dim))
            init.uniform_(self._stop, -INI, INI)
        self.stop = stop

    def forward(self, attn_mem, mem_sizes, lstm_in):
        """atten_mem: Tensor of size [batch_size, max_sent_num, input_dim]"""
        batch_size, max_sent_num, input_dim = attn_mem.size()
        if self.stop:
            attn_mem = torch.cat([attn_mem, torch.zeros(batch_size, 1, input_dim).to(attn_mem.device)], dim=1)
            for i, sent_num in enumerate(mem_sizes):
                attn_mem[i, sent_num, :] += self._stop
            mem_sizes = [mem_size+1 for mem_size in mem_sizes]
        attn_feat, hop_feat, lstm_states, init_i = self._prepare(attn_mem)
        lstm_in = torch.cat([init_i, lstm_in], dim=1).transpose(0, 1)
        query, final_states = self._lstm(lstm_in, lstm_states)
        query = query.transpose(0, 1)
        for _ in range(self._n_hop):
            query = LSTMPointerNet.attention(
                hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)


        output = LSTMPointerNet.attention_score(
            attn_feat, query, self._attn_v, self._attn_wq)
        return output  # unormalized extraction logit

    def extract(self, attn_mem, mem_sizes, k):
        """extract k sentences, decode only, batch_size==1"""
        batch_size = attn_mem.size(0)
        max_sent = attn_mem.size(1)
        if self.stop:
            attn_mem = torch.cat([attn_mem, self._stop.repeat(batch_size, 1).unsqueeze(1)], dim=1)
        attn_feat, hop_feat, lstm_states, lstm_in = self._prepare(attn_mem)
        lstm_in = lstm_in.squeeze(1)
        if self._lstm_cell is None:
            self._lstm_cell = MultiLayerLSTMCells.convert(
                self._lstm).to(attn_mem.device)
        extracts = []
        if self.stop:
            for sent_num in range(max_sent):
                h, c = self._lstm_cell(lstm_in, lstm_states)
                query = h[-1]
                for _ in range(self._n_hop):
                    query = LSTMPointerNet.attention(
                        hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)
                score = LSTMPointerNet.attention_score(
                    attn_feat, query, self._attn_v, self._attn_wq)
                score = score.squeeze()
                for e in extracts:
                    score[e] = -1e6
                ext = score.max(dim=0)[1].item()
                if ext == max_sent and sent_num != 0:
                    break
                elif sent_num == 0 and ext == max_sent:
                    ext = score.topk(2, dim=0)[1][1].item()
                extracts.append(ext)
                lstm_states = (h, c)
                lstm_in = attn_mem[:, ext, :]
        else:
            for _ in range(k):
                h, c = self._lstm_cell(lstm_in, lstm_states)
                query = h[-1]
                for _ in range(self._n_hop):
                    query = LSTMPointerNet.attention(
                        hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)
                score = LSTMPointerNet.attention_score(
                    attn_feat, query, self._attn_v, self._attn_wq)
                score = score.squeeze()
                for e in extracts:
                    score[e] = -1e6
                ext = score.max(dim=0)[1].item()
                extracts.append(ext)
                lstm_states = (h, c)
                lstm_in = attn_mem[:, ext, :]
        return extracts

    def sample(self, attn_mem, mem_sizes, k=4):
        assert self.stop == True
        """sample k sentences, decode only, batch_size==1"""
        batch_size = attn_mem.size(0)
        max_sent = attn_mem.size(1)
        if self.stop:
            attn_mem = torch.cat([attn_mem, self._stop.repeat(batch_size, 1).unsqueeze(1)], dim=1)
        attn_feat, hop_feat, lstm_states, lstm_in = self._prepare(attn_mem)
        lstm_in = lstm_in.squeeze(1)
        if self._lstm_cell is None:
            self._lstm_cell = MultiLayerLSTMCells.convert(
                self._lstm).to(attn_mem.device)
        extracts = []
        log_scores = []
        for sent_num in range(max_sent):
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[-1]
            for _ in range(self._n_hop):
                query = LSTMPointerNet.attention(
                    hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)
            score = LSTMPointerNet.attention_score(
                attn_feat, query, self._attn_v, self._attn_wq)
            score = score.squeeze()
            for e in extracts:
                score[e] = -1e6
            softmax_score = F.softmax(score)
            ext = softmax_score.multinomial(num_samples=1)
            _score = softmax_score.gather(0, ext)
            if ext.item() == max_sent and sent_num != 0:
                break
            elif sent_num == 0 and ext.item() == max_sent:
                # force model to sample a largest one
                while(ext.item() == max_sent):
                    ext = softmax_score.multinomial(num_samples=1)
                    _score = softmax_score.gather(0, ext)
            extracts.append(ext.item())
            log_scores.append(torch.log(_score))
            lstm_states = (h, c)
            lstm_in = attn_mem[:, ext.item(), :].squeeze(1)
        return extracts, log_scores

    def _prepare(self, attn_mem):
        attn_feat = torch.matmul(attn_mem, self._attn_wm.unsqueeze(0))
        hop_feat = torch.matmul(attn_mem, self._hop_wm.unsqueeze(0))
        bs = attn_mem.size(0)
        n_l, d = self._init_h.size()
        size = (n_l, bs, d)
        lstm_states = (self._init_h.unsqueeze(1).expand(*size).contiguous(),
                       self._init_c.unsqueeze(1).expand(*size).contiguous())
        d = self._init_i.size(0)
        init_i = self._init_i.unsqueeze(0).unsqueeze(1).expand(bs, 1, d)
        return attn_feat, hop_feat, lstm_states, init_i

    @staticmethod
    def attention_score(attention, query, v, w):
        """ unnormalized attention score"""
        sum_ = attention.unsqueeze(1) + torch.matmul(
            query, w.unsqueeze(0)
        ).unsqueeze(2)  # [B, Nq, Ns, D]
        score = torch.matmul(
            F.tanh(sum_), v.unsqueeze(0).unsqueeze(1).unsqueeze(3)
        ).squeeze(3)  # [B, Nq, Ns]
        return score

    @staticmethod
    def attention(attention, query, v, w, mem_sizes):
        """ attention context vector"""
        score = LSTMPointerNet.attention_score(attention, query, v, w)
        if mem_sizes is None:
            norm_score = F.softmax(score, dim=-1)
        else:
            mask = len_mask(mem_sizes, score.device).unsqueeze(-2)
            norm_score = prob_normalize(score, mask)
        output = torch.matmul(norm_score, attention)
        return output



class LSTMPointerNet_entity(nn.Module):
    """Pointer network as in Vinyals et al """
    def __init__(self, input_dim, n_hidden, n_layer,
                 dropout, n_hop, side_dim, stop, hard_attention=False):
        super().__init__()
        self._init_h = nn.Parameter(torch.Tensor(n_layer, n_hidden))
        self._init_c = nn.Parameter(torch.Tensor(n_layer, n_hidden))
        self._init_i = nn.Parameter(torch.Tensor(input_dim))
        init.uniform_(self._init_h, -INI, INI)
        init.uniform_(self._init_c, -INI, INI)
        init.uniform_(self._init_i, -0.1, 0.1)
        self._lstm = nn.LSTM(
            input_dim, n_hidden, n_layer,
            bidirectional=False, dropout=dropout
        )
        self._lstm_cell = None

        # attention parameters
        self._attn_wm = nn.Parameter(torch.Tensor(input_dim, n_hidden))
        self._attn_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self._attn_v = nn.Parameter(torch.Tensor(n_hidden))
        init.xavier_normal_(self._attn_wm)
        init.xavier_normal_(self._attn_wq)
        init.uniform_(self._attn_v, -INI, INI)

        # hop parameters
        self._hop_wm = nn.Parameter(torch.Tensor(input_dim, n_hidden))
        self._hop_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self._hop_v = nn.Parameter(torch.Tensor(n_hidden))
        init.xavier_normal_(self._hop_wm)
        init.xavier_normal_(self._hop_wq)
        init.uniform_(self._hop_v, -INI, INI)
        self._n_hop = n_hop

        # side info attention
        if not hard_attention:
            self.side_wm = nn.Parameter(torch.Tensor(side_dim, n_hidden))
            self.side_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
            self.side_v = nn.Parameter(torch.Tensor(n_hidden))
            init.xavier_normal_(self.side_wm)
            init.xavier_normal_(self.side_wq)
            init.uniform_(self.side_v, -INI, INI)
        else:
            self.side_wq = nn.Parameter(torch.Tensor(n_hidden, 1))
            self.side_wbi = nn.Bilinear(side_dim, side_dim, 1)
            init.xavier_normal_(self.side_wq)
            self._start = nn.Parameter(torch.Tensor(side_dim))
            init.uniform_(self._start)

        if not hard_attention:
            self._attn_ws = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
            init.xavier_normal_(self._attn_ws)
        else:
            self._attn_ws = nn.Parameter(torch.Tensor(side_dim, n_hidden))
            init.xavier_normal_(self._attn_ws)

        # pad entity
        self._pad_entity = nn.Parameter(torch.Tensor(side_dim))
        init.uniform_(self._pad_entity)

        # eos entity
        if hard_attention:
            self._eos_entity = nn.Parameter(torch.Tensor(side_dim))
            init.uniform_(self._eos_entity)

        # stop token
        if stop:
            self._stop = nn.Parameter(torch.Tensor(input_dim))
            init.uniform_(self._stop, -INI, INI)
        self.stop = stop

        self._hard_attention = hard_attention
        if self._hard_attention:
            self.side_dim = side_dim


    def forward(self, attn_mem, mem_sizes, lstm_in, side_mem, side_sizes, ground_entity=None):
        """atten_mem: Tensor of size [batch_size, max_sent_num, input_dim]"""
        if self._hard_attention:
            assert ground_entity is not None
        batch_size, max_sent_num, input_dim = attn_mem.size()
        side_dim = side_mem.size(2)
        if self.stop:
            attn_mem = torch.cat([attn_mem, torch.zeros(batch_size, 1, input_dim).to(attn_mem.device)], dim=1)
            for i, sent_num in enumerate(mem_sizes):
                attn_mem[i, sent_num, :] += self._stop
            mem_sizes = [mem_size+1 for mem_size in mem_sizes]
        if not self._hard_attention:
            side_mem = torch.cat([side_mem, torch.zeros(batch_size, 1, side_dim).to(side_mem.device)], dim=1) #b * ns * s
            for i, side_size in enumerate(side_sizes):
                side_mem[i, side_size, :] += self._pad_entity
            side_sizes = [side_size+1 for side_size in side_sizes]
        else:
            side_mem = torch.cat([side_mem, torch.zeros(batch_size, 2, side_dim).to(side_mem.device)], dim=1) #b * ns * s
            for i, side_size in enumerate(side_sizes):
                side_mem[i, side_size, :] += self._pad_entity
                side_mem[i, side_size+1, :] += self._eos_entity
            side_sizes = [side_size+2 for side_size in side_sizes]

        attn_feat, hop_feat, lstm_states, init_i = self._prepare(attn_mem)
        if not self._hard_attention:
            side_feat = self._prepare_side(side_mem) #b * ns * side_h
        else:
            side_feat = side_mem

        lstm_in = torch.cat([init_i, lstm_in], dim=1).transpose(0, 1)
        query, final_states = self._lstm(lstm_in, lstm_states)


        query = query.transpose(0, 1)
        for _ in range(self._n_hop):
            query = LSTMPointerNet_entity.attention(
                hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)
        if not self._hard_attention:
            side_e = LSTMPointerNet_entity.attention(side_feat, query, self.side_v, self.side_wq, side_sizes)
        else:
            #side_e, selection = LSTMPointerNet_entity.hard_attention(side_feat, query, self.side_wbi, self.side_wq, self._start, ground_entity)
            side_e, selection = LSTMPointerNet_entity.hard_attention_teacher_forcing(side_feat, query, self.side_wbi, self.side_wq,
                                                                     self._start, ground_entity)
        output = LSTMPointerNet_entity.attention_wiz_side(attn_feat, query, side_e, self._attn_v, self._attn_wq, self._attn_ws)
        if not self._hard_attention:
            return output  # unormalized extraction logit
        else:
            return output, selection

    def extract(self, attn_mem, mem_sizes, k, side_mem, side_sizes):
        """extract k sentences, decode only, batch_size==1"""
        batch_size = attn_mem.size(0)
        max_sent = attn_mem.size(1)
        if self.stop:
            attn_mem = torch.cat([attn_mem, self._stop.repeat(batch_size, 1).unsqueeze(1)], dim=1)
        side_mem = torch.cat([side_mem.unsqueeze(0), self._pad_entity.repeat(batch_size, 1).unsqueeze(1)], dim=1)

        attn_feat, hop_feat, lstm_states, lstm_in = self._prepare(attn_mem)
        if not self._hard_attention:
            side_feat = self._prepare_side(side_mem)
        else:
            side_feat = side_mem
        lstm_in = lstm_in.squeeze(1)
        if self._lstm_cell is None:
            self._lstm_cell = MultiLayerLSTMCells.convert(
                self._lstm).to(attn_mem.device)
        extracts = []
        if self._hard_attention:
            max_side = side_mem.size(1)
            side_dim = side_mem.size(2)
            context = self._start.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1)
        if self.stop:
            for sent_num in range(max_sent):
                h, c = self._lstm_cell(lstm_in, lstm_states)
                query = h[-1]
                for _ in range(self._n_hop):
                    query = LSTMPointerNet.attention(
                        hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)
                if not self._hard_attention:
                    side_e = LSTMPointerNet_entity.attention(side_feat, query, self.side_v, self.side_wq, side_sizes)
                else:
                    side_e, selected = LSTMPointerNet_entity.hard_attention_decoding(side_feat, query, self.side_wbi, self.side_wq, context)
                    context = context + selected
                    #print('context:', context)
                score = LSTMPointerNet_entity.attention_wiz_side(attn_feat, query, side_e, self._attn_v, self._attn_wq,
                                                                  self._attn_ws)
                score = score.squeeze()
                for e in extracts:
                    score[e] = -1e6
                ext = score.max(dim=0)[1].item()
                if ext == max_sent and sent_num != 0:
                    break
                elif sent_num == 0 and ext == max_sent:
                    ext = score.topk(2, dim=0)[1][1].item()
                extracts.append(ext)
                lstm_states = (h, c)
                lstm_in = attn_mem[:, ext, :]
        else:
            for _ in range(k):
                h, c = self._lstm_cell(lstm_in, lstm_states)
                query = h[-1]
                for _ in range(self._n_hop):
                    query = LSTMPointerNet.attention(
                        hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)
                if not self._hard_attention:
                    side_e = LSTMPointerNet_entity.attention(side_feat, query, self.side_v, self.side_wq, side_sizes)
                else:
                    side_e, selected = LSTMPointerNet_entity.hard_attention_decoding(side_feat, query, self.side_wbi, self.side_wq, context)
                    context = context + selected
                score = LSTMPointerNet_entity.attention_wiz_side(attn_feat, query, side_e, self._attn_v, self._attn_wq,
                                                                  self._attn_ws)
                score = score.squeeze()
                for e in extracts:
                    score[e] = -1e6
                ext = score.max(dim=0)[1].item()
                extracts.append(ext)
                lstm_states = (h, c)
                lstm_in = attn_mem[:, ext, :]
        return extracts

    def _prepare(self, attn_mem):
        attn_feat = torch.matmul(attn_mem, self._attn_wm.unsqueeze(0))
        hop_feat = torch.matmul(attn_mem, self._hop_wm.unsqueeze(0))
        bs = attn_mem.size(0)
        n_l, d = self._init_h.size()
        size = (n_l, bs, d)
        lstm_states = (self._init_h.unsqueeze(1).expand(*size).contiguous(),
                       self._init_c.unsqueeze(1).expand(*size).contiguous())
        d = self._init_i.size(0)
        init_i = self._init_i.unsqueeze(0).unsqueeze(1).expand(bs, 1, d)
        return attn_feat, hop_feat, lstm_states, init_i

    def _prepare_side(self, side_mem):
        side_feat = torch.matmul(side_mem, self.side_wm.unsqueeze(0))
        return side_feat

    @staticmethod
    def attention_score(attention, query, v, w):
        """ unnormalized attention score"""
        sum_ = attention.unsqueeze(1) + torch.matmul(
            query, w.unsqueeze(0)
        ).unsqueeze(2)  # [B, Nq, Ns, D]
        score = torch.matmul(
            F.tanh(sum_), v.unsqueeze(0).unsqueeze(1).unsqueeze(3)
        ).squeeze(3)  # [B, Nq, Ns]
        return score

    @staticmethod
    def attention_wiz_side(attention, query, side, v, w, s):
        """ unnormalized attention score"""
        sum_ = attention.unsqueeze(1) + torch.matmul(
            query, w.unsqueeze(0)
        ).unsqueeze(2) + torch.matmul(side, s.unsqueeze(0)).unsqueeze(2)
        # [B, Nq, Ns, D]
        score = torch.matmul(
            F.tanh(sum_), v.unsqueeze(0).unsqueeze(1).unsqueeze(3)
        ).squeeze(3)  # [B, Nq, Ns]
        return score

    @staticmethod
    def attention(attention, query, v, w, mem_sizes):
        """ attention context vector"""
        score = LSTMPointerNet_entity.attention_score(attention, query, v, w)
        if mem_sizes is None:
            norm_score = F.softmax(score, dim=-1)
        else:
            mask = len_mask(mem_sizes, score.device).unsqueeze(-2)
            norm_score = prob_normalize(score, mask)
        output = torch.matmul(norm_score, attention)
        return output


    @staticmethod
    def hard_attention(attention, query, w_bi, wq, _start, ground_truth=None):
        """ attention context vector"""
        # ground truth B * Nsent * Nside
        # attention B * Nside * Side
        # output = ground_truth.unsqueeze(3) * attention.unsqueeze(1) # B*Nsent*Nside*Side teacher forcing
        # output = output.sum(dim=2) # B*Nsent*Side
        side_dim = attention.size(2)
        n_side = attention.size(1)
        batch_size = attention.size(0)
        n_sent = query.size(1)
        all_output = torch.zeros(batch_size, n_sent, side_dim).to(attention.device)
        context = _start.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1)
        all_selection = torch.zeros(batch_size, n_sent, n_side).to(attention.device)
        for sent_id in range(n_sent):
            _query = query[:, sent_id, :].unsqueeze(1)
            bilinear = w_bi(context.unsqueeze(2).repeat(1,1,n_side,1), attention.unsqueeze(1).repeat(1, 1, 1, 1)).squeeze(3) # B*1*Nside
            selection = bilinear + torch.matmul(_query, wq.unsqueeze(0))
            all_selection[:, sent_id, :] = selection.squeeze(1)
            selected = F.sigmoid(selection) # B*1*Nside
            selected = selected.gt(0.5).float()
            output = selected.unsqueeze(3) * attention.unsqueeze(1)
            output = output.sum(dim=2)  # B*Nsent*Side
            all_output[:, sent_id, :] = output.squeeze(1)
            context = context + output

        return all_output, all_selection

    @staticmethod
    def hard_attention_teacher_forcing(attention, query, w_bi, wq, _start, ground_truth=None):
        """ attention context vector"""
        # ground truth B * Nsent * Nside
        # attention B * Nside * Side
        if ground_truth is not None:
            output = ground_truth.unsqueeze(3) * attention.unsqueeze(1) # B*Nsent*Nside*Side teacher forcing
            output = output.sum(dim=2) # B*Nsent*Side

            side_dim = attention.size(2)
            n_side = attention.size(1)
            batch_size = attention.size(0)
            n_sent = query.size(1)
            #all_zeros = torch.zeros(batch_size, 1, n_side).to(ground_truth.device)
            #all_zeros[:, :, -1] = 1.
            #pre_mask = torch.cat(
            #    (all_zeros, ground_truth), dim=1
            #)[:, :-1, :]
            pre_mask = ground_truth[:, :-1, :]
            #pre_mask = pre_mask.cumsum(dim=1) # B*Nsent*Nside
            context = pre_mask.unsqueeze(3) * attention.unsqueeze(1) # B*Nsent*Nside*Side
            context = context.sum(dim=2) # B*Nsent*Side
            context = torch.cat([
                _start.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1), context
            ], dim=1)
            # print('pre_context:', context)
            context = context.cumsum(dim=1)
            # print('post_context:', context)
            # print('size:')
            # print(context.size(), attention.size())
            # print(n_side, n_sent)
            bilinear = w_bi(context.unsqueeze(2).repeat(1,1,n_side,1), attention.unsqueeze(1).repeat(1, n_sent, 1, 1)).squeeze(3) # B*Nsent*Nside
            # print('bilinear:', bilinear.size())
            # print('attention:', attention.size())
            # print('query:', query.size())
            #print(torch.matmul(query, wq.unsqueeze(0)).size())
            selection = bilinear + torch.matmul(query, wq.unsqueeze(0))
            # print('out:', output.size())
            # print('selection:', selection.size())
            del pre_mask, context
            return output, selection
        else: # decoding time
            output = []
            raise Exception('decoding not implemented yet')
            return output, selection

    @staticmethod
    def hard_attention_decoding(attention, query, w_bi, wq, context):
        n_side = attention.size(1)
        batch_size = attention.size(0)
        n_sent = query.size(1) #should equal 1
        #context = context.unsqueeze(3) * attention.unsqueeze(1)
        #context = context.sum(dim=2)
        #print(context)
        bilinear = w_bi(context.unsqueeze(2).repeat(1, 1, n_side, 1),
                        attention.unsqueeze(1).repeat(1, n_sent, 1, 1)).squeeze(3)  # B*Nsent*Nside
        selection = bilinear + torch.matmul(query, wq.unsqueeze(0))
        selected = F.sigmoid(selection)
        print('selected:', selected)
        selected = selected.gt(0.5).float()
        #print(selected)
        # if selected.sum() < 1.:
        #     selected = selection.eq(selection.max()).float()
        #     #print('new selected:', selected)
        # if selected.sum() > 1.:
        #     selected[:, :, -1] = 0 # if the model has selected one, then force pad entity to zero
        output = selected.unsqueeze(3) * attention.unsqueeze(1)  # B*Nsent*Nside*Side
        output = output.sum(dim=2)  # B*Nsent*Side
        new_selected = selected.unsqueeze(3) * attention.unsqueeze(1)
        new_selected = new_selected.sum(dim=2)
        return output, new_selected





class PtrExtractSumm(nn.Module):
    """ rnn-ext"""
    def __init__(self, emb_dim, vocab_size, conv_hidden,
                 lstm_hidden, lstm_layer, bidirectional,
                 n_hop=1, dropout=0.0, stop=False):
        super().__init__()

        embedding = None

        self._sent_enc = ConvSentEncoder(
            vocab_size, emb_dim, conv_hidden, dropout, False, petrainable=False, embedding=embedding)
        self._art_enc = LSTMEncoder(
            3*conv_hidden, lstm_hidden, lstm_layer,
            dropout=dropout, bidirectional=bidirectional
        )
        enc_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self._extractor = LSTMPointerNet(
            enc_out_dim, lstm_hidden, lstm_layer,
            dropout, n_hop, stop
        )

    def forward(self, article_sents, sent_nums, target):
        enc_out = self._encode(article_sents, sent_nums)
        bs, nt = target.size()
        d = enc_out.size(2)
        ptr_in = torch.gather(
            enc_out, dim=1, index=target.unsqueeze(2).expand(bs, nt, d)
        )
        output = self._extractor(enc_out, sent_nums, ptr_in)
        return output

    def extract(self, article_sents, sent_nums=None, k=4, force_ext=True):
        enc_out = self._encode(article_sents, sent_nums)
        output = self._extractor.extract(enc_out, sent_nums, k)
        return output

    def sample(self, article_sents, sent_nums=None, k=4):
        enc_out = self._encode(article_sents, sent_nums)
        output, log_scores = self._extractor.sample(enc_out, sent_nums)
        return output, log_scores

    def _encode(self, article_sents, sent_nums):
        if sent_nums is None:  # test-time excode only
            enc_sent = self._sent_enc(article_sents[0]).unsqueeze(0)
        else:
            max_n = max(sent_nums)
            enc_sents = [self._sent_enc(art_sent)
                         for art_sent in article_sents]
            def zero(n, device):
                z = torch.zeros(n, self._art_enc.input_size).to(device)
                return z
            enc_sent = torch.stack(
                [torch.cat([s, zero(max_n-n, s.device)], dim=0)
                   if n != max_n
                 else s
                 for s, n in zip(enc_sents, sent_nums)],
                dim=0
            )
        lstm_out = self._art_enc(enc_sent, sent_nums)
        return lstm_out

    def set_embedding(self, embedding):
        self._sent_enc.set_embedding(embedding)

class PtrExtractSummEntity(nn.Module):
    """ rnn-ext"""
    def __init__(self, emb_dim, vocab_size, conv_hidden,
                 lstm_hidden, lstm_layer, bidirectional,
                 n_hop=1, dropout=0.0):
        super().__init__()
        enc_out_dim = lstm_hidden * (2 if bidirectional else 1)

        self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self._sent_enc = ConvSentEncoder(
            vocab_size, emb_dim, conv_hidden, dropout, False, embedding=self._embedding)
        self._entity_enc = ConvEntityEncoder(
            vocab_size, emb_dim, conv_hidden, dropout, False, embedding=self._embedding,
            context=False, context_hidden=emb_dim, context_input_dim=enc_out_dim
        )


        self._art_enc = LSTMEncoder(
            3*conv_hidden, lstm_hidden, lstm_layer,
            dropout=dropout, bidirectional=bidirectional
        )

        self._extractor = LSTMPointerNet_entity(
                enc_out_dim, lstm_hidden, lstm_layer,
                dropout, n_hop, 3*conv_hidden, True, False
            )

    def forward(self, article_sents, sent_nums, target, clusters, cluster_nums):
        enc_out = self._encode(article_sents, sent_nums)

        entity_out = self._encode_entity(clusters, cluster_nums, enc_out)

        bs, nt = target.size()
        d = enc_out.size(2)
        ptr_in = torch.gather(
            enc_out, dim=1, index=target.unsqueeze(2).expand(bs, nt, d)
        )

        output = self._extractor(enc_out, sent_nums, ptr_in, entity_out, cluster_nums)
        return output


    def extract(self, article_sents, clusters, sent_nums=None, cluster_nums=None, k=4, force_ext=True):
        enc_out = self._encode(article_sents, sent_nums)
        if not self._context:
            entity_out = self._encode_entity(clusters, cluster_nums)
        else:
            entity_out = self._encode_entity(clusters, cluster_nums, enc_out)
        # print('entity_out:', entity_out)

        output = self._extractor.extract(enc_out, sent_nums, k, entity_out, cluster_nums)
        return output

    def _encode(self, article_sents, sent_nums):
        if sent_nums is None:  # test-time excode only
            enc_sent = self._sent_enc(article_sents[0]).unsqueeze(0)
        else:
            max_n = max(sent_nums)
            enc_sents = [self._sent_enc(art_sent)
                         for art_sent in article_sents]
            def zero(n, device):
                z = torch.zeros(n, self._art_enc.input_size).to(device)
                return z
            enc_sent = torch.stack(
                [torch.cat([s, zero(max_n-n, s.device)], dim=0)
                   if n != max_n
                 else s
                 for s, n in zip(enc_sents, sent_nums)],
                dim=0
            )
        lstm_out = self._art_enc(enc_sent, sent_nums)
        return lstm_out

    def _encode_entity(self, clusters, cluster_nums, context=None):
        if cluster_nums is None: # test-time excode only
            if context is None:
                enc_entity = self._entity_enc(clusters[0], clusters[1], clusters[2], context)
            else:
                enc_entity = self._entity_enc(clusters[0], clusters[1], clusters[2], context[0, :, :])
        else:
            if context is None:
                clusters = clusters[:3]
                enc_entities = [self._entity_enc(cluster_words, cluster_wpos, cluster_spos) for cluster_words, cluster_wpos, cluster_spos in list(zip(*clusters))]
            else:
                clusters = clusters[:3]
                enc_entities = [self._entity_enc(cluster_words, cluster_wpos, cluster_spos, context[id, :, :])
                                for id, (cluster_words, cluster_wpos, cluster_spos) in enumerate(list(zip(*clusters)))]
            max_n = max(cluster_nums)
            def zero(n, device):
                z = torch.zeros(n, self._art_enc.input_size).to(device)
                return z

            enc_entity = torch.stack(
                [torch.cat([s, zero(max_n - n, s.device)], dim=0)
                 if n != max_n
                 else s
                 for s, n in zip(enc_entities, cluster_nums)],
                dim=0
            )

        return enc_entity


    def set_embedding(self, embedding):
        assert self._embedding.weight.size() == embedding.size()
        self._embedding.weight.data.copy_(embedding)