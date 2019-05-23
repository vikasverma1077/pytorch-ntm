
from blocks_core import BlocksCore

import torch
import torch.nn as nn


class BlockWrapper(nn.Module):
    
    def __init__(self, ntokens, nhid, dropout=0.0, num_blocks=4, update_topk=4):
        super(BlockWrapper, self).__init__()
        #self.myrnn = rnn_models.RNNModel("LSTM", ntokens, nhid, nhid,
        #                    nlayers=1, dropout=dropout, tie_weights=False,
        #                    use_cudnn_version=False, use_adaptive_softmax=False,
        #                    cutoffs=[10000], discrete_input=False, num_blocks=num_blocks, topk=update_topk, use_gru=True).cuda()
        
        self.myrnn = BlocksCore(nhid, num_blocks_in=1, num_blocks_out=num_blocks, topkval=update_topk, step_att=True, do_gru=True)
        
        #self.myrnn = nn.GRU(ntokens, nhid)
        self.nhid = nhid

        print('using blocks wrapper!')

    def forward(self, inp, h):
        self.myrnn.blockify_params()
        hlst = []
        h = h[0]
        for step in range(inp.shape[0]):
            cx = torch.zeros_like(h)
            h,cx,mask = self.myrnn(inp[step],h,cx)
            hlst.append(h)
        output = torch.stack(hlst)

        return output, h.unsqueeze(0)

if __name__ == "__main__":
    nhid = 128
    ntokens = 128

    blocks = BlockWrapper(ntokens, nhid).cuda()
    gru = torch.nn.GRU(ntokens, nhid).cuda()

    x = torch.randn(5, 1, ntokens).cuda()

    h0 = torch.randn(1, 1, nhid).cuda()
    h0_blocks = torch.randn(1, 1, nhid).cuda()

    og, hg = gru(x, h0)
    print('gru of x: o,h', og.shape, hg.shape)

    ob, hb = blocks(x, h0_blocks)
    print('block res: o,h', ob.shape, hb.shape)

