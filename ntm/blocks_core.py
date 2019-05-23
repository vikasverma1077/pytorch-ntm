import torch
import torch.nn as nn

from attention import MultiHeadAttention
from BlockLSTM import BlockLSTM
from BlockGRU import BlockGRU
'''
Core blocks module.  Takes:
    input: (ts, mb, h)
    hx: (ts, mb, h)
    cx: (ts, mb, h)
    output: 
    output, hx, cx
'''

class BlocksCore(nn.Module):


    def __init__(self, nhid, num_blocks_in, num_blocks_out, topkval, step_att, do_gru): 
        super(BlocksCore, self).__init__()
        self.nhid = nhid
        self.num_blocks_in = num_blocks_in
        self.num_blocks_out = num_blocks_out
        self.block_size_in = 128#nhid // num_blocks_in
        self.block_size_out = nhid // num_blocks_out
        self.topkval = topkval
        self.step_att = step_att
        self.do_gru = do_gru

        print('bs in', self.block_size_in)
        print('bs out', self.block_size_out)

        self.mha = MultiHeadAttention(n_head=1, d_model_read=self.block_size_out, d_model_write=self.block_size_out, d_model_out=self.block_size_out, d_k=64, d_v=64, num_blocks_read=self.num_blocks_out, num_blocks_write=self.num_blocks_out, topk=self.num_blocks_out, grad_sparse=False)

        self.att_out = self.block_size_out

        self.inp_att = MultiHeadAttention(n_head=1, d_model_read=self.block_size_out, d_model_write=self.block_size_in, d_model_out=self.att_out, d_k=64, d_v=64, num_blocks_read=num_blocks_out, num_blocks_write=num_blocks_in+1,residual=False, topk=self.num_blocks_in+1, grad_sparse=False, skip_write=False)

        if do_gru:
            self.block_lstm = BlockGRU(self.att_out*self.num_blocks_out, self.nhid, k=self.num_blocks_out)
        else:
            self.block_lstm = BlockLSTM(self.att_out*self.num_blocks_out, self.nhid, k=self.num_blocks_out)

    def blockify_params(self):
        self.block_lstm.blockify_params()

    def forward(self, inp, hx, cx, do_print=False):

        hxl = []
        cxl = []

        inp_use = inp #layer_input[idx_step]
                        
        #use attention here.  
        inp_use = inp_use.reshape((inp_use.shape[0], self.num_blocks_in, self.block_size_in))
        inp_use = torch.cat([inp_use, torch.zeros_like(inp_use[:,0:1,:])], dim=1)
        
        #print('inp use shape pre-att', inp_use.shape)
        #print('hx shape', hx.shape)
        
        inp_use, iatt, _ = self.inp_att(hx.reshape((hx.shape[0], self.num_blocks_out, self.block_size_out)), inp_use, inp_use)
        inp_use = inp_use.reshape((inp_use.shape[0], self.att_out*self.num_blocks_out))

        null_score = iatt.mean((0,1))[1]

        if do_print:
            print('inp attention on step', input.shape[0], '(total steps)', idx_step, iatt[0])
            print('iat shape', iatt.shape)
            print('iat summed', iatt.mean((0,1)))
            print('iat null_score', null_score)


        topk_mat = torch.topk(iatt[:,:,0], dim=1, k=self.topkval)[0][:,-1] #64 x 1
        topk_mat = topk_mat.reshape((inp_use.shape[0],1)).repeat(1,self.num_blocks_out) #64 x num_blocks
        mask = torch.gt(iatt[:,:,0], topk_mat - 0.01).float()
        
        if do_print:
            print('step', idx_step, 'out of', inp.shape[0])
            print('att at 0', iatt[0])
            print('mask at 0', mask[0])
                        
                        
        mask = mask.reshape((inp_use.shape[0],self.num_blocks_out,1)).repeat((1,1,self.block_size_out)).reshape((inp_use.shape[0], self.num_blocks_out*self.block_size_out))
                        
        mask = mask.detach()

        if self.do_gru:
            hx_new = self.block_lstm(inp_use, hx)
            cx_new = hx_new
        else:
            hx_new, cx_new = self.block_lstm(inp_use, hx, cx)

        hx_old = hx*1.0
        cx_old = cx*1.0

        if self.step_att:
            hx_new = hx_new.reshape((hx_new.shape[0], self.num_blocks_out, self.block_size_out))
            hx_new,attn_out,extra_loss_att = self.mha(hx_new,hx_new,hx_new)
            hx_new = hx_new.reshape((hx_new.shape[0], self.nhid))
            extra_loss = extra_loss_att

        hx = (mask)*hx_new + (1-mask)*hx_old
        cx = (mask)*cx_new + (1-mask)*cx_old

        return hx, cx, mask


if __name__ == "__main__":
    bc = BlocksCore(512, 1, 4, 4)

    inp = torch.randn(10, 512)
    hx = torch.randn(10,512)
    cx = torch.randn(10,512)

    hx, cx = bc(inp, hx, cx)

    print('hx cx shape', hx.shape, cx.shape)
