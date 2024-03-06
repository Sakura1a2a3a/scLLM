import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadAttention(nn.Module):
    def __init__(self, hidden_size, nhead):
        super(MultiheadAttention, self).__init__()
        self.hidden_size =hidden_size
        self.nhead = nhead
        self.head_dim = hidden_size//nhead

        assert self.head_dim *nhead == hidden_size,"Must be divisible!"

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key =nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.fc_out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, mask=None):
        batch_size= x.shape[0]
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # Split into multiple heads
        query = query.view(batch_size,-1, self.nhead, self.head_dim).transpose(1,2)
        key = key.view(batch_size, -1, self.nhead,self.head_dim).transpose(1,2)
        value = value.view(batch_size,-1, self.nhead, self.head_dim).transpose(1,2)

        energy = torch.matmul(query, key.transpose(-2, -1))/(self.head_dim** 0.5)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            mask = mask.expand(-1, -1, energy.size(2), -1)

            # identity matrix diagonal be True
            batch_size,nheads, seq_length, _ = energy.size()
            attn_mask = mask.repeat(1, nheads, 1, 1)
            eye_mask =torch.eye(seq_length, device=x.device).bool()
            eye_mask = eye_mask.unsqueeze(0).unsqueeze(0)
            eye_mask =eye_mask.expand(batch_size, nheads, -1, -1)
            attn_mask = attn_mask&(~eye_mask)

            energy = energy.masked_fill(attn_mask, float("-inf"))


        attention =torch.softmax(energy, dim=-1)
        out = torch.matmul(attention, value)

        out = out.transpose(1, 2).contiguous().view(batch_size,-1, self.hidden_size)
        out =self.fc_out(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, hidden_size, ff_size):
        super(FeedForward,self).__init__()
        self.linear1 = nn.Linear(hidden_size, ff_size)
        self.linear2 = nn.Linear(ff_size, hidden_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, nhead, ff_size):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(hidden_size, nhead)
        self.feed_forward = FeedForward(hidden_size, ff_size)
        self.norm1=nn.LayerNorm(hidden_size)
        self.norm2=nn.LayerNorm(hidden_size)
        self.dropout=nn.Dropout(0.1)

    def forward(self, x, mask = None):
        out=self.self_attn(x, mask)
        x = x + self.dropout(out)
        x = self.norm1(x)

        out = self.feed_forward(x)
        x =x + self.dropout(out)
        x= self.norm2(x)

        return x

class TransformerEncoder(nn.Module):
    def __init__(self,hidden_size,nhead, ff_size, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(hidden_size, nhead, ff_size) for _ in range(num_layers)])

    def forward(self, x, mask = None):
        for layer in self.layers:
            x = layer(x,mask)
        return x
