"""Self Attention Module


"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, in_channel, out_channel=None, attn_dropout=0.1):
        """
        :param in_channel: previous layer's output feature dimension
        :param out_channel: size of output vector, defaults to in_channel
        """
        super(SelfAttention, self).__init__()
        self.in_channel = in_channel

        if out_channel is not None:
            self.out_channel = out_channel
        else:
            self.out_channel = in_channel

        self.temperature = self.out_channel ** 0.5

        self.q_map = nn.Conv1d(in_channel, self.out_channel, 1, bias=False)
        self.k_map = nn.Conv1d(in_channel, self.out_channel, 1, bias=False)
        self.v_map = nn.Conv1d(in_channel, self.out_channel, 1, bias=False)

        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x):
        """
        :param x: the feature maps from previous layer,
                      shape: (batch_size, in_channel, num_points)
        :return: y: attentioned features maps,
                        shape： (batch_size, out_channel, num_points)
        """
        q = self.q_map(x)  # (batch_size, out_channel, num_points)
        k = self.k_map(x)  # (batch_size, out_channel, num_points)
        v = self.v_map(x)  # (batch_size, out_channel, num_points)

        attn = torch.matmul(q.transpose(1,2) / self.temperature, k)

        attn = self.dropout(F.softmax(attn, dim=-1))
        y = torch.matmul(attn, v.transpose(1,2)) # (batch_size, num_points, out_channel)

        return y.transpose(1, 2)


class QGPA(nn.Module):
    def __init__(self, attn_dropout=0.1):

        super(QGPA, self).__init__()
        self.in_channel = self.out_channel = 320

        self.temperature = self.out_channel ** 0.5
        self.layer_norm = nn.LayerNorm(self.in_channel)
        proj_dim = 512
        self.q_map = nn.Conv1d(2048, proj_dim, 1, bias=False)
        self.k_map = nn.Conv1d(2048, proj_dim, 1, bias=False)

        self.v_map = nn.Linear(self.in_channel, self.out_channel)
        self.fc = nn.Conv1d(self.in_channel, self.out_channel, 1, bias=False)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, query, support, prototype):

        batch, dim = query.shape[0], query.shape[1]
        way = support.shape[0] + 1
        residual = prototype
        q = self.q_map(query.transpose(1, 2))
        if len(support.shape) == 4:
            support = support.squeeze()
        support = torch.cat([support.mean(0).unsqueeze(0), support], dim=0)
        k = self.k_map(support.transpose(1, 2))
        v = self.v_map(prototype)
        q = q.view(q.shape[1], q.shape[2] * q.shape[0])
        k = k.view(k.shape[1], k.shape[2] * k.shape[0])

        attn = torch.matmul(q.transpose(0, 1) / self.temperature, k)
        attn = attn.reshape(batch, way, dim, dim)
        attn = F.softmax(attn, dim=-1)
        v = v.unsqueeze(2)
        output = torch.matmul(attn, v.transpose(-2, -1)).squeeze(-1).transpose(1, 2)
        output = self.dropout(self.fc(output)).transpose(1, 2)
        output = self.layer_norm(output + residual)

        return output
    
    

class TAMA_s3(nn.Module):
    def __init__(self, attn_dropout=0.1):
        super().__init__()
        self.temperature = 12 ** 0.5
        
        self.layer_norm = nn.LayerNorm(320)
        
        self.q_map = nn.Conv1d(24, 12, 1, bias=False)
        self.k_map = nn.Conv1d(2048, 12, 1, bias=False)
        self.v_map = nn.Conv1d(320, 320, 1, bias=False)
        
        self.fc = nn.Conv1d(320, 320, 1, bias=False)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, text_emb, support_feat):
        """
        text_emb: (1, in_dim, num_classes)
        support_feat: (n_support, in_dim, num_points)
        return: (n_support, out_dim, num_points)
        """
        # n_support, in_dim, num_points = support_feat.shape
        
        q = self.q_map(text_emb.transpose(1, 2))        # (1, 12, 320)
        k = self.k_map(support_feat.transpose(1, 2))    # (n_support, 12, 320)
        v = self.v_map(support_feat)                    # (n_support, 320, 2048)
        
        
        attn = torch.matmul(q.transpose(1,2), k) / self.temperature  
        attn = F.softmax(attn, dim=-1)      
        
        
        output = torch.matmul(attn, v)
        output = self.dropout(self.fc(output))                     # (n_support, out_dim, num_points)

        # 残差连接和LN
        output = output + support_feat 
        output = self.layer_norm(output.transpose(1, 2)).transpose(1, 2)
        
        return output
    
class TAMA_sc(nn.Module):
    def __init__(self, attn_dropout=0.1):
        super().__init__()
        self.temperature = 20 ** 0.5
        
        self.layer_norm = nn.LayerNorm(320)
        
        self.q_map = nn.Conv1d(40, 20, 1, bias=False)
        self.k_map = nn.Conv1d(2048, 20, 1, bias=False)
        self.v_map = nn.Conv1d(320, 320, 1, bias=False)
        
        self.fc = nn.Conv1d(320, 320, 1, bias=False)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, text_emb, support_feat):
        """
        text_emb: (1, in_dim, num_classes)
        support_feat: (n_support, in_dim, num_points)
        return: (n_support, out_dim, num_points)
        """
        # n_support, in_dim, num_points = support_feat.shape
        
        q = self.q_map(text_emb.transpose(1, 2))        # (1, 12, 320)
        k = self.k_map(support_feat.transpose(1, 2))    # (n_support, 12, 320)
        v = self.v_map(support_feat)                    # (n_support, 320, 2048)
        
        
        attn = torch.matmul(q.transpose(1,2), k) / self.temperature  
        attn = F.softmax(attn, dim=-1)      
        
        
        output = torch.matmul(attn, v)
        output = self.dropout(self.fc(output))                     # (n_support, out_dim, num_points)

        # 残差连接和LN
        output = output + support_feat 
        output = self.layer_norm(output.transpose(1, 2)).transpose(1, 2)
        
        return output
    
    
class self_attention(nn.Module):
    def __init__(self, attn_dropout=0.1):
        super().__init__()
        self.temperature = 2048 ** 0.5
        
        self.layer_norm = nn.LayerNorm(320)
        
        self.q_map = nn.Conv1d(320, 320, 1, bias=False)
        self.k_map = nn.Conv1d(320, 320, 1, bias=False)
        self.v_map = nn.Conv1d(320, 320, 1, bias=False)
        
        self.fc = nn.Linear(320, 320)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, feat):
        """
        feat: (n_support, in_dim, num_points)
        return: (n_support, out_dim, num_points)
        """
        # n_support, in_dim, num_points = feat.shape
        
        q = self.q_map(feat)        # (n_support, 160, num_points)
        k = self.k_map(feat)        # (1, 160, num_classes)
        v = self.v_map(feat)        # (1, 320, num_classes)
        
        
        attn = torch.matmul(q.transpose(1,2), k) / self.temperature  
        attn = F.softmax(attn, dim=-1)      
        
        
        output = torch.matmul(attn, v.transpose(1,2))    # (n_support, num_points, 320)
        output = self.dropout(self.fc(output))           # (n_support, num_points, 320)

        # 残差连接和LN
        output = output.transpose(1,2) + feat    # (n_support, 320, num_points)
        output = self.layer_norm(output.transpose(1, 2)).transpose(1, 2)
        
        return output

