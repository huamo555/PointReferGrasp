import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from model.pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation
from model.attention import MultiheadAttention, TransformerEncoder, TransformerDecoder,\
      TransformerEncoderLayer, TransformerDecoderLayer, PositionEmbeddingSine1D
from model.mm_group import GPBlock

# from pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation
# from attention import MultiheadAttention, TransformerEncoder, TransformerDecoder,\
#       TransformerEncoderLayer, TransformerDecoderLayer, PositionEmbeddingSine1D
# from mm_group import GPBlock

from torchvision.ops import roi_align
from transformers import AutoModel, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false" # this disables a huggingface tokenizer warning (printed every epoch)


class Point_Encoder(nn.Module):
    def __init__(self, emb_dim, normal_channel, additional_channel, N_p):
        super().__init__()
        self.N_p = N_p
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstractionMsg(self.N_p, [0.2,0.4], [16, 32], 256+256, [[128, 128, 256], [128, 196, 256]])

    def forward(self, xyz):

        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  #[B, 3, npoint_sa1] --- [B, 320, npoint_sa1]

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  #[B, 3, npoint_sa2] --- [B, 512, npoint_sa2]

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  #[B, 3, N_p]        --- [B, 512, N_p]

        return [[l0_xyz, l0_points], [l1_xyz, l1_points], [l2_xyz, l2_points], [l3_xyz, l3_points]]



class PointReferIterativeEinsum(nn.Module):
    def __init__(self, normal_channel=False, local_rank=None,
                N_p = 64, emb_dim = 512, proj_dim = 512, num_heads = 4, N_raw = 2048, num_affordance=18,
                freeze_text_encoder = False, text_encoder_type="roberta-base", n_groups=40,
                num_stages=2): 
        
        super().__init__()
        
        self.n_groups = n_groups
        self.emb_dim = emb_dim
        self.N_p = N_p
        self.N_raw = N_raw
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.local_rank = local_rank
        self.normal_channel = normal_channel
        self.num_stages = num_stages 
        if self.normal_channel:
            self.additional_channel = 3
        else:
            self.additional_channel = 0

        self.text_encoder = AutoModel.from_pretrained("/data3/gaoyuming/.cache/huggingface/hub/models--roberta-base/")
        self.tokenizer = AutoTokenizer.from_pretrained("/data3/gaoyuming/.cache/huggingface/hub/models--roberta-base/")
        self.freeze_text_encoder = freeze_text_encoder
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)
        self.text_resizer = nn.Sequential(nn.Linear(self.text_encoder.config.hidden_size, emb_dim, bias=True),
                                          nn.LayerNorm(emb_dim, eps=1e-12))

        self.point_encoder = Point_Encoder(self.emb_dim, self.normal_channel, self.additional_channel, self.N_p)

        self.pos1d = nn.Parameter(torch.zeros(1, self.n_groups, self.emb_dim))
        nn.init.trunc_normal_(self.pos1d, std = 0.2)
        

        self.decoders = nn.ModuleList()
        for _ in range(self.num_stages):

            decoder_stage = TransformerDecoder(
                TransformerDecoderLayer(self.emb_dim, nheads=num_heads, dropout=0),
                num_layers=1, 
                norm=nn.LayerNorm(self.emb_dim)
            )
            self.decoders.append(decoder_stage)
            
        self.gpb1 = GPBlock(embed_dims=512, num_group_token=self.n_groups, lan_dim=emb_dim)  
        self.gpb2 = GPBlock(embed_dims=512, num_group_token=self.n_groups, lan_dim=emb_dim)  
        self.gpb3 = GPBlock(embed_dims=512, num_group_token=self.n_groups, lan_dim=emb_dim)  

        self.fp3 = PointNetFeaturePropagation(in_channel=512+self.emb_dim, mlp=[768, 512])  
        self.fp2 = PointNetFeaturePropagation(in_channel=832, mlp=[768, 512]) 
        self.fp1 = PointNetFeaturePropagation(in_channel=518+self.additional_channel, mlp=[512, 512]) 
        
    def forward(self, text, xyz):
        B, C, N = xyz.size()
    
        t_feat, t_mask = self.forward_text(list(text), xyz.device)  # [B, L, C], [B, L]

        is_problematic = torch.all(t_mask, dim=1)
        if is_problematic.any():
            t_mask[is_problematic, -1] = False
            print("Warning: Patched an all-True t_mask to prevent NaN.")
        F_p_wise = self.point_encoder(xyz)     
    
        p_0, p_1, p_2, p_3 = F_p_wise
        
        p_3[1] = self.gpb1(t_feat, p_3[1].transpose(-2, -1)).transpose(-2, -1)
        up_sample_2 = self.fp3(p_2[0], p_3[0], p_2[1], p_3[1])
        
        up_sample_2 = self.gpb2(t_feat, up_sample_2.transpose(-2, -1)).transpose(-2, -1)
        up_sample_1 = self.fp2(p_1[0], p_2[0], p_1[1], up_sample_2)
        
        up_sample_1 = self.gpb3(t_feat, up_sample_1.transpose(-2, -1)).transpose(-2, -1)         
        point_features = self.fp1(p_0[0], p_1[0], torch.cat([p_0[0], p_0[1]],1), up_sample_1) # [B, C, N]
        
        all_mask_logits = []
        queries = t_feat 
        current_point_features = point_features
    
        for i in range(self.num_stages):
            if i > 0:
                prev_mask_logits = all_mask_logits[-1]
                gate = torch.sigmoid(prev_mask_logits).unsqueeze(1)
                current_point_features = point_features + point_features * gate
            else: 
                current_point_features = point_features
    
            decoder = self.decoders[i]
            
            refined_queries = decoder(
                queries,
                current_point_features.transpose(-2, -1),
                tgt_key_padding_mask=~t_mask,
                query_pos=self.pos1d.repeat(B, 1, 1)
            )
            
    
            refined_queries_masked = refined_queries * t_mask.unsqueeze(-1).float()
            _3daffordance_logits = torch.einsum('blc,bcn->bln', refined_queries_masked, current_point_features)
            
           
            valid_token_count = t_mask.float().sum(1).unsqueeze(-1)

            mask_logits = _3daffordance_logits.sum(1) / torch.clamp(valid_token_count, min=1.0)
    
            
            all_mask_logits.append(mask_logits)
            
            queries = refined_queries + refined_queries
            
        return all_mask_logits

    def forward_text(self, text_queries, device):
        tokenized_queries = self.tokenizer.batch_encode_plus(text_queries, padding='max_length', truncation=True,
                                                            max_length=self.n_groups,
                                                            return_tensors='pt')
        tokenized_queries = tokenized_queries.to(device)
        with torch.inference_mode(mode=self.freeze_text_encoder):
            encoded_text = self.text_encoder(**tokenized_queries).last_hidden_state
        return self.text_resizer(encoded_text), tokenized_queries.attention_mask.bool()




def get_PointRefer_Iterative_Einsum(normal_channel=False, local_rank=None,
    N_p = 64, emb_dim = 512, proj_dim = 512, num_heads = 4, N_raw = 2048, num_affordance=17, n_groups=40, num_stages=2):
    
    model = PointReferIterativeEinsum( normal_channel, local_rank,
    N_p, emb_dim, proj_dim, num_heads, N_raw, num_affordance, n_groups=n_groups, num_stages=num_stages)
    return model

if __name__ == "__main__":
    import yaml
    file = open('../config/config_seen.yaml', 'r', encoding='utf-8')
    string = file.read()
    dict = yaml.safe_load(string)

    # 重要提示：为了能正确运行这个文件进行测试，你需要将下面的get_PointRefer
    # 修改为 get_PointRefer_Iterative_Einsum
    model = get_PointRefer_Iterative_Einsum(
        N_p=dict['N_p'], emb_dim=dict['emb_dim'],
        proj_dim=dict['proj_dim'], num_heads=dict['num_heads'], N_raw=dict['N_raw'],
        num_affordance = dict['num_affordance'], n_groups=8).cuda()
    

    text = ('what are three sitting on what are three sitting on', 'what are three')
    xyz = torch.rand(2, 3, 2048).cuda()

    # forward会返回一个列表，包含了每个stage的输出
    all_logits = model(text, xyz)
    
    # 最终的预测结果通常是最后一个stage的输出
    final_logits = all_logits[-1]
    
    # 打印每个stage的输出形状
    for i, logits in enumerate(all_logits):
        print(f"Shape of logits from stage {i+1}: {logits.shape}")