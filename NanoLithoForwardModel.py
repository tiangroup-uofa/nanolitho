import os
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
import math 

class CLR():
    def __init__(self, train_dataloader, base_lr=1e-5, max_lr=100):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.bn = len(train_dataloader) - 1
        ratio = self.max_lr/self.base_lr
        self.mult = ratio ** (1/self.bn)
        self.best_loss = 1e9
        self.iteration = 0
        self.lrs = []
        self.losses = []
        
    def calc_lr(self, loss):
        self.iteration +=1
        if math.isnan(loss) or loss > 4 * self.best_loss:
            return -1
        if loss < self.best_loss and self.iteration > 1:
            self.best_loss = loss
        mult = self.mult ** self.iteration
        lr = self.base_lr * mult
        self.lrs.append(lr)
        self.losses.append(loss)
        return lr
        
    def plot(self, start=10, end=-5):
        lrs_cpu = [lr.cpu() if torch.is_tensor(lr) else lr for lr in self.lrs]
        losses_cpu = [loss.cpu().detach().numpy() if torch.is_tensor(loss) else loss 
                  for loss in self.losses]
        plt.xlabel("Learning Rate")
        plt.ylabel("Losses")
        plt.plot(lrs_cpu[start:end], losses_cpu[start:end])
        plt.xscale('log')
        plt.savefig('/teamspace/studios/this_studio/lr_finder.png')
        plt.show()
    
class OneCycle():
    def __init__(self, nb, max_lr, momentum_vals=(0.95, 0.85), prcnt= 10 , div=10):
        self.nb = nb
        self.div = div
        self.step_len =  int(self.nb * (1- prcnt/100)/2)
        self.high_lr = max_lr
        self.low_mom = momentum_vals[1]
        self.high_mom = momentum_vals[0]
        self.prcnt = prcnt
        self.iteration = 0 
        self.lrs = []
        self.moms = []
        
    def calc(self):
        self.iteration += 1
        lr = self.calc_lr()
        mom = self.calc_mom()
        return (lr, mom)
        
    def calc_lr(self):
        if self.iteration==self.nb:
            self.iteration = 0
            self.lrs.append(self.high_lr/self.div)
            return self.high_lr/self.div
        if self.iteration > 2 * self.step_len:
            ratio = (self.iteration - 2 * self.step_len) / (self.nb - 2 * self.step_len)
            lr = self.high_lr * ( 1 - ratio * (1-(1/self.div))/self.div ) 
        elif self.iteration > self.step_len:
            ratio = 1- (self.iteration -self.step_len)/self.step_len
            lr = self.high_lr * (1 + ratio * (self.div - 1)) / self.div
        else:
            ratio = self.iteration/self.step_len
            lr = self.high_lr * (1 + ratio * (self.div - 1)) / self.div
        self.lrs.append(lr)
        return lr
    
    def calc_mom(self):
        if self.iteration==self.nb:
            self.iteration = 0
            self.moms.append(self.high_mom)
            return self.high_mom
        if self.iteration > 2 * self.step_len:
            mom = self.high_mom
        elif self.iteration > self.step_len:
            ratio = (self.iteration -self.step_len)/self.step_len
            mom = self.low_mom + ratio * (self.high_mom - self.low_mom)
        else:
            ratio = self.iteration/self.step_len
            mom = self.high_mom - ratio * (self.high_mom - self.low_mom)
        self.moms.append(mom)
        return mom

def find_lr(clr, train_dataloader, opt, model, device):
    criterion = CustomLoss()
    running_loss = 0.0
    avg_beta = 0.98
    model.train()
    for i, (input, target) in tqdm(enumerate(train_dataloader)):
        target = target.to(device)
        input = {k: v.to(device) for k, v in input.items()}
        output = model(input)
        loss = criterion(output, target)
        running_loss = avg_beta * running_loss + (1-avg_beta) * loss
        smoothed_loss = running_loss / (1 - avg_beta**(i+1))
        lr = clr.calc_lr(smoothed_loss)
        if lr == -1:
            break
        for pg in opt.param_groups:
            pg['lr'] = lr   
        opt.zero_grad()
        loss.backward()
        opt.step()
    clr.plot()


def cross_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scale = 1.0 / (d_k ** 0.5)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output, attention_weights

class GlobalTrajEncoder(nn.Module): 
    def __init__(self, input_dim=2, num_dims=128, output_dims=64):
        super().__init__()
        self.num_dims = num_dims
        self.input_proj = nn.Linear(input_dim, num_dims)
        self.mlp_block = nn.Sequential(
            nn.LayerNorm(num_dims), 
            nn.Linear(num_dims, num_dims * 4),
            nn.SiLU(),
            nn.Linear(num_dims * 4, num_dims)
        )
        self.output = nn.Sequential(
            nn.LayerNorm(num_dims),
            nn.Linear(num_dims, output_dims)
        )
    
    def forward(self, input, num_layers=5): 
        x = self.input_proj(input)  
        for i in range(num_layers):
            residual = x
            x = self.mlp_block(x)
            x = x + residual
        output = self.output(x)
        return output


class CustomAttention(nn.Module):
    def __init__(self, embed_size, num_heads=8, attention_type="self"):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.attention_type = attention_type
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
        self.ln = nn.LayerNorm(embed_size)

    def forward(self, target, source=None, mask=None):
        residual = target
        N, seq_len_q, embed_size = target.shape
        if self.attention_type == "self" or source is None:
            source = target
            seq_len_kv = seq_len_q
        else:
            seq_len_kv = source.shape[1] 
        Q = self.query(target)
        K = self.key(source)
        V = self.value(source)
        Q = Q.view(N, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(N, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(N, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
        out = out.transpose(1, 2).contiguous().view(N, seq_len_q, embed_size)
        out = self.fc_out(out)
        return self.ln(out + residual)


class AttentionGate(nn.Module): 
    def __init__(self, in_channels_encoder, in_channels_decoder): 
        super().__init__() 
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels_decoder, in_channels_decoder, 1), 
            nn.BatchNorm2d(num_features=in_channels_decoder) 
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels_encoder, in_channels_encoder, 1), 
            nn.BatchNorm2d(num_features=in_channels_encoder) 
        )
        total_channels = in_channels_encoder + in_channels_decoder 
        self.combined = nn.Sequential(
           nn.ReLU(), 
           nn.Conv2d(total_channels, total_channels, 1),
           nn.BatchNorm2d(num_features=total_channels),
           nn.Conv2d(total_channels, in_channels_encoder, 1)
        )
       
    def forward(self, in_encoder, in_decoder, gate): 
        encoder_out = self.encoder(in_encoder) 
        decoder_out = self.decoder(in_decoder)
        combined = torch.cat([encoder_out, decoder_out], dim=1) 
        combined_out = self.combined(combined) 
        combined_out = torch.sigmoid(combined_out) 
        return in_encoder * combined_out * gate


class PhysicsSetAttention(nn.Module):
    def __init__(self, spatial_dim, point_dim=512, num_heads=8):
        super().__init__()
        assert spatial_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = spatial_dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.query_proj = nn.Linear(spatial_dim, spatial_dim)
        self.key_proj   = nn.Linear(point_dim,   spatial_dim)
        self.value_proj = nn.Linear(point_dim,   spatial_dim)
        self.out_proj   = nn.Linear(spatial_dim, spatial_dim)
        self.norm       = nn.GroupNorm(num_groups=8, num_channels=spatial_dim)

    def forward(self, spatial_features, point_embeddings):
        B, C, H, W = spatial_features.shape
        N = point_embeddings.shape[1]
        x = spatial_features.view(B, C, -1).permute(0, 2, 1)
        Q = self.query_proj(x)
        K = self.key_proj(point_embeddings)
        V = self.value_proj(point_embeddings)
        Q = Q.view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N,   self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N,   self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        scores = scores / N
        out = torch.matmul(scores, V)
        out = out.transpose(1, 2).contiguous().view(B, H*W, C)
        out = self.out_proj(out)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return self.norm(out + spatial_features)





class MultiScaleStencil(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=8, bias=False),
            nn.BatchNorm2d(32), nn.SiLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64), nn.SiLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(128), nn.SiLU()
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(256), nn.SiLU()
        )

    def forward(self, stencil):
        st1 = self.down1(stencil)
        st2 = self.down2(st1)
        st3 = self.down3(st2)
        st4 = self.down4(st3)
        return st1, st2, st3, st4


class PeriodicCoordGen(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell_sizes = [16, 32, 64, 128]
        self.division_predictor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, len(self.cell_sizes)),
            nn.Softmax(dim=-1)
        )

    def forward(self, stencil, target_h, target_w):
        B = stencil.shape[0]
        scale_weights = self.division_predictor(stencil)
        coord_maps = []
        for cell_size in self.cell_sizes:
            ys = torch.linspace(0, 1, cell_size, device=stencil.device)
            xs = torch.linspace(0, 1, cell_size, device=stencil.device)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
            grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
            tiled = F.interpolate(grid, size=(target_h, target_w), mode='nearest')
            coord_maps.append(tiled)
        coord_maps = torch.stack(coord_maps, dim=1)
        weights    = scale_weights.view(B, 4, 1, 1, 1)
        return (coord_maps * weights).sum(dim=1)

    
class SimulationDataset(Dataset): 
    def __init__(self, file_path, train=True, split=0.8, num_samples=None): 
        folder_path = Path(file_path)
        dataset_paths = list(folder_path.glob('*.npz'))
        random.seed(42)
        random.shuffle(dataset_paths)
        split_idx = int(len(dataset_paths) * split)
        self.paths = dataset_paths[:split_idx] if train else dataset_paths[split_idx:]
        if num_samples is not None: 
            self.paths = self.paths[:num_samples]
    
    def __len__(self): 
        return len(self.paths)
    
    def __getitem__(self, idx): 
        file_path = self.paths[idx]
        with np.load(file_path, allow_pickle=True) as data: 
            input_stencil = np.nan_to_num(data['input_stencil'])
            target_dep = np.nan_to_num(data['target_deposition'])
            traj = data['trajectory']
            traj[:, 0] = traj[:, 0] / (2*np.pi)
            traj[:, 1] = traj[:, 1] / 0.1
            traj_tensor = torch.from_numpy(traj).float()
            point_count = len(traj_tensor)
            stencil_tensor = torch.from_numpy(input_stencil).float()
            target_tensor = torch.from_numpy(target_dep).float() / point_count
            return {
                'stencil': stencil_tensor.unsqueeze(0),
                'traj': traj_tensor,  
            }, target_tensor.unsqueeze(0)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.traj_gte         = GlobalTrajEncoder(input_dim=2,   num_dims=64,  output_dims=64)
        self.combined_gte     = GlobalTrajEncoder(input_dim=128, num_dims=256, output_dims=512)

        self.multi_scale_tokens = MultiScaleStencil()
        self.coord_gen          = PeriodicCoordGen()

        self.stencil_token = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 64), nn.SiLU()
        )

        

        

        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.SiLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.SiLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.SiLU(),
        )

        self.coord_proj_bttl = nn.Conv2d(256 + 2, 256, kernel_size=1)
        self.coord_proj1     = nn.Conv2d(128 + 2, 128, kernel_size=1)
        self.coord_proj2     = nn.Conv2d(64  + 2, 64,  kernel_size=1)
        self.coord_proj3     = nn.Conv2d(64  + 2, 64,  kernel_size=1)

        self.mst_attn1 = CustomAttention(embed_size=128, attention_type='cross')
        self.mst_attn2 = CustomAttention(embed_size=64,  attention_type='cross')

        self.pnt_attn_bttl = PhysicsSetAttention(spatial_dim=256, point_dim=512)
        self.pnt_attn1     = PhysicsSetAttention(spatial_dim=128, point_dim=512)
        self.pnt_attn2     = PhysicsSetAttention(spatial_dim=64,  point_dim=512)

        self.skip1_gate = AttentionGate(in_channels_encoder=64,  in_channels_decoder=128)
        self.skip2_gate = AttentionGate(in_channels_encoder=32,  in_channels_decoder=64)

        self.dec1_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.SiLU(),
        )
        self.dec1_conv = nn.Sequential(
            nn.GroupNorm(8, 128+64),
            nn.Conv2d(128 + 64, 128, kernel_size=3, padding=1), nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.SiLU(),
        )
        self.dec2_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.SiLU(),
        )
        self.dec2_conv = nn.Sequential(
            nn.GroupNorm(8, 64+32),
            nn.Conv2d(64 + 32, 64, kernel_size=3, padding=1), nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.SiLU(),
        )
        self.dec3_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.SiLU(),
        )
        self.dec3_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.SiLU(),
        )

        self.out_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.out_conv.weight, gain=0.01)
        nn.init.constant_(self.out_conv.bias, 0.0)



    def apply_attn(self, x, token, attn_layer):
        B, C, H, W = x.shape
        q  = x.view(B, C, -1).permute(0, 2, 1)
        kv = token.view(B, C, -1).permute(0, 2, 1)
        out = attn_layer(target=q, source=kv)
        return out.permute(0, 2, 1).view(B, C, H, W)

    def forward(self, x_dict):
        trajectory = x_dict['traj']
        stencil    = x_dict['stencil']
        batch, num_pts, _ = trajectory.shape

        #stencil encoder 
        f1         = self.enc1(stencil)
        f2         = self.enc2(f1)
        bottleneck = self.enc3(f2)

        stencil_token = self.stencil_token(bottleneck)

        #normalize:
        f2 = F.group_norm(f2, num_groups = 8)
        f1 = F.group_norm(f1, num_groups = 8)

        #trajectory encoder 
        points_flat      = trajectory.view(batch * num_pts, 2)
        traj_embed       = self.traj_gte(points_flat, num_layers=2)
        stencil_expanded = stencil_token.unsqueeze(1).expand(
                               -1, num_pts, -1
                           ).reshape(batch * num_pts, 64)
        combined_features = self.combined_gte(
                                torch.cat([traj_embed, stencil_expanded], dim=-1),
                                num_layers=3
                            ).view(batch, num_pts, 512)

      

        #analytical gating factor
        gate = torch.sigmoid(
            torch.log(torch.tensor(num_pts, dtype=torch.float32, device=stencil.device))
        ).view(1, 1, 1, 1).expand(batch, 1, 1, 1)

        #multi scale stencil tokens
        ms_st1, ms_st2, ms_st3, ms_st4 = self.multi_scale_tokens(stencil)

        coord_256 = self.coord_gen(stencil, 256, 256)
        coord_32  = F.interpolate(coord_256, size=(32,32),   mode='nearest')
        coord_64  = F.interpolate(coord_256, size=(64,64),   mode='nearest')
        coord_128 = F.interpolate(coord_256, size=(128,128), mode='nearest')

        
        x = bottleneck

        
        x = torch.cat([x, coord_32], dim=1)
        x = self.coord_proj_bttl(x)
        x = self.pnt_attn_bttl(x, combined_features)

        
        #decoder
        x = self.dec1_up(x)
        x = torch.cat([x, coord_64], dim=1)
        x = self.coord_proj1(x)
        x = self.apply_attn(x, ms_st3, self.mst_attn1)
        x = self.pnt_attn1(x, combined_features)

        
        f2_refined = self.skip1_gate(f2, x, gate)

        x = torch.cat([x, f2_refined], dim=1)
        x = self.dec1_conv(x)

        
        #second decoder layer
        x = self.dec2_up(x)
        x = torch.cat([x, coord_128], dim=1)
        x = self.coord_proj2(x)
        x = self.apply_attn(x, ms_st2, self.mst_attn2)
        x = self.pnt_attn2(x, combined_features)

        
        f1_refined = self.skip2_gate(f1, x, gate)


        x = torch.cat([x, f1_refined], dim=1)
        x = self.dec2_conv(x)

      

        #third decoder layer
        x = self.dec3_up(x)
        x = torch.cat([x, coord_256], dim=1)
        x = self.coord_proj3(x)
        x = self.dec3_conv(x)

        out = self.out_conv(x)


        return out


class MSSSIMLoss(nn.Module): 
    def __init__(self, data_range=2.0): 
        super(MSSSIMLoss, self).__init__()
        self.msssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=data_range).to(device)
        
    def forward(self, pred, target): 
        return 1.0 - self.msssim(pred, target)
    
class AdaptiveMSELoss(nn.Module):
    def __init__(self, n_pts): 
        super(AdaptiveMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.n_pts = n_pts 
    def forward(self, pred, target): 
        return self.mse(pred * self.n_pts, target * self.n_pts)

class CustomLoss(nn.Module): 
    def __init__(self, data_range=1.0): 
        super(CustomLoss, self).__init__() 
        self.msssim = MSSSIMLoss() 
        self.mae = nn.L1Loss() 
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target): 
        return (0.80 * self.mse(pred, target) + 0.20 * self.msssim(pred, target)) 

class DiceLoss(nn.Module): 
    def __init__(self): 
        super(DiceLoss, self).__init__() 
    
    def forward(self, pred, target, smooth=1): 
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

class HybridLoss(nn.Module): 
    def __init__(self): 
        super(HybridLoss, self).__init__()
        self.msssim = MSSSIMLoss() 
        self.mae = nn.L1Loss() 
        self.mse = nn.MSELoss() 
        self.dice = DiceLoss()

    def forward(self, pred, target, percentile=0.2):
        threshold_val = torch.quantile(pred, percentile)  
        pred_binary = torch.ones_like(pred) 
        target_binary = torch.ones_like(target)  
        pred_binary[pred > threshold_val] = 1 
        pred_binary[pred <= threshold_val] = 0
        target_binary[pred > threshold_val] = 1 
        target_binary[pred <= threshold_val] = 0 
        dice_loss    = self.dice(pred_binary, target_binary)
        ms_ssim_loss = self.msssim(pred, target) 
        mae_loss     = self.mae(pred, target) 
        return (0.6 * ms_ssim_loss + 0.2 * mae_loss + 0.2 * dice_loss) 


# --- Training & Evaluation ---
def train(optimizer, train_loader, val_dataset, model, epochs, device, scheduler=None):
    criterion = CustomLoss()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch: {epoch}")
        for batch_idx, (batch_dict, target) in enumerate(pbar):

            # enable diagnostic on first batch of each epoch only
            if batch_idx == 0:
                model.diagnostic = True

            inputs = {k: v.to(device) for k, v in batch_dict.items()}
            target = target.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            model.diagnostic = False  # off for all remaining batches

            if scheduler is not None: 
                scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        print(f"Epoch {epoch} Avg Loss: {train_loss / len(train_loader):.6f}")
        evaluate_monitor(model, val_dataset, device, epoch, [0, 1, 2])

        if train_loss / len(train_loader) < 0.1: 
            os.makedirs('/teamspace/studios/this_studio/saved_modelsvDS3', exist_ok=True)
            save_path = f'/teamspace/studios/this_studio/saved_modelsvDS3/mse_{epoch}_360point_pt4.pth'
            torch.save(model.state_dict(), save_path)

            
def evaluate_monitor(model, val_dataset, device, epoch, indices, base_path='/teamspace/studios/this_studio/epoch_monitor3'):
    os.makedirs(base_path, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for idx in indices:
            batch_dict, target = val_dataset[idx]
            inputs = {k: v.unsqueeze(0).to(device) for k, v in batch_dict.items()}
            outputs = model(inputs)
            pred = outputs.squeeze().cpu().numpy()
            target_np = target.squeeze().cpu().numpy()
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].imshow(target_np, cmap='viridis'); axes[0].set_title("GT Delta")
            axes[1].imshow(pred, cmap='viridis'); axes[1].set_title(f"Pred Delta Ep {epoch}")
            plt.savefig(os.path.join(base_path, f"ep_{epoch}_idx_{idx}.png"))
            plt.close()

def EvaluateValidation(model, val_dataset, device, base_path='/teamspace/studios/this_studio/validation_eval_final2'): 
    os.makedirs(base_path, exist_ok=True)
    model.eval() 
    error_data = []
    
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure().to(device)
    
    print(f"Starting evaluation on {len(val_dataset)} samples...")
    with torch.no_grad():
        for i in range(len(val_dataset)): 
            batch_dict, target = val_dataset[i]
            inputs = {k: v.unsqueeze(0).to(device) for k, v in batch_dict.items()}
            target = target.unsqueeze(0).to(device) 
            stencil_np = batch_dict['stencil'].cpu().numpy()
            target_np = target.cpu().numpy().squeeze()
            pred = model(inputs)
            pred_np = pred.cpu().numpy().squeeze()
            
            mae = np.mean(np.abs(target_np - pred_np))
            mse = np.mean(np.abs(target_np - pred_np)**2)
            
            target_min = target.min()
            target_max = target.max()
            data_range = target_max - target_min
            
            if data_range > 0:
                target_normalized = (target - target_min) / data_range
                pred_normalized = (pred - target_min) / data_range
                ms_ssim_val = ms_ssim(pred_normalized, target_normalized).item()
            else:
                if torch.allclose(pred, target, rtol=1e-3, atol=1e-3):
                    ms_ssim_val = 1.0
                else:
                    ms_ssim_val = 0.0
            
            error_data.append({'idx': i, 'mae': mae, 'mse': mse, 'ms_ssim': ms_ssim_val})
            
            if i % 50 == 0:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Determine colorbar limits based on target range for fair comparison
                vmin, vmax = target_np.min(), target_np.max()
                
                im0 = axes[0].imshow(stencil_np.squeeze(), cmap='viridis', vmin=0, vmax=1)
                axes[0].set_title(f'Stencil (Idx {i})')
                
                im1 = axes[1].imshow(target_np.squeeze(), cmap='viridis', vmin=vmin, vmax=vmax)
                axes[1].set_title('Ground Truth')
                
                im2 = axes[2].imshow(pred_np.squeeze(), cmap='viridis', vmin=vmin, vmax=vmax)
                axes[2].set_title(f'Pred (MAE: {mae:.4e}, MS-SSIM: {ms_ssim_val:.4f})')
                
                for ax in axes: 
                    ax.set_xticks([])
                    ax.set_yticks([])
                
                fig.colorbar(im2, ax=axes.ravel().tolist(), label='Intensity')
                plt.savefig(os.path.join(base_path, f'sample_{i}.png'), bbox_inches='tight')
                plt.close(fig)
            
            if i % 10 == 0: 
                print(f"Processed {i}/{len(val_dataset)} samples...")

    mae_list = np.array([d['mae'] for d in error_data])
    mse_list = np.array([d['mse'] for d in error_data])
    ms_ssim_list = np.array([d['ms_ssim'] for d in error_data])
    
    print('\nValidation Set Error metrics:')
    print(f'Average MSE: {np.mean(mse_list):.6e}')
    print(f'Average MAE: {np.mean(mae_list):.6e}')
    print(f'Average MS-SSIM: {np.mean(ms_ssim_list):.6f} ± {np.std(ms_ssim_list):.6f}')
     
    print(f'\nMS-SSIM Range: [{np.min(ms_ssim_list):.6f}, {np.max(ms_ssim_list):.6f}]')
    
    return error_data


# --- Execution ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
file_path = '/teamspace/studios/this_studio/forward_model_360pointv2'

train_data = SimulationDataset(file_path=file_path)
val_data = SimulationDataset(file_path=file_path, train=False)
loader = DataLoader(train_data, batch_size=16, shuffle=True)

model = UNet().to(device)
path = '/teamspace/studios/this_studio/saved_modelsvDS3/mse_15_360point_pt4.pth'

def load_weights(model, path):
    saved_state = torch.load(path)
    model_state = model.state_dict()
    matched = {}
    skipped = []
    for k, v in saved_state.items():
        if k not in model_state:
            skipped.append(f"{k} — not in current model")
        elif model_state[k].shape != v.shape:
            skipped.append(f"{k} — shape mismatch: saved {v.shape} vs model {model_state[k].shape}")
        else:
            matched[k] = v
    model_state.update(matched)
    model.load_state_dict(model_state)
    print(f"Loaded:  {len(matched)}/{len(model_state)} layers")
    print(f"Skipped: {len(skipped)}")
    for s in skipped:
        print(f"  {s}")

load_weights(model, path)

def freeze_for_settransformer(model):
    freeze_prefixes = [
        'enc1', 'enc2', 'enc3',
        'stencil_token',
    ]
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in freeze_prefixes):
            param.requires_grad = False
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    frozen    = [(n, p.numel()) for n, p in model.named_parameters() if not p.requires_grad]
    print(f"Trainable: {sum(p for _, p in trainable):,}")
    print(f"Frozen:    {sum(p for _, p in frozen):,}")
    print("\nTrainable layers:")
    for n, p in trainable:
        print(f"  {n}: {p:,}")


optimizer = optim.Adam([
    {'params': model.traj_gte.parameters(),     'lr': 1e-4},   # gentle adaptation
    {'params': model.combined_gte.parameters(), 'lr': 1e-4},   # gentle adaptation
    {'params': [p for n, p in model.named_parameters() 
                if not any(x in n for x in ['traj_gte', 'combined_gte'])
                and p.requires_grad], 
     'lr': 1e-4}                                                # normal lr for rest
])

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer=optimizer,
    max_lr=5e-4,
    epochs=15,
    steps_per_epoch=len(loader),
    pct_start=0.2,
    div_factor=5,
    final_div_factor=500
)

OCObj = OneCycle(nb=20, max_lr=0.01) 
lr_finder = CLR(train_dataloader=loader)

EvaluateValidation(model = model, val_dataset = val_data, device = device)