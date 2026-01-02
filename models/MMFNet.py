import torch
from torch import nn
import torch.nn.functional as F
import torch_dct as dct
import math


# Assuming RevIN is in your local path
# from layers.RevIN import RevIN

class GatingNetwork(nn.Module):
    """
    Optional gating mechanism to weigh the importance of different experts.
    """

    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        # Added dim=-1 to ensure softmax is applied across the expert weights
        return F.softmax(self.fc(x), dim=-1)


class MultiScaleDCTModel(nn.Module):
    def __init__(self, configs, revin=True):
        super().__init__()

        # Model Parameters
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.latent_n = configs.nz
        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len

        # Normalization Layer
        # self.revin_layer = RevIN(self.enc_in) if revin else None

        # Learnable Frequency Masks
        self.mask1 = nn.Parameter(torch.ones(self.seq_len) * 0.1)
        self.mask2 = nn.Parameter(torch.ones(self.seq_len) * 0.1)
        self.mask3 = nn.Parameter(torch.ones(self.seq_len) * 0.1)

        self.dropout = nn.Dropout(0.1)

        # Multi-scale Feature Extractors (Experts)
        self.mlp1 = nn.Linear(self.seq_len, self.latent_n)
        self.mlp2 = nn.Linear(self.seq_len, self.latent_n)
        self.mlp3 = nn.Linear(self.seq_len, self.latent_n)

        # Final projection back to prediction length
        self.inverse_mlp = nn.Linear(self.latent_n * 3, self.pred_len)

    def forward(self, x):
        """
        Input x shape: [Batch, Seq_Len, Channels]
        """
        # 1. Instance Normalization (RevIN)
        # if self.revin_layer:
        #     x = self.revin_layer(x, 'norm')

        # 2. Local Normalization & Permute
        batch_size = x.shape[0]
        seq_mean = torch.mean(x, dim=1, keepdim=True)
        x_var = torch.var(x, dim=1, keepdim=True) + 1e-5

        # Switch to [Batch, Channels, Seq_Len] for linear layers
        x = (x - seq_mean).permute(0, 2, 1)

        # 3. Multi-Scale Processing
        # Scales: Could be moved to config to avoid hardcoding
        scale1 = 2
        scale2 = 24  # Standardized from your multiple assignments
        scale3 = self.seq_len

        x_v1 = self._process_scale(x, scale1, self.mask1, self.mlp1)
        x_v2 = self._process_scale(x, scale2, self.mask2, self.mlp2)
        x_v3 = self._process_scale(x, scale3, self.mask3, self.mlp3)

        # 4. Fusion and Inverse Transform
        x_m = torch.cat([x_v1, x_v2, x_v3], dim=-1)
        x_out = self.inverse_mlp(x_m)

        # Return to time domain via Inverse DCT
        x_out = dct.idct(x_out).permute(0, 2, 1)

        # 5. Denormalization
        x_out = x_out * self.length_ratio * torch.sqrt(x_var)
        x_out = x_out + seq_mean

        # if self.revin_layer:
        #     x_out = self.revin_layer(x_out, 'denorm')

        return x_out

    def _process_scale(self, x, scale, mask, expert_mlp):
        """
        Processes the input through DCT, applies a scale-based mask,
        and projects through an MLP.
        """
        batch_size, channels, _ = x.shape

        # Frequency Domain Transformation
        x_dct = dct.dct(x)

        # Apply learnable mask
        x_dct = x_dct * mask

        # Specific frequency filtering logic (as per your original code)
        # Note: This logic assumes specific relationships between seq_len and latent_n
        if scale == self.seq_len:
            x_dct[:, :, -int(self.latent_n * 2 / 3):] = 0
        elif scale == self.seq_len // 4:
            x_dct[:, :, :int(self.latent_n * 3 / 4)] = 0
            x_dct[:, :, -int(self.latent_n * 3 / 16):] = 0
        else:
            x_dct[:, :, :int(self.latent_n * 5 / 6)] = 0

        # Project features
        return expert_mlp(x_dct)