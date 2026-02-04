import torch
import torch.nn as nn

class CrossOmicsAttentionNet(nn.Module):
    def __init__(
        self,
        clin_dim=12,
        rna_dim=5000, 
        mirna_dim=500,
        embed_dim=32, # Small embedding helps prevent overfitting on small N
        num_heads=4,
        num_classes=4,
        dropout=0.25,
    ):
        super().__init__()
        
        self.clin_dim = clin_dim
        self.rna_dim = rna_dim
        self.mirna_dim = mirna_dim
        
        # Modality-specific encoders
        self.clin_proj = nn.Sequential(
            nn.Linear(clin_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.rna_proj = nn.Sequential(
            nn.Linear(rna_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.mirna_proj = nn.Sequential(
            nn.Linear(mirna_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x):
        # Manually slice the combined feature vector back into modalities
        x_clin = x[:, 0:self.clin_dim]
        x_rna = x[:, self.clin_dim : self.clin_dim + self.rna_dim]
        x_mirna = x[:, self.clin_dim + self.rna_dim :]
        
        clin_emb = self.clin_proj(x_clin)
        rna_emb = self.rna_proj(x_rna)
        mirna_emb = self.mirna_proj(x_mirna)
        
        # Cross-modal attention: Clinical acts as Query for Omics Keys/Values
        query = clin_emb.unsqueeze(1)
        key_value = torch.stack([rna_emb, mirna_emb], dim=1)
        
        attn_output, _ = self.attention(query, key_value, key_value)
        logits = self.classifier(attn_output.squeeze(1))
        
        return logits