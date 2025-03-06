import torch
import torch.nn as nn
from torch.nn import functional as F
from torchtune.modules import RotaryPositionalEmbeddings


# Add xformers import
try:
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
    print("successfully imported xformers")
    
except ImportError:
    XFORMERS_AVAILABLE = False
# XFORMERS_AVAILABLE = False    # manual xformers override in case it's numerically different, should be fixed now
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
    print("successfully imported flash attention")
except ImportError:
    FLASH_ATTN_AVAILABLE = False



class EfficientTransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.0, use_xformers=True, max_seq_len=4096):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = float(1 / (self.head_dim ** -0.5))     # xformers also requires 1/ already
        self.use_xformers = use_xformers and XFORMERS_AVAILABLE
        self.use_flash_attn = FLASH_ATTN_AVAILABLE
        
        # Keep the same parameter initialization for both implementations
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.use_dropout = dropout > 0
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        
        # RoPE (Rotary Position Embedding)
        self.rope = RotaryPositionalEmbeddings(self.head_dim, max_seq_len=max_seq_len)
    
    def forward(self, x, cls_attention_only=False):
        # Pre-norm for attention
        norm_x = self.norm1(x)
        batch_size, seq_len, hidden_size = norm_x.shape
        
        # Split the computation based on cls_attention_only for better efficiency
        if cls_attention_only:
            # Extract just what we need - CLS token only
            cls_token = x[:, 0:1]  # [batch, 1, hidden]
            
            # For CLS-only attention, only compute q for CLS token, but k,v for all tokens
            q_cls = self.q_proj(norm_x[:, 0:1]).view(batch_size, 1, self.num_heads, self.head_dim)  # Only compute q for CLS token
            k = self.k_proj(norm_x).view(batch_size, seq_len, self.num_heads, self.head_dim)  # Compute k for all tokens
            v = self.v_proj(norm_x).view(batch_size, seq_len, self.num_heads, self.head_dim)  # Compute v for all tokens

            # apply rope after view and BEFORE transpose
            q_cls = self.rope(q_cls)
            k = self.rope(k)
            
            # always do normal attention due to only having the single cls token that we need attention for
            if self.use_flash_attn:
                attn = flash_attn_func(q_cls, k, v, softmax_scale=self.scale)
            else:
                attn = self.normal_attention(q_cls, k, v)
            return self.post_attention_stuff(cls_token, attn.view(batch_size, 1, hidden_size))
        
        # Standard attention for all tokens (non-cls-only layers)
        else:
            # Compute Q, K, V for all tokens
            q = self.q_proj(norm_x).view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = self.k_proj(norm_x).view(batch_size, seq_len, self.num_heads, self.head_dim)
            v = self.v_proj(norm_x).view(batch_size, seq_len, self.num_heads, self.head_dim)

            # apply rope after view and BEFORE transpose
            q = self.rope(q)
            k = self.rope(k)
            if self.use_xformers and not self.use_flash_attn:
                attn = xops.memory_efficient_attention(q, k, v, scale=self.scale)
            else:
                if self.use_flash_attn:
                    attn = flash_attn_func(q, k, v, softmax_scale=self.scale)
                else:
                    attn = self.normal_attention(q, k, v)
        
        return self.post_attention_stuff(x, attn.view(batch_size, seq_len, hidden_size))
    
    def normal_attention(self, q, k, v):
        # Match xFormers' exact pattern of operations, https://facebookresearch.github.io/xformers/components/ops.html
        q = q * self.scale
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        attn = torch.matmul(attn, v)
        return attn.transpose(1, 2).contiguous()
    
    def post_attention_stuff(self, x, attn):
        attn = self.out_proj(attn)

        if self.use_dropout:
            attn = self.dropout(attn)

        # Residual connection and post-norm MLP
        x = x + attn
        mlp_out = self.mlp(self.norm2(x))

        if self.use_dropout:
            mlp_out = self.dropout(mlp_out)

        # residual
        x = x + mlp_out
        return x


class EfficientTransformer(nn.Module):
    def __init__(self, hidden_size, num_heads, num_layers, dropout=0.0, use_xformers=True, max_seq_len=4096):
        super().__init__()
        self.layers = nn.ModuleList([
            EfficientTransformerBlock(hidden_size, num_heads, dropout, use_xformers, max_seq_len)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.num_layers = num_layers
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # Optimize the last layer by only computing CLS token attention
            cls_attention_only = (i == self.num_layers - 1)
            x = layer(x, cls_attention_only=cls_attention_only)
        return self.norm(x)

class Dinov2ForTimeSeriesClassification(nn.Module):
    def __init__(self, size, num_classes, classifier_type, cls_option="patches_only", use_reg=True, dropout_rate=0.2):
        super().__init__()
        self.size = size
        self.num_classes = num_classes
        self.classifier_type = classifier_type
        self.cls_option = cls_option
        self.use_reg = use_reg
        self.return_class_token = cls_option == "both"
        self.dropout_rate = dropout_rate
        # Transformer parameters - keep fixed head count
        self.num_heads = 8
        self.num_layers = 6
        # Calculate exact context length based on patches and frames
        self.patches_per_frame = 13 * 18  # 234 patches per frame
        self.max_frames = 3
        # Determine transformer embedding dimension
        self.use_dino_embed_size = False
        self.transformer_dim = 64

        self.dinov2 = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{self.size[0].lower()}14{"_reg"*self.use_reg}')
        # print(self.dinov2)
        # self.dinov2_transformer_blocks = len(self.dinov2.blocks)
        # print("BLOCKS:", self.dinov2_transformer_blocks)
        self.dinov2_embed_dim = self.dinov2.embed_dim
        
        # use the specified loss function
        if self.classifier_type == "bce":
            self.loss_fct = torch.nn.BCEWithLogitsLoss()
        else:
            self.loss_fct = torch.nn.CrossEntropyLoss()

        #
        if self.use_dino_embed_size or self.transformer_dim is None:
            self.embed_dim = self.dinov2_embed_dim
        else:
            self.embed_dim = self.transformer_dim
            # Need dimension to be divisible by 2*num_heads for rotary embeddings, and some other limitations for xformers attention, but I decided against
            # auto fixing it, let the user set the correct transformer_dim!
            
        # Add projection layer from DinoV2 embedding to transformer embedding
        if self.dinov2_embed_dim != self.embed_dim:
            self.projection = nn.Linear(self.dinov2_embed_dim, self.embed_dim)
        else:
            self.projection = nn.Identity()
        
        # Set context length based on configuration
        if cls_option == "patches_only":
            # All patches + 1 CLS token
            self.context_length = self.patches_per_frame * self.max_frames + 1
        else:  # "both"
            # All patches + frame CLS tokens + 1 learnable CLS token
            self.context_length = self.patches_per_frame * self.max_frames + self.max_frames + 1
        
        # Learnable CLS token (uses transformer embedding dimension)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        
        # Transformer encoder with xformers
        self.transformer = EfficientTransformer(
            hidden_size=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=dropout_rate,
            use_xformers=True,  # Ensure xformers is used
            max_seq_len=self.context_length
        )
        
        # Classification head (from transformer dimension to num_classes)
        self.fc_head = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x, labels=None):
        batch_size, seq_len, channels, height, width = x.shape
        
        # Process all frames through DinoV2
        x = x.reshape(-1, channels, height, width)
        # for some reason n=1 is the last layer. n=1 default so we omit, if want to specify specific transformer block add n=, (indexing is "backwards")
        # patch should be (batch*seq_len, width*height, embed_dim) and cls (batch*seq_len, embed_dim)
        out_tuple = self.dinov2.get_intermediate_layers(x, reshape=False, return_class_token=self.return_class_token, norm=True)[0]
        
        # Prepare learnable CLS token
        learnable_cls = self.cls_token.expand(batch_size, 1, -1)
        
        # Handle sequence composition based on configuration
        if self.cls_option == "patches_only":
            # Project the patches first (linear operation only cares about last dim)
            # proj out is [batch*seq_len, patches_per_frame, embed_dim]
            out = self.projection(out_tuple).reshape(batch_size, seq_len * self.patches_per_frame, self.embed_dim)
        else:  # "both"
            # out_tuple[1] shape: [batch*seq_len, dinov2_embed_dim]
            # Add dimension to match patches: [batch*seq_len, 1, dinov2_embed_dim]
            cls_tokens = out_tuple[1].unsqueeze(1)
            
            # Concatenate CLS with patches: [batch*seq_len, 1+patches_per_frame, dinov2_embed_dim]
            # Project once: [batch*seq_len, 1+patches_per_frame, embed_dim] and
            # Reshape: [batch, seq_len*(1+patches_per_frame), embed_dim]
            out = self.projection(torch.cat([cls_tokens, out_tuple[0]], dim=1)).reshape(batch_size, seq_len * (1 + self.patches_per_frame), self.embed_dim)
        # prepend cls to dino output
        out = torch.cat([learnable_cls, out], dim=1)
        
        # Apply transformer
        out = self.transformer(out)[:, 0]  # Shape: [batch, embed_dim] (squeezes size 1 dim in between, transformer output is already only "1" size due to only outputting cls token, but we have to squeeze the dim)
        
        # Final classification using the cls token
        out = self.fc_head(out)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            last_label = labels[:, -1, :]
            loss = self.loss_fct(out.view(-1, self.num_classes), last_label.view(-1, self.num_classes))
                
        return out, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        # fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        fused_available = True  # wasn't working with dino so i disabled
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer