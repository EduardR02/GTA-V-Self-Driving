import torch
import torch.nn as nn
from torch.nn import functional as F
from torchtune.modules import RotaryPositionalEmbeddings
import os


try:
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
    print("successfully imported xformers")
    
except ImportError:
    XFORMERS_AVAILABLE = False
    print("xformers not available")
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
    print("successfully imported flash attention")
except ImportError:
    print("flash attention not available")
    FLASH_ATTN_AVAILABLE = False


# pos_weights = torch.tensor([0.585, 11.285, 25.556, 11.392]) # calculated from data

class EfficientTransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.0, use_xformers=True, max_seq_len=4096, cls_attention_only=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = float(1 / (self.head_dim ** -0.5))     # xformers also requires 1/ already
        self.use_xformers = use_xformers and XFORMERS_AVAILABLE
        self.use_flash_attn = FLASH_ATTN_AVAILABLE
        self.cls_attention_only = cls_attention_only
        
        # Keep the same parameter initialization for both implementations
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.use_dropout = dropout > 0
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.RMSNorm(hidden_size)
        self.norm2 = nn.RMSNorm(hidden_size)

        # swiglu mlp
        self.multiple_of = 64
        self.hidden_dim = 4 * hidden_size
        self.hidden_dim = int(2 * self.hidden_dim / 3)  # to keep param count roughly same as normal 4 * hidden_size (with swiglu we need ffn matrices instead of one)
        self.hidden_dim = self.multiple_of * ((self.hidden_dim + self.multiple_of - 1) // self.multiple_of)      # round up to multiple_of
        
        self.mlp_linear_1 = nn.Linear(self.hidden_size, self.hidden_dim, bias=False)
        self.mlp_linear_2 = nn.Linear(self.hidden_size, self.hidden_dim, bias=False)
        self.mlp_linear_3 = nn.Linear(self.hidden_dim, self.hidden_size, bias=False)
        
        # RoPE (Rotary Position Embedding)
        self.rope = RotaryPositionalEmbeddings(self.head_dim, max_seq_len=max_seq_len)
        
        # Select attention implementation based on available backends
        if self.use_flash_attn:
            self.attn_func = self._flash_attention
        elif self.use_xformers:
            self.attn_func = self._xformers_attention
        else:
            self.attn_func = self.sdpa
        
        # Bind appropriate forward method based on attention mode
        if cls_attention_only:
            self.forward = self._forward_cls_only
        else:
            self.forward = self._forward_full

    def swiglu_mlp(self, x):
        # swiglu(x) = (xW1 + b1) * swish(xW2 + b2), where swish(x) = x * sigmoid(x)
        return self.mlp_linear_3(F.silu(self.mlp_linear_1(x)) * self.mlp_linear_2(x))
    
    def _flash_attention(self, q, k, v):
        """Flash attention wrapper"""
        return flash_attn_func(q, k, v, softmax_scale=self.scale)
    
    def _xformers_attention(self, q, k, v):
        """xFormers memory efficient attention wrapper"""
        return xops.memory_efficient_attention(q, k, v, scale=self.scale)
    
    def _forward_cls_only(self, x):
        """Forward pass that only computes CLS token attention (last layer optimization)"""
        norm_x = self.norm1(x)
        batch_size, seq_len, hidden_size = norm_x.shape
        
        # Only compute query for CLS token, but key/value for all tokens
        q_cls = self.q_proj(norm_x[:, 0:1]).view(batch_size, 1, self.num_heads, self.head_dim)
        k = self.k_proj(norm_x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(norm_x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        q_cls = self.rope(q_cls)
        k = self.rope(k)
        
        attn = self.attn_func(q_cls, k, v)
        attn = attn.view(batch_size, 1, hidden_size)
        x = x[:, 0:1]  # Keep only CLS token
        
        return self.post_attention_stuff(x, attn)
    
    def _forward_full(self, x):
        """Forward pass that computes full attention for all tokens"""
        norm_x = self.norm1(x)
        batch_size, seq_len, hidden_size = norm_x.shape
        
        # Compute Q, K, V for all tokens
        q = self.q_proj(norm_x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(norm_x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(norm_x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # apply rope after view and BEFORE transpose
        q = self.rope(q)
        k = self.rope(k)
        
        attn = self.attn_func(q, k, v)
        attn = attn.view(batch_size, seq_len, hidden_size)
        
        return self.post_attention_stuff(x, attn)
    
    def sdpa(self, q, k, v):
        """PyTorch SDPA fallback - automatically uses optimized kernels when available"""
        # SDPA expects (batch, heads, seq, head_dim) but we have (batch, seq, heads, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # PyTorch SDPA handles scaling internally
        attn = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        
        return attn.transpose(1, 2).contiguous()
    
    def post_attention_stuff(self, x, attn):
        attn = self.out_proj(attn)

        if self.use_dropout:
            attn = self.dropout(attn)

        # Residual connection and post-norm MLP
        x = x + attn
        mlp_out = self.swiglu_mlp(self.norm2(x))

        if self.use_dropout:
            mlp_out = self.dropout(mlp_out)

        # residual
        x = x + mlp_out
        return x


class EfficientTransformer(nn.Module):
    def __init__(self, hidden_size, num_heads, num_layers, dropout=0.0, use_xformers=True, max_seq_len=4096):
        super().__init__()
        # Last layer only computes CLS token attention (we only use [:, 0] anyway)
        self.layers = nn.ModuleList([
            EfficientTransformerBlock(
                hidden_size, num_heads, dropout, use_xformers, max_seq_len,
                cls_attention_only=(i == num_layers - 1)
            )
            for i in range(num_layers)
        ])
        self.norm = nn.RMSNorm(hidden_size)
        self.num_layers = num_layers
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class Dinov2ForTimeSeriesClassification(nn.Module):
    def __init__(self, size, num_classes, classifier_type, cls_option="patches_only", use_reg=True, dropout_rate=0.0, use_pos_weights=True):
        super().__init__()
        self.size = size
        self.num_classes = num_classes
        self.classifier_type = classifier_type
        self.cls_option = cls_option
        self.use_reg = use_reg
        self.return_class_token = cls_option == "both"
        self.dropout_rate = dropout_rate
        # Transformer parameters - keep fixed head count
        self.num_heads = 6
        self.num_layers = 6
        # Calculate exact context length based on patches and frames
        self.patches_per_frame = 13 * 18  # 234 patches per frame
        self.max_frames = 3
        # Determine transformer embedding dimension
        self.use_dino_embed_size = False
        self.transformer_dim = 48

        self.dinov2 = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{self.size[0].lower()}14{"_reg"*self.use_reg}')
        # print(self.dinov2)
        # self.dinov2_transformer_blocks = len(self.dinov2.blocks)
        # print("BLOCKS:", self.dinov2_transformer_blocks)
        self.dinov2_embed_dim = self.dinov2.embed_dim
        
        # use the specified loss function
        if self.classifier_type == "bce":
            if use_pos_weights:
                pos_weights = torch.tensor([1.0, 1.0, 1.0, 1.0])  # Start unbiased, training will schedule
                self.loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
            else:
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

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, muon_lr=None):
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


class Dinov3ForTimeSeriesClassification(nn.Module):
    def __init__(self, size, num_classes, dropout_rate=0.0, 
                 repo_dir="dinov3", dtype=torch.bfloat16, cls_option="patches_only", use_transformers=False, use_pos_weights=True):
        super().__init__()
        self.size = size
        self.num_classes = num_classes
        self.expected_input_hw = (192, 240) # 16x16 size patches
        self.dropout_rate = dropout_rate
        self.max_frames = 3
        self.use_transformers = use_transformers
        self.cls_option = cls_option

        # Aggregator params
        self.num_heads = 8
        self.num_layers = 6
        self.use_dino_embed_size = False
        self.transformer_dim = 128

        if use_transformers:
            from transformers import AutoModel
            
            size2hf = {
                "base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
                "large": "facebook/dinov3-vitl16-pretrain-lvd1689m", 
                "huge": "facebook/dinov3-vith16-pretrain-lvd1689m",
            }
            
            self.dinov3 = AutoModel.from_pretrained(
                size2hf[self.size],
                device_map="auto",
                dtype=dtype,
                attn_implementation="sdpa"
            )
            self.dinov3_embed_dim = self.dinov3.config.hidden_size
            self.num_register_tokens = self.dinov3.config.num_register_tokens
            print(f"HF DINOv3 {size}: {self.dinov3_embed_dim}D, {self.num_register_tokens} register tokens")
            
        else:
            size2hub = {
                "base":  "dinov3_vitb16",
                "large": "dinov3_vitl16", 
                "huge":  "dinov3_vith16",
            }
            # replace with your path to weights (also you will need to clone the dinov3 repo)
            weights_path = {
                "base": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
                "large": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
                "huge": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
            }
            weights_path_full = os.path.join(repo_dir, "weights", weights_path[self.size])
            self.dinov3 = torch.hub.load(repo_dir, size2hub[self.size], source='local', weights=weights_path_full)
            
            self.dinov3 = self.dinov3.to(dtype) # immediately convert model (tiny bit faster than doing nothing and letting autocast handle it)
            
            self.dinov3_embed_dim = self.dinov3.norm.weight.shape[0]
            self.num_register_tokens = getattr(self.dinov3, 'num_register_tokens', 0)

        # Bind specialized token extraction and forward methods based on configuration
        self.is_patches_only = (cls_option == "patches_only")
        
        if self.use_transformers:
            if self.is_patches_only:
                self._token_extractor = self._extract_tokens_patches_only_transformers
                self._forward_processor = self._forward_patches_only
            else:  # "both"
                self._token_extractor = self._extract_tokens_both_transformers
                self._forward_processor = self._forward_both
        else:
            if self.is_patches_only:
                self._token_extractor = self._extract_tokens_patches_only_torch_hub
                self._forward_processor = self._forward_patches_only
            else:  # "both"
                self._token_extractor = self._extract_tokens_both_torch_hub
                self._forward_processor = self._forward_both

        # Compute patches per frame
        with torch.no_grad():
            device = next(self.dinov3.parameters()).device
            dummy = torch.randn(1, 3, *self.expected_input_hw, device=device, dtype=dtype)
            test_output = self._token_extractor(dummy)
            
            self.patches_per_frame = test_output.shape[1] if self.is_patches_only else test_output[0].shape[1]
            
            del dummy, test_output
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        print(f"DINOv3 {size}: {self.dinov3_embed_dim}D, {self.patches_per_frame} patches/frame, cls_option={cls_option}")
        print(f"Training context: {self.max_frames} frames = ~{1 + self.patches_per_frame * self.max_frames + (self.max_frames if not self.is_patches_only else 0)} tokens")

        pos_weights = torch.tensor([1.0, 11.285, 25.556/2, 11.392])
        self.loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)

        # Set embedding dimensions
        if self.use_dino_embed_size or self.transformer_dim is None:
            self.embed_dim = self.dinov3_embed_dim
        else:
            self.embed_dim = self.transformer_dim
        
        self.projection = nn.Linear(self.dinov3_embed_dim, self.embed_dim) if self.dinov3_embed_dim != self.embed_dim else nn.Identity()
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        # Calculate context length for current training config
        context_length = 1 + self.patches_per_frame * self.max_frames
        if not self.is_patches_only:
            context_length += self.max_frames  # Add frame CLS tokens
        
        # Set RoPE to support more frames for future extension without retraining
        # Use FIXED maximum to ensure checkpoint compatibility when changing max_frames
        MAX_FRAMES_SUPPORTED = 10  # Fixed value - same for all checkpoints regardless of training frames
        rope_max_seq_len = 1 + self.patches_per_frame * MAX_FRAMES_SUPPORTED
        if not self.is_patches_only:
            rope_max_seq_len += MAX_FRAMES_SUPPORTED
        
        print(f"RoPE supports up to {MAX_FRAMES_SUPPORTED} frames ({rope_max_seq_len} tokens) for future extension")

        self.transformer = EfficientTransformer(
            hidden_size=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=dropout_rate,
            use_xformers=True,
            max_seq_len=rope_max_seq_len
        )

        self.fc_head = nn.Linear(self.embed_dim, num_classes)

    def _extract_tokens_patches_only_transformers(self, x):
        """Extract patch tokens only - HuggingFace Transformers version"""
        outputs = self.dinov3(pixel_values=x)
        hidden_states = outputs.last_hidden_state
        # Skip CLS (pos 0) and register tokens, keep patches only
        return hidden_states[:, 1 + self.num_register_tokens:, :]
    
    def _extract_tokens_both_transformers(self, x):
        """Extract CLS + patch tokens - HuggingFace Transformers version"""
        outputs = self.dinov3(pixel_values=x)
        hidden_states = outputs.last_hidden_state
        cls_tokens = hidden_states[:, 0:1, :]  # CLS token at position 0
        patch_tokens = hidden_states[:, 1 + self.num_register_tokens:, :]  # Skip CLS + register tokens
        return patch_tokens, cls_tokens
    
    def _extract_tokens_patches_only_torch_hub(self, x):
        """Extract patch tokens only - torch.hub version"""
        return self.dinov3.get_intermediate_layers(
            x, n=1, reshape=False, return_class_token=False, return_extra_tokens=False, norm=True
        )[0]
    
    def _extract_tokens_both_torch_hub(self, x):
        """Extract CLS + patch tokens - torch.hub version"""
        patch_tokens, cls_tokens = self.dinov3.get_intermediate_layers(
            x, n=1, reshape=False, return_class_token=True, return_extra_tokens=False, norm=True
        )[0]
        # torch.hub returns cls_tokens as [batch*seq_len, embed_dim], add dimension for concatenation
        cls_tokens = cls_tokens.unsqueeze(1)  # [batch*seq_len, 1, embed_dim]
        return patch_tokens, cls_tokens
    
    def _forward_patches_only(self, x, batch_size, seq_len):
        """Process frames with patch tokens only"""
        patch_tokens = self._token_extractor(x)  # [batch*seq_len, patches_per_frame, dinov3_embed_dim]
        out = self.projection(patch_tokens).reshape(batch_size, seq_len * self.patches_per_frame, self.embed_dim)
        return out
    
    def _forward_both(self, x, batch_size, seq_len):
        """Process frames with both CLS and patch tokens"""
        patch_tokens, cls_tokens = self._token_extractor(x)
        # Combine CLS + patches for each frame: [batch*seq_len, 1+patches_per_frame, dinov3_embed_dim]
        combined = torch.cat([cls_tokens, patch_tokens], dim=1)
        out = self.projection(combined).reshape(batch_size, seq_len * (1 + self.patches_per_frame), self.embed_dim)
        return out
    
    def forward(self, x, labels=None):
        batch_size, seq_len = x.shape[:2]
        x = x.reshape(-1, *x.shape[2:])  # [batch*seq_len, C, H, W]

        # Process DINOv3 features
        out = self._forward_processor(x, batch_size, seq_len)

        # Prepend learnable CLS token and apply transformer
        learnable_cls = self.cls_token.expand(batch_size, 1, -1)
        out = torch.cat([learnable_cls, out], dim=1)
        out = self.transformer(out)[:, 0]  # Extract CLS representation
        out = self.fc_head(out)

        loss = None
        if labels is not None:
            last_label = labels[:, -1]
            loss = self.loss_fct(out, last_label)
        
        return out, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, muon_lr=None):
        from muon import SingleDeviceMuonWithAuxAdam
        # Hidden/body modules to train with Muon (weights only): aggregator only
        hidden_modules = [self.transformer]
        hidden_params = [p for m in hidden_modules for p in m.parameters() if p.requires_grad]
        hidden_weights = [p for p in hidden_params if p.ndim >= 2]
        hidden_gains_biases = [p for p in hidden_params if p.ndim < 2]

        # Non-hidden parts to train with AdamW (backbone + projection + head + embeddings)
        nonhidden_modules = [self.dinov3, self.projection, self.fc_head]
        nonhidden_params = [p for m in nonhidden_modules for p in m.parameters() if p.requires_grad]
        nonhidden_params = nonhidden_params + [self.cls_token]

        # Split AdamW params into decay/no-decay to keep prior policy
        adam_decay = [p for p in nonhidden_params if getattr(p, "ndim", 0) >= 2]
        adam_nodecay = [p for p in nonhidden_params if getattr(p, "ndim", 0) < 2]
        # Add all gains/biases (1D) from hidden to no-decay AdamW
        adam_nodecay += hidden_gains_biases

        # Sanity prints: counts of parameters in each group
        hidden_num_tensors = len(hidden_params)
        hidden_num_params = sum(p.numel() for p in hidden_params)
        muon_num_tensors = len(hidden_weights)
        muon_num_params = sum(p.numel() for p in hidden_weights)
        adam_all = adam_decay + adam_nodecay
        adam_num_tensors = len(adam_all)
        adam_num_params = sum(p.numel() for p in adam_all)
        print(f"hidden params: {hidden_num_tensors} tensors, {hidden_num_params:,} parameters")
        print(f"Muon-optimized (hidden weights): {muon_num_tensors} tensors, {muon_num_params:,} parameters")
        print(f"AdamW-optimized (head + gains/biases): {adam_num_tensors} tensors, {adam_num_params:,} parameters")

        muon_lr = 0.02 if muon_lr is None else muon_lr  # default if not provided

        param_groups = [
            dict(params=hidden_weights, use_muon=True,
                 lr=muon_lr, weight_decay=weight_decay),
            dict(params=adam_decay, use_muon=False,
                 lr=learning_rate, betas=betas, weight_decay=weight_decay),
            dict(params=adam_nodecay, use_muon=False,
                 lr=learning_rate, betas=betas, weight_decay=0.0),
        ]
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
        return optimizer