import numpy as np
import config
import torch
import os
from contextlib import nullcontext
from dataloader import get_dataloader
from model import Dinov2ForTimeSeriesClassification, Dinov3ForTimeSeriesClassification
import matplotlib.pyplot as plt
import math
import time
import psutil
from tqdm import tqdm


current_data_dirs = [config.stuck_data_dir_name, config.turns_data_dir_name, config.new_data_dir_name]  # has to be list

# Get the current process
process = psutil.Process(os.getpid())
# Set the priority to "High Priority" class
process.nice(psutil.HIGH_PRIORITY_CLASS)
print(torch.backends.cudnn.version())
print(torch.version.cuda)
# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = os.path.join('models', 'dinov3')
eval_interval = 1000
log_interval = 10
eval_iters = 10
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
fine_tune = False   # train the entire model or just the top
freeze_non_dino_layers = False
init_from = 'scratch' # 'scratch' or 'resume'
dino_size = "base"  # small is ~21M, base ~86M, large ~300M, giant ~1.1B
use_dino_registers = True
load_checkpoint_name = "v3_base_12layer.pt"
save_checkpoint_name = "v3_base_12layer.pt"
metrics_name = "v3_base_12layer.png"
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
batch_size = 128    # if gradient_accumulation_steps > 1, this is the micro-batch size
train_split = 0.95   # test val split, keep same for resume
convert_to_greyscale = False
sequence_len = 3
sequence_stride = 20
flip_prob = 0.33
warp_prob = 0.1
zoom_prob = 0.3
dropout_p = 0.1     # 0 to disable
classifier_type = "bce" # "cce" or "bce"
restart_schedules = False
cls_option = "both"    # "cls_only", "both", or "patches_only"
shift_labels = False
show_per_class_during_training = True

# adamw optimizer
learning_rate = 3e-4 # max learning rate
# Global Muon learning rate (used for param groups with use_muon=True)
muon_learning_rate = 0.02   # 0.02 is recommended by Muon
max_iters = 100000 # total number of training iterations
# optimizer settings
weight_decay = 0.075
beta1 = 0.9
beta2 = 0.995
grad_clip = 0.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 1000 # how many steps to warm up for
lr_decay_iters = 60000 # should be ~= max_iters per Chinchilla
min_lr = 1e-6 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# pos_weights scheduling settings  
schedule_pos_weights = False # whether to schedule pos_weights during training
pos_warmup_iters = 1000 # how many steps to use no pos_weights (same as lr warmup)
pos_decay_iters = 60000 # when to finish ramping up pos_weights (same as lr)
start_pos_weights = [1.0, 1.0, 1.0, 1.0] # starting weights (no bias)
end_pos_weights = [0.585, 11.285, 25.556, 11.392] # ending weights (calculated from data)
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
# change this to bf16 if your gpu actually supports it
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config_dict = {k: globals()[k] for k in config_keys} # will be useful for logging
num_workers = 1
os.makedirs(out_dir, exist_ok=True)
train_dataloader, val_dataloader = None, None
if classifier_type == "bce":
    id2label = {0: "w", 1: "a", 2: "s", 3: "d"}
else:
    id2label = config.outputs



torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


def init_dataloaders():
    # put this in a function to not load it when inference script imports from here
    global train_dataloader, val_dataloader
    train_dataloader = get_dataloader(current_data_dirs, batch_size, train_split, True, classifier_type,
                                    sequence_len, sequence_stride, flip_prob, warp_prob, zoom_prob, shift_labels=shift_labels, shuffle=True, num_workers=num_workers)
    val_dataloader = get_dataloader(current_data_dirs, batch_size, train_split, False, classifier_type,
                                    sequence_len, sequence_stride, flip_prob, warp_prob, zoom_prob, shift_labels=shift_labels, shuffle=True, num_workers=num_workers)


iter_num = 0
best_val_loss = 1e9
iter_num_on_load = 0
model = optimizer = scaler = None


# claude wrote a quick check to see if there are any conversion issues, after taking a look i decided to inference in fp16 for 60% more fps
def check_fp16_conversion_issues(model):
    # Save original FP32 model parameters
    fp32_params = {name: param.clone().detach().cpu() for name, param in model.named_parameters()}
    
    # Convert to FP16 and back to FP32
    model = model.half().float()
    
    # Check for large discrepancies or INF/NaN values
    issues_found = False
    for name, param in model.named_parameters():
        # Calculate original value range
        orig_param = fp32_params[name]
        orig_max = orig_param.abs().max().item()
        
        # Calculate difference
        diff = (param - orig_param).abs()
        rel_diff = diff / (orig_param.abs() + 1e-8)  # Avoid div by zero
        max_diff = diff.max().item()
        max_rel_diff = rel_diff.max().item()
        
        # Check if values got clipped (FP16 max is ~65504)
        clip_threshold = 65000  # Just below FP16 max
        clipped_values = (orig_param.abs() > clip_threshold).sum().item()
        
        # Check if values got zeroed (FP16 min is ~6e-5)
        min_threshold = 1e-4  # Just above FP16 min
        zeroed_values = ((orig_param.abs() < min_threshold) & 
                         (orig_param.abs() > 0) & 
                         (param == 0)).sum().item()
        
        if max_rel_diff > 0.01 or clipped_values > 0 or zeroed_values > 0:
            issues_found = True
            print(f"Issue in {name}:")
            print(f"  Original range: [{orig_param.min().item()}, {orig_param.max().item()}]")
            print(f"  Max absolute difference: {max_diff}")
            print(f"  Max relative difference: {max_rel_diff*100:.2f}%")
            if clipped_values > 0:
                print(f"  {clipped_values} values likely clipped (above {clip_threshold})")
            if zeroed_values > 0:
                print(f"  {zeroed_values} small values likely zeroed")
                
    if not issues_found:
        print("No significant FP16 conversion issues detected")
    
    return issues_found


def load_model(sample_only=False):
    global model, optimizer, scaler, iter_num, best_val_loss, iter_num_on_load
    checkpoint = None
    if init_from == 'scratch':
        print("Initializing a new model from scratch")
        if config.use_dinov3:
            model = Dinov3ForTimeSeriesClassification(dino_size, len(id2label), dropout_rate=dropout_p, dtype=ptdtype, cls_option=cls_option)
        else:
            model = Dinov2ForTimeSeriesClassification(dino_size, len(id2label), classifier_type=classifier_type, cls_option=cls_option, use_reg=use_dino_registers, dropout_rate=dropout_p)
    elif init_from == 'resume':
        print(f"Resuming training from {out_dir}")
        ckpt_path = os.path.join(out_dir, load_checkpoint_name)
        checkpoint = torch.load(ckpt_path, map_location=device)
        if config.use_dinov3:
            model = Dinov3ForTimeSeriesClassification(dino_size, len(id2label), dropout_rate=dropout_p, dtype=ptdtype, cls_option=cls_option)
        else:
            model = Dinov2ForTimeSeriesClassification(dino_size, len(id2label), classifier_type=classifier_type, cls_option=cls_option, use_reg=use_dino_registers, dropout_rate=dropout_p)
        state_dict = checkpoint['model']
        print(checkpoint["config"])   # if you forgot your model setup lol
        # added back in from nanogpt, apparently torch compile adds this
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        # print({(k, v.shape) for k, v in state_dict.items() if not 'dinov2' in k})
        model.load_state_dict(state_dict, strict=False)
        # check_fp16_conversion_issues(model)
        iter_num = checkpoint['iter_num']
        iter_num_on_load = iter_num
        best_val_loss = checkpoint['best_val_loss']
        del state_dict  # very important to clear this checkpoint reference

    if sample_only:
        model.eval()
        for module in model.modules():
            module.eval()
        model.to(device)
        del checkpoint
        if compile:
            print("compiling the model...")
            model = torch.compile(model)    # requires PyTorch 2.0
        torch.cuda.empty_cache()
        return model

    for module in model.modules():
        module.train()

    # freeze or unfreeze model
    for name, param in model.named_parameters():
        if "dinov2" in name or "dinov3" in name:
            param.requires_grad = fine_tune
        if sequence_len > 1 and freeze_non_dino_layers and "classifier" in name and "layer_norm" not in name:
            param.requires_grad = False
            print(f"Freezing {name}")
        # maybe also freeze layernorm no matter what here

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    model.to(device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

    # optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type, muon_lr=muon_learning_rate)
    if init_from == 'resume':
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except Exception as e:
            print(e)
            print("Probably trying to fine-tune full model after resuming from only last layer tuned model, proceeding with reset iter num for learning rate warmup.")
            iter_num = 0
        # iter_num = 0
        # Issue when loading model VRAM usage is higher, therefore OOMs with same params
        # https://discuss.pytorch.org/t/gpu-memory-usage-increases-by-90-after-torch-load/9213/3
        del checkpoint  # dereference seems crucial
        torch.cuda.empty_cache()
    if restart_schedules:
        iter_num = 0
    # compile the model
    if compile:
        print("compiling the model...")
        model = torch.compile(model)    # requires PyTorch 2.0

    return model


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    
    total_eval_steps = eval_iters * gradient_accumulation_steps
    
    for split in ['train', 'val']:
        dataloader_iter = iter(train_dataloader if split == 'train' else val_dataloader)
        X, Y, Y_CPU, dataloader_iter = get_batch(dataloader_iter, split)
        
        # Pre-allocate on GPU for efficiency, transfer at end
        losses_gpu = torch.zeros(total_eval_steps, dtype=torch.float32, device=device)
        
        # Always accumulate for batch processing (more efficient than per-step)
        all_logits = []
        all_labels = []
        
        next_y_cpu = None
        for step in tqdm(range(total_eval_steps), desc=f"{split.capitalize()} Eval"):
            with ctx:
                logits, loss = model(X, labels=Y)
            
            # Async prefetch next batch while model is doing the forward pass on the GPU
            if step < total_eval_steps - 1:
                X, Y, next_y_cpu, dataloader_iter = get_batch(dataloader_iter, split)
            
            # Store loss on GPU - avoid CPU transfers in loop
            losses_gpu[step] = loss.detach()
            
            # Always accumulate for batch processing (more efficient)
            all_logits.append(logits.detach().cpu())
            all_labels.append(Y_CPU)
            
            Y_CPU = next_y_cpu
        
        # Efficient batch operations at the end
        out[split] = losses_gpu.mean().cpu().item()  # Single GPU->CPU transfer
        
        # Batch process all accuracy calculations
        stacked_logits = torch.cat(all_logits, dim=0)
        stacked_labels = torch.cat(all_labels, dim=0)
        
        if show_per_class_during_training:
            overall_accuracy, per_class_metrics = calc_accuracy(stacked_logits, stacked_labels, return_per_class=True)
            out[split + "_accuracy"] = overall_accuracy
            out[split + "_per_class"] = per_class_metrics
        else:
            overall_accuracy = calc_accuracy(stacked_logits, stacked_labels, return_per_class=False)
            out[split + "_accuracy"] = overall_accuracy
        
        # Clean up accumulated tensors immediately
        del all_logits, all_labels, stacked_logits, stacked_labels, losses_gpu
    
    # Cleanup
    del X, Y, Y_CPU, next_y_cpu, dataloader_iter
    torch.cuda.empty_cache()
    model.train()
    return out


def print_evaluation_results(iter_num, losses_and_accs):
    """Print formatted evaluation results including per-class metrics if available"""
    print(f"\n{'='*80}")
    print(f"EVALUATION AT STEP {iter_num}")
    print(f"{'='*80}")
    print(f"Train Loss: {losses_and_accs['train']:.4f} | Val Loss: {losses_and_accs['val']:.4f}")
    print(f"Train Acc:  {losses_and_accs['train_accuracy']*100:.2f}% | Val Acc:  {losses_and_accs['val_accuracy']*100:.2f}%")
    
    if show_per_class_during_training and 'train_per_class' in losses_and_accs:
        train_per_class = losses_and_accs['train_per_class']
        val_per_class = losses_and_accs['val_per_class']
        
        if classifier_type == "bce" and train_per_class is not None and val_per_class is not None:
            train_total, train_recall, train_specificity = train_per_class
            val_total, val_recall, val_specificity = val_per_class
            
            print(f"\nDetailed Per-Class Metrics:")
            print(f"{'Class':<6} {'Train Total':<12} {'Val Total':<12} {'Train Recall':<14} {'Val Recall':<12} {'Train Spec.':<12} {'Val Spec.':<10}")
            print(f"{'-'*88}")
            
            class_names = ["W", "A", "S", "D"]
            for i in range(len(train_total)):
                class_name = class_names[i] if i < len(class_names) else f"C{i}"
                train_tot = f"{train_total[i]*100:.1f}%" if not np.isnan(train_total[i]) else "N/A"
                val_tot = f"{val_total[i]*100:.1f}%" if not np.isnan(val_total[i]) else "N/A"
                train_rec = f"{train_recall[i]*100:.1f}%" if not np.isnan(train_recall[i]) else "N/A"
                val_rec = f"{val_recall[i]*100:.1f}%" if not np.isnan(val_recall[i]) else "N/A"
                train_spec = f"{train_specificity[i]*100:.1f}%" if not np.isnan(train_specificity[i]) else "N/A"
                val_spec = f"{val_specificity[i]*100:.1f}%" if not np.isnan(val_specificity[i]) else "N/A"
                print(f"{class_name:<6} {train_tot:<12} {val_tot:<12} {train_rec:<14} {val_rec:<12} {train_spec:<12} {val_spec:<10}")
    
    print(f"{'='*80}\n")


# https://stackoverflow.com/questions/11352047/finding-moving-average-from-data-points-in-python
def moving_average(data, window_size):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    return (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size


def _extract_final_labels(labels, sequence_len):
    """Extract final labels from sequence if needed"""
    if sequence_len > 1:
        return labels[:, -1, :]
    return labels

def _compute_per_class_metrics(preds, labels):
    """Compute per-class accuracy metrics for BCE case"""
    num_classes = labels.shape[-1]
    
    # Total accuracy per class
    per_class_total_acc = np.mean(preds == labels, axis=0)
    
    # Recall: accuracy when label was true
    per_class_recall = np.array([
        np.mean(preds[labels[:, i], i]) if np.any(labels[:, i]) else np.nan
        for i in range(num_classes)
    ])
    
    # Specificity: accuracy when label was false
    per_class_specificity = np.array([
        np.mean(~preds[~labels[:, i], i]) if np.any(~labels[:, i]) else np.nan
        for i in range(num_classes)
    ])
    
    return per_class_total_acc, per_class_recall, per_class_specificity

@torch.no_grad()
def calc_accuracy(logits, labels, return_per_class=False):
    """Calculate accuracy metrics with optional per-class breakdown"""
    # Convert logits to predictions
    preds = logits.detach().cpu().float()
    
    if classifier_type == "bce":
        # Binary classification with sigmoid
        preds = torch.nn.functional.sigmoid(preds).numpy() >= 0.5
        labels = labels.detach().cpu().numpy().astype(bool)
        labels = _extract_final_labels(labels, sequence_len)
        
        # Overall exact match accuracy
        exact_match_acc = np.mean(np.all(preds == labels, axis=-1))
        
        if return_per_class:
            per_class_metrics = _compute_per_class_metrics(preds, labels)
            return exact_match_acc, per_class_metrics
        return exact_match_acc
        
    else:
        # Multi-class classification with argmax
        preds = torch.argmax(preds, dim=-1).numpy()
        labels = labels.detach().cpu().numpy()
        labels = _extract_final_labels(labels, sequence_len)
        labels = np.argmax(labels, axis=-1)
        
        exact_match_acc = np.mean(preds == labels)
        if return_per_class:
            return exact_match_acc, None  # CCE per-class metrics not implemented
        return exact_match_acc


def plot_metrics(metrics, window_size=50):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot losses
    ax1.plot(range(len(metrics["losses"])), metrics["losses"], label='Training Loss', color='#1f77b4',
             alpha=0.3)  # light blue
    smoothed_losses = moving_average(metrics["losses"], window_size)
    half_window = window_size // 2
    ax1.plot(range(half_window - 1, len(metrics["losses"]) - half_window), smoothed_losses,
             label='Smoothed Training Loss', color='#1f77b4')
    ax1.plot(metrics["val_loss_iters"], metrics["val_losses"], label='Validation Loss', color='#ff7f0e',
             linestyle='dashed', marker='o')  # light orange
    ax1.set_ylabel('Loss')
    ax1.yaxis.set_label_position("right")
    ax1.yaxis.tick_right()
    ax1.legend(loc='upper left')
    ax1.set_title('Training and Validation Losses')

    # Plot accuracies
    ax2.plot(range(len(metrics["accuracy"])), metrics["accuracy"], label='Training Accuracy (Train Augmentation)',
             color='#2ca02c', alpha=0.3)  # green
    smoothed_accuracies = moving_average(metrics["accuracy"], window_size)
    ax2.plot(range(half_window - 1, len(metrics["accuracy"]) - half_window), smoothed_accuracies,
             label='Smoothed Training Accuracy (Train Augmentation)', color='#2ca02c')

    ax2.plot(metrics["val_loss_iters"], metrics["train_accs"], label='Training Accuracy (Val Augmentation)',
             color='#9467bd', linestyle='dashed', marker='s')  # purple
    ax2.plot(metrics["val_loss_iters"], metrics["val_accs"], label='Validation Accuracy', color='#d62728',
             linestyle='dashed', marker='o')  # red

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Accuracy')
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.legend(loc='lower left')
    ax2.set_title('Training and Validation Accuracies')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, metrics_name))
    plt.close(fig)


def plot_class_metrics(metrics):
    """Plot per-class metrics in a 2x2 grid - one subplot for each class"""
    if not show_per_class_during_training or "train_class_metrics" not in metrics:
        return  # Skip if per-class metrics aren't available
    
    # Create figure with better spacing
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 13))
    axes = [ax1, ax2, ax3, ax4]
    class_names = ["W", "A", "S", "D"]
    
    # Better color palette with more distinct colors for 6 lines
    train_colors = {
        'total': '#1f77b4',      # Blue
        'recall': '#d62728',     # Red  
        'specificity': '#2ca02c'  # Green
    }
    
    val_colors = {
        'total': '#aec7e8',      # Light Blue
        'recall': '#ff9896',     # Light Red
        'specificity': '#98df8a'  # Light Green
    }
    
    # Different markers for each metric type
    markers = {
        'total': 'o',        # Circle
        'recall': 's',       # Square
        'specificity': '^'   # Triangle up
    }
    
    for i, (ax, class_name) in enumerate(zip(axes, class_names)):
        # Extract data for this class
        train_total = [data[0][i] if data[0] is not None else np.nan for data in metrics["train_class_metrics"]]
        train_recall = [data[1][i] if data[1] is not None else np.nan for data in metrics["train_class_metrics"]]
        train_spec = [data[2][i] if data[2] is not None else np.nan for data in metrics["train_class_metrics"]]
        
        val_total = [data[0][i] if data[0] is not None else np.nan for data in metrics["val_class_metrics"]]
        val_recall = [data[1][i] if data[1] is not None else np.nan for data in metrics["val_class_metrics"]]
        val_spec = [data[2][i] if data[2] is not None else np.nan for data in metrics["val_class_metrics"]]
        
        x_vals = metrics["val_loss_iters"]
        
        # Plot training metrics (solid lines, darker colors, larger markers)
        ax.plot(x_vals, train_total, label='Train Total', color=train_colors['total'], 
                linestyle='-', marker=markers['total'], markersize=6, linewidth=2.5, markeredgewidth=1, markeredgecolor='white')
        ax.plot(x_vals, train_recall, label='Train Recall', color=train_colors['recall'], 
                linestyle='-', marker=markers['recall'], markersize=6, linewidth=2.5, markeredgewidth=1, markeredgecolor='white')
        ax.plot(x_vals, train_spec, label='Train Specificity', color=train_colors['specificity'], 
                linestyle='-', marker=markers['specificity'], markersize=6, linewidth=2.5, markeredgewidth=1, markeredgecolor='white')
        
        # Plot validation metrics (dashed lines, lighter colors, full opacity)
        ax.plot(x_vals, val_total, label='Val Total', color=val_colors['total'], 
                linestyle='--', marker=markers['total'], markersize=5, linewidth=2)
        ax.plot(x_vals, val_recall, label='Val Recall', color=val_colors['recall'], 
                linestyle='--', marker=markers['recall'], markersize=5, linewidth=2)
        ax.plot(x_vals, val_spec, label='Val Specificity', color=val_colors['specificity'], 
                linestyle='--', marker=markers['specificity'], markersize=5, linewidth=2)
        
        # Improved styling
        ax.set_title(f'{class_name} Metrics', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        
        # Better legend positioning and styling
        legend = ax.legend(fontsize=10, loc='lower right', frameon=True, fancybox=True, 
                          shadow=True, framealpha=0.9, ncol=2)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('gray')
        
        # Improved grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Consistent y-axis with better ticks
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
        
        # Better tick styling
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Add subtle background color for each subplot
        ax.set_facecolor('#fafafa')
    
    # Better overall layout
    plt.tight_layout(pad=3.0)
    
    # Add overall title
    fig.suptitle('Per-Class Performance Metrics', fontsize=16, fontweight='bold', y=0.98)
    
    # Save with higher DPI for better quality
    base_name = metrics_name.split('.')[0] 
    ext = metrics_name.split('.')[-1]
    class_metrics_name = f"{base_name}_class_acc.{ext}"
    plt.savefig(os.path.join(out_dir, class_metrics_name), dpi=300, bbox_inches='tight')
    plt.close(fig)


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def get_pos_weights(it):
    # pos_weights scheduler (cosine annealing from no bias to full data-driven weights)
    if not schedule_pos_weights:
        return torch.tensor(end_pos_weights)
    
    start_weights = torch.tensor(start_pos_weights)
    end_weights = torch.tensor(end_pos_weights)
    
    # 1) linear warmup for pos_warmup_iters steps (keep at start_weights)
    if it < pos_warmup_iters:
        return start_weights
    # 2) if it > pos_decay_iters, return end weights
    if it > pos_decay_iters:
        return end_weights
    # 3) in between, use cosine annealing from start to end weights
    decay_ratio = (it - pos_warmup_iters) / (pos_decay_iters - pos_warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 - math.cos(math.pi * decay_ratio))  # Note: 1 - cos for increasing
    return start_weights + coeff * (end_weights - start_weights)


def get_batch(dataloader_iter, split):
    try:
        x, y = next(dataloader_iter)
    except StopIteration:
        dataloader_iter = iter(train_dataloader if split == 'train' else val_dataloader)
        x, y = next(dataloader_iter)
    y_cpu = y
    y_cpu = y_cpu.detach().cpu()    # not necessary, but to make sure
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y, y_cpu, dataloader_iter


def _update_learning_rate(current_lr, iter_num, optimizer):
    """Update learning rate only when it changes"""
    if decay_lr:
        new_lr = get_lr(iter_num)
        if new_lr != current_lr:
            current_lr = new_lr     # for returning
            # Apply group-specific LRs using use_muon flag
            ratio = new_lr / learning_rate if learning_rate > 0 else 1.0
            for param_group in optimizer.param_groups:
                if param_group.get('use_muon', False):
                    param_group['lr'] = muon_learning_rate * ratio
                else:
                    param_group['lr'] = new_lr
    elif current_lr != learning_rate:
        current_lr = learning_rate
        for param_group in optimizer.param_groups:
            if param_group.get('use_muon', False):
                param_group['lr'] = muon_learning_rate
            else:
                param_group['lr'] = learning_rate
    return current_lr

def _update_pos_weights(model, iter_num):
    """Update pos_weights in the model's loss function"""
    if schedule_pos_weights:
        current_pos_weights = get_pos_weights(iter_num).to(device)
        model.loss_fct.pos_weight = current_pos_weights
        return current_pos_weights
    return None

def _collect_evaluation_metrics(losses_and_accs, metrics_dict, local_iter_num):
    """Collect and store evaluation metrics efficiently"""
    metrics_dict["val_losses"].append(losses_and_accs["val"])
    metrics_dict["val_loss_iters"].append(local_iter_num)
    metrics_dict["val_accs"].append(losses_and_accs["val_accuracy"])
    metrics_dict["train_accs"].append(losses_and_accs["train_accuracy"])
    
    # Efficient per-class metrics collection
    train_per_class = losses_and_accs.get("train_per_class")
    val_per_class = losses_and_accs.get("val_per_class")
    metrics_dict["train_class_metrics"].append(train_per_class)
    metrics_dict["val_class_metrics"].append(val_per_class)

def train_loop():
    global iter_num, best_val_loss, num_workers
    num_workers = 4
    t0 = time.time()
    init_dataloaders()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    metrics_dict = {"losses": [], "val_losses": [], "val_loss_iters": [], "accuracy": [], "val_accs": [], "train_accs": [], "train_class_metrics": [], "val_class_metrics": []}
    print("Total train samples:", len(train_dataloader.dataset))
    print("Total val samples:", len(val_dataloader.dataset))
    model = load_model()
    dataloader_iter = iter(train_dataloader)
    X, Y, Y_CPU, dataloader_iter = get_batch(dataloader_iter, 'train')
    # Cache learning rate to avoid recalculation
    current_lr = None
    
    while True:
        # Update learning rate efficiently
        current_lr = _update_learning_rate(current_lr, iter_num, optimizer)
        # Update pos_weights efficiently
        current_pos_weights = _update_pos_weights(model, iter_num)

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and iter_num != iter_num_on_load and iter_num != 0:
            losses_and_accs = estimate_loss(model)
            
            # Collect metrics efficiently
            _collect_evaluation_metrics(losses_and_accs, metrics_dict, local_iter_num)
            
            # Use the separated print function
            print_evaluation_results(iter_num, losses_and_accs)
            
            if losses_and_accs['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses_and_accs['val'] if losses_and_accs['val'] < best_val_loss else best_val_loss
                if iter_num > 0:
                    # torch.compile prepends stuff to the name!!! (_orig_mod.), so can't use startswith, but need to use "in"
                    state_dict = model.state_dict() if fine_tune else {k: v for k, v in model.state_dict().items() if ("dinov2" not in k and "dinov3" not in k)}
                    checkpoint = {
                        'model': state_dict,
                        'optimizer': optimizer.state_dict(),
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config_dict,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, save_checkpoint_name))
                    if best_val_loss == losses_and_accs['val']:
                        best_ckpt_name = save_checkpoint_name.split(".")[0] + "-best." + save_checkpoint_name.split(".")[1]
                        torch.save(checkpoint, os.path.join(out_dir, best_ckpt_name))
                    del state_dict, checkpoint
            plot_metrics(metrics_dict)
            plot_class_metrics(metrics_dict)  # Plot per-class metrics alongside main plots
        if iter_num == 0 and eval_only:
            break

        accumulated_loss = 0.0
        # Accumulate for batch accuracy computation (more efficient than per-step)
        if gradient_accumulation_steps == 1:
            # Optimized path for no gradient accumulation
            with ctx:
                logits, loss = model(X, labels=Y)
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y, next_y_cpu, dataloader_iter = get_batch(dataloader_iter, 'train')
            accumulated_loss = loss.item()
            accuracy = calc_accuracy(logits, Y_CPU)
            Y_CPU = next_y_cpu
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        else:
            # Standard gradient accumulation path
            train_logits = []
            train_labels = []
            for micro_step in range(gradient_accumulation_steps):
                with ctx:
                    logits, loss = model(X, labels=Y)
                    loss = loss / gradient_accumulation_steps   # scale the loss to account for gradient accumulation
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X, Y, next_y_cpu, dataloader_iter = get_batch(dataloader_iter, 'train')
                accumulated_loss += loss.item()
                # Accumulate for batch accuracy calculation
                train_logits.append(logits.detach().cpu())
                train_labels.append(Y_CPU)
                Y_CPU = next_y_cpu
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()
            
            # Batch compute accuracy for gradient accumulation case
            if train_logits:
                stacked_logits = torch.cat(train_logits, dim=0)
                stacked_labels = torch.cat(train_labels, dim=0)
                accuracy = calc_accuracy(stacked_logits, stacked_labels)
                del train_logits, train_labels, stacked_logits, stacked_labels
            else:
                accuracy = 0.0
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)
        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        metrics_dict["losses"].append(accumulated_loss)
        metrics_dict["accuracy"].append(accuracy)
        if iter_num % log_interval == 0:
            # Format pos_weights for printing
            pos_weights_str = ""
            if current_pos_weights is not None:
                pw = current_pos_weights.cpu().numpy()
                pos_weights_str = f", pos_weights: [{pw[0]:.3f}, {pw[1]:.3f}, {pw[2]:.3f}, {pw[3]:.3f}]"
            
            print(f"iter {iter_num}: loss {accumulated_loss:.4f}, time {dt*1000:.2f}ms, accuracy {accuracy*100:.3f}%, lr {current_lr:.6f}{pos_weights_str}")
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            break


@torch.no_grad()
def comprehensive_eval(checkpoint_path):
    """Full dataset evaluation - like estimate_loss but for entire val set"""
    print(f"Running comprehensive evaluation on {checkpoint_path}...")
    
    # Load model using existing infrastructure
    global model, num_workers
    num_workers = 2
    old_init_from = globals()['init_from']
    old_load_name = globals()['load_checkpoint_name']
    
    globals()['init_from'] = 'resume' 
    globals()['load_checkpoint_name'] = checkpoint_path
    model = load_model(sample_only=True)
    globals()['init_from'] = old_init_from
    globals()['load_checkpoint_name'] = old_load_name
    
    if val_dataloader is None:
        init_dataloaders()
    
    model.eval()
    
    total_batches = len(val_dataloader)
    dataloader_iter = iter(val_dataloader)
    X, Y, Y_CPU, dataloader_iter = get_batch(dataloader_iter, 'val')
    
    # Just collect raw data - no processing
    losses = torch.zeros(total_batches)
    all_logits = []
    all_labels = []
    
    next_y_cpu = None
    for k in tqdm(range(total_batches)):
        with ctx:
            logits, loss = model(X, labels=Y)
        
        # Async prefetch next batch
        if k < total_batches - 1:
            X, Y, next_y_cpu, dataloader_iter = get_batch(dataloader_iter, 'val')
            
        losses[k] = loss.cpu().item()
        
        # Just collect raw detached tensors
        all_logits.append(logits.detach().cpu())
        all_labels.append(Y_CPU)
                
        Y_CPU = next_y_cpu
    
    # Stack everything and call calc_accuracy once
    stacked_logits = torch.cat(all_logits, dim=0)
    stacked_labels = torch.cat(all_labels, dim=0)
    
    avg_loss = losses.mean().item()
    overall_accuracy, per_class_metrics = calc_accuracy(stacked_logits, stacked_labels, return_per_class=True)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"Dataset: {len(val_dataloader.dataset)} samples")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Overall Exact Match Accuracy: {overall_accuracy*100:.2f}%")
    
    if classifier_type == "bce" and per_class_metrics is not None:
        total_acc, true_acc, false_acc = per_class_metrics
        print(f"\nPer-Class Detailed Accuracy:")
        print(f"{'Class':<8} {'Total':<8} {'When=1':<8} {'When=0':<8}")
        print(f"{'-'*35}")
        
        class_names = ["W", "A", "S", "D"]
        for i in range(len(total_acc)):
            class_name = class_names[i] if i < len(class_names) else f"C{i}"
            print(f"{class_name:<8} {total_acc[i]*100:>6.1f}% {true_acc[i]*100:>7.1f}% {false_acc[i]*100:>7.1f}%")
    
    print(f"{'='*80}")
    
    # Cleanup
    del X, Y, Y_CPU, next_y_cpu, dataloader_iter, loss, all_logits, all_labels
    torch.cuda.empty_cache()


if __name__ == "__main__":
    train_loop()
    # comprehensive_eval("v3_base_full_balanced-best.pt")
