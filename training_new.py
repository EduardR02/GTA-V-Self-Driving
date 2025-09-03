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
out_dir = os.path.join('models', 'transformer')
eval_interval = 500
log_interval = 1
eval_iters = 10
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
fine_tune = False   # train the entire model or just the top
freeze_non_dino_layers = False
init_from = 'scratch' # 'scratch' or 'resume'
use_dinov3 = True  # Toggle: True for DINOv3, False for DINOv2
dino_size = "large"  # small is ~21M, base ~86M, large ~300M, giant ~1.1B
use_dino_registers = True
load_checkpoint_name = "optim_48_512_35k.pt"
save_checkpoint_name = "test.pt"
metrics_name = "test.png"
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
batch_size = 128    # if gradient_accumulation_steps > 1, this is the micro-batch size
train_split = 0.95   # test val split, keep same for resume
convert_to_greyscale = False
sequence_len = 3
sequence_stride = 20
flip_prob = 0.25
warp_prob = 0.2
zoom_prob = 0.3
dropout_p = 0.2     # 0 to disable
classifier_type = "bce" # "cce" or "bce"
restart_schedules = False
cls_option = "both"    # "cls_only", "both", or "patches_only"
shift_labels = True

# adamw optimizer
learning_rate = 2e-4 # max learning rate
max_iters = 100000 # total number of training iterations
# optimizer settings
weight_decay = 0.05
beta1 = 0.9
beta2 = 0.999
grad_clip = 0.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 400 # how many steps to warm up for
lr_decay_iters = 25000 # should be ~= max_iters per Chinchilla
min_lr = 5e-6 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
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
        if use_dinov3:
            model = Dinov3ForTimeSeriesClassification(dino_size, len(id2label), expected_input_hw=(192, 240), dropout_rate=dropout_p)
        else:
            model = Dinov2ForTimeSeriesClassification(dino_size, len(id2label), classifier_type=classifier_type, cls_option=cls_option, use_reg=use_dino_registers, dropout_rate=dropout_p)
    elif init_from == 'resume':
        print(f"Resuming training from {out_dir}")
        ckpt_path = os.path.join(out_dir, load_checkpoint_name)
        checkpoint = torch.load(ckpt_path, map_location=device)
        if use_dinov3:
            model = Dinov3ForTimeSeriesClassification(dino_size, len(id2label), expected_input_hw=(192, 240), dropout_rate=dropout_p)
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
            print("compiling the model... (takes a ~minute)")
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
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
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
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)    # requires PyTorch 2.0

    return model


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        dataloader_iter = iter(train_dataloader if split == 'train' else val_dataloader)
        X, Y, Y_CPU, dataloader_iter = get_batch(dataloader_iter, split)
        losses = torch.zeros(eval_iters * gradient_accumulation_steps)
        accuracies = torch.zeros(eval_iters * gradient_accumulation_steps)
        Y_CPU_NEW = None
        for k in range(eval_iters * gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, labels=Y)
            # async prefetch next batch while model is doing the forward pass on the GPU
            if k < eval_iters * gradient_accumulation_steps - 1:
                X, Y, Y_CPU_NEW, dataloader_iter = get_batch(dataloader_iter, split)
            losses[k] = loss.cpu().item()
            accuracies[k] = calc_accuracy(logits, Y_CPU)
            Y_CPU = Y_CPU_NEW
        out[split] = losses.mean()
        out[split + "_accuracy"] = accuracies.mean()
    del X, Y, Y_CPU, Y_CPU_NEW, dataloader_iter, loss
    torch.cuda.empty_cache()
    model.train()
    return out


# https://stackoverflow.com/questions/11352047/finding-moving-average-from-data-points-in-python
def moving_average(data, window_size):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    return (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size


@torch.no_grad()
def calc_accuracy(logits, labels):
    # Convert logits and labels to numpy arrays
    preds = logits.detach().cpu().float()
    if classifier_type == "bce":
        preds = torch.nn.functional.sigmoid(preds)
        preds = (preds.numpy() >= 0.5)
        labels = labels.detach().cpu().numpy().astype(bool)
        if sequence_len > 1:
            labels = labels[:, -1, :]   # get the last label in the sequence
        return np.mean(np.all(preds == labels, axis=-1))
    else:
        preds = torch.argmax(preds, dim=-1).numpy()
        labels = labels.detach().cpu().numpy()
        if sequence_len > 1:
            labels = labels[:, -1, :]
        labels = np.argmax(labels, axis=-1)
        return np.mean(preds == labels)


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


def train_loop():
    global iter_num, best_val_loss
    t0 = time.time()
    init_dataloaders()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    metrics_dict = {"losses": [], "val_losses": [], "val_loss_iters": [], "accuracy": [], "val_accs": [], "train_accs": []}
    print("Total train samples:", len(train_dataloader.dataset))
    print("Total val samples:", len(val_dataloader.dataset))
    model = load_model()
    dataloader_iter = iter(train_dataloader)
    X, Y, Y_CPU, dataloader_iter = get_batch(dataloader_iter, 'train')
    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and iter_num != iter_num_on_load and iter_num != 0:
            losses_and_accs = estimate_loss(model)
            metrics_dict["val_losses"].append(losses_and_accs["val"])
            metrics_dict["val_loss_iters"].append(local_iter_num)
            metrics_dict["val_accs"].append(losses_and_accs["val_accuracy"])
            metrics_dict["train_accs"].append(losses_and_accs["train_accuracy"])
            print(f"step {iter_num}: train loss {losses_and_accs['train']:.4f}, val loss {losses_and_accs['val']:.4f}, train acc {losses_and_accs['train_accuracy']*100:.3f}, val acc {losses_and_accs['val_accuracy']*100:.3f}")
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
        if iter_num == 0 and eval_only:
            break

        lossf = 0
        accuracy = 0
        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, labels=Y)
                loss = loss / gradient_accumulation_steps   # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y, Y_CPU_NEW, dataloader_iter = get_batch(dataloader_iter, 'train')
            lossf += loss.item()
            accuracy += calc_accuracy(logits, Y_CPU)
            Y_CPU = Y_CPU_NEW
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        accuracy = accuracy / gradient_accumulation_steps
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
        metrics_dict["losses"].append(lossf)
        metrics_dict["accuracy"].append(accuracy)
        if iter_num % log_interval == 0:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            # lossf = loss.item() * gradient_accumulation_steps

            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, accuracy {accuracy*100:.3f}%, lr {lr:.6f}")
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            break


if __name__ == "__main__":
    train_loop()
