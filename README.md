# GTA V Self-Driving


## Demo
- TODO: add driving videos here, one without cars on the road, one with, 
and one with some interesting moments like getting unstuck, hard turns, etc.

## Model
A frozen **DINOv3** ViT backbone feeds patch tokens through a linear projection down to a smaller embedding dimension, 
followed by a custom **Transformer** temporal aggregator with **Rotary Positional Embeddings (RoPE)** and **SwiGLU** MLPs, 
and a final linear classification head for the four output classes (W, A, S, D).
BCE Loss is used so that each class is predicted independently.

### Architecture details
- **Backbone**: DINOv3 ViT (16x16 patches), frozen. Configurable size (base/large/huge). Loaded via `torch.hub` from a local clone or HuggingFace Transformers.
- **Projection**: Linear layer from DINOv3 embeddings down to a smaller transformer dimension.
- **Temporal aggregator**: A small Transformer with RMSNorm, RoPE, SwiGLU MLP, and a learnable CLS token prepended to the full sequence of patch tokens from all 3 frames. The last transformer layer only computes attention for the CLS token as an optimization, since only the CLS representation is used for classification. Uses Flash Attention or xFormers when available, falls back to PyTorch SDPA. The exact depth, width, and head count are still being experimented with — smaller aggregators don't overfit but seem to lose something, while larger ones overfit. It's unclear what works best here.
- **Classification head**: Single linear layer from the CLS token to 4 outputs.
- **Input**: 3 frames. Resolution doesn't really matter (padding handles patch alignment), the current 240x180 is just what the training data was originally recorded at.

## Evolution from LSTM to Transformer

The model started as a frozen **DINOv2** backbone (base, 14x14 patches) with some CNN layers on top, followed by an **LSTM** and a linear head. 
This worked, but the LSTM's sequential nature limited what the model could learn from the frame sequence.

The current version replaces all of that post-backbone processing with a Transformer aggregator. This has several advantages:
- The Transformer sees all patch tokens from all frames simultaneously, rather than getting a single pooled vector per frame.
- RoPE gives smooth positional awareness across the full token sequence without learned position embeddings.
- SwiGLU and RMSNorm are more modern and stable than the original LayerNorm + GELU setup.
- The CLS-only attention optimization in the last layer saves compute without hurting quality.

The backbone was also upgraded from DINOv2 (14x14 patches) to DINOv3 (16x16 patches), which slightly changes the input resolution requirements but gives better features.

## Training

- **Optimizer**: [Muon](https://github.com/KellerJordan/Muon) (`SingleDeviceMuonWithAuxAdam`) for the transformer aggregator weights, AdamW for the backbone, projection, head, and all biases/norms.
- **Scheduling**: Cosine decay with linear warmup. Positive class weights for BCE loss are scheduled during training to handle class imbalance (steering and braking are much rarer than accelerating).
- **Mixed precision**: bfloat16 (preferred) or float16 with GradScaler.
- **Gradient accumulation**: Used to simulate larger effective batch sizes.
- **`torch.compile`**: Enabled for speed.

## RNN vs no RNN (historical)
The issue with not having some kind of sequence as input is that the model cannot judge its speed and current steering.
This makes the forward vs backward distinction button almost noise, and because the training data has MUCH more forward, 
the backward examples are essentially noise. Another problem that this model has is that it cannot regulate its speed, 
so when it drives in the city it might simply adjust its confidence in the forward prediction overall based on the 
"scenery", instead of its speed (because it would have seen many "no press" examples in scenery matching the city).

Using an RNN (or now a Transformer) fixes most of this. The Transformer is strictly better here because it can attend 
to all patches across all frames simultaneously, rather than receiving a pooled summary per frame.

## Proportional outputs during inference
This works incredibly well, and I am sad that I did not do this earlier. By simply taking the steering delta,
meaning the difference between the output to steer left and the output to steer right, and using this to determine 
the total steering until the next prediction. Being able to do this is why using BCE loss is such a big advantage, 
because we predict each class independently. 
We can simply do "time it took to predict" * "steering delta" to get the button press duration, as we can assume that 
the prediction speed is mostly constant, so we can use it to approximate the time until the next prediction.
Using this method the steering becomes extremely smooth, and even though the training data driving is "crazy", 
the model averages it out and actually **lane keeps** quite well, in gta... crazy. This also makes overtaking and 
general turns much smoother, confident, and significantly reduces crashes.

This unfortunately does not work that well for forward and backward, the reason being that frequently "tapping" the 
accelerator will make the car disproportionately slower than holding it down. If we simply use the models prediction 
to determine the length of the press, it will almost never do a full press because the confidence is never exactly 1. 
You could of course do some gradual and more complicated thresholding, but instead we just do the most basic thing 
by using a single threshold value for forward and backward, which thresholds the output to 0 or 1, so that if the output 
becomes one, we can do a full press.

## Sequence length
For the model to be able to understand its speed, you need at least a sequence length of 2. However this causes another
problem: The model gets stuck a lot, and not from crashing, but simply from staying on a random place on the map, 
in the middle of the road. The reason is that when collecting the training data, you will want to coast in some places. 
When the model sees this in the training data, it will memorize these places as "no input here", and when it reaches 
them and has low speed, it will stop from friction and not be able to start again, because it memorized "no input".

To fix this we increase the sequence length to: drumroll... 3. This way the model also has access to it's acceleration.
Now it can understand that when it starts decelerating, but its speed is already slow, it should predict forward, and
vice versa. This understanding generally makes all situations in the training data "make sense", like examples of 
getting "unstuck", turning, overtaking, breaking to not crash, etc.

## Sequence stride
I did not test this much, but it makes sense to pick a sequence stride that you can closely match during inference based 
on time passed. I chose 20 frames, which corresponds to about 0.25 seconds (during training,
during inference this is about 2 - 3 frames).

## Augmentations
There is a slight difference with having images from this game than other random images. This is that we have a minimap,
which tells us where to go, and is always in the same place. So for distorting or flip augmentations, we need to inpaint 
the area of the minimap, do the augmentation, and then put the minimap back in its original place and size. This also 
makes it harder for the model to "cheat" by looking at the minimap becoming "weird", and not adapting to those examples.

Current augmentations (via Albumentations):
- **Diagonal warp**: Simulates lateral drift/recovery and updates labels to "steer back".
- **Minimap protection**: Extract minimap, inpaint the area, apply spatial transforms, re-insert minimap.
- **Color jitter**, **random sky mask** (blacks out top of image), **horizontal flip** (with label swapping A↔D), and **zoom**.

## Shifting labels
It makes sense to account for the time it takes for the model to predict the output at inference time during training.
First of all, not doing this makes it so the models predictions are always behind by that time,
and secondly this will make the models input during inference look more out of distribution, because 
it creates its own past during inference, which will be driven "with lag" and look different from the training data.

Labels are shifted by `round(fps_at_recording / fps_at_test)` frames.

## Using only CLS token
It is hard to describe, but when I tried it just did not work. When driving, the model looks blind.
It kind of stays on the road, then suddenly it looks like it lost vision and starts turning into a barrier.
This looks very weird compared to the model that uses all patches. Even when the loss goes down as expected,
even though it is somewhat higher than when using all patches, the val loss and accuracy are similar. But the driving
performance is just terrible. This looks useless even compared to the earlier checkpoints of the patch model.

## Training data used
Stats:
- resolution: 240x180, downsized from 800x600 game resolution
- about 428k "regular" driving frames of going fast and overtaking, driven on the highway, in the city, and in the outskirts.
- about 27k frames of "getting unstuck", which consisted of starting in a stuck position and usually reversing
- about 35k frames of "difficult" turns on small roads and in the city, also trying to make the model follow the minimap waypoint.
- all data was driven
    - at 80fps
    - with a waypoint on the minimap
    - with hood camera (and invisible car so that the hood does not block the lower part of the screen)
    - sunny weather
- data is stored in HDF5 (`.h5`) files containing `images` (uint8) and `labels` (float32)

Unfortunately at the time of recording the data I did not record each button press separately, but predetermined classes, as follows, of which only a single class could be true at once:
- w, a, s, d, wa, wd, no press  

This means that to use them for training now, I had to first convert these (imperfect) recordings into:
- w, a, s, d

Where multiple, or no classes could be true at once.

## Getting unstuck data
To teach the model how to get unstuck, I collected some data of for example starting out in front of an (indestructible) telephone pole, reversing, then returning to the road.  
When recording this data, I waited at the start to "pad" for sequence length, so that the first label would be pressing backward already, while being positioned right in front of the blocking object.  
This causes an issue when training, because the "padding" data is junk, therefore you want to skip it (in case you shorten your sequence length or train without sequences). So my pytorch dataset and dataloaders are able to account for this type of data. Just make sure if you create this type of data you either keep the padding consistent, or record the exact padding amount for each example.

## Dataset and Dataloader
The dataset and dataloader can account for:
- shifting labels
- stuck type data
- special augmentations (with the minimap)
- converting labels from old format to w, a, s, d
- both: sampling sequences and single image-label pairs
- training and validation split
- multiple data directories

## Inference
- Screen capture via `mss` at 800x600 from a fixed monitor region.
- A `deque` buffer stores timestamped frames; `select_images` picks frames closest to the training stride to match temporal spacing.
- **Steering (A/D)**: Proportional — key press duration scales with the model's confidence within the current frame time. If both A and D are predicted, only the stronger one is kept.
- **Speed (W/S)**: Binary — on/off based on a threshold.

## Tech Stack
| Component | Technology |
|-----------|-----------|
| Backbone | DINOv3 ViT (16x16 patches), frozen |
| Temporal aggregator | Custom Transformer with RoPE + SwiGLU |
| Attention | Flash Attention / xFormers / PyTorch SDPA |
| Optimizer | Muon + AdamW |
| Mixed precision | bfloat16 |
| Data format | HDF5 |
| Augmentations | Albumentations |
| Screen capture | mss |
| Framework | PyTorch |

## Usage
- Install dependencies: `pip install -r requirements.txt` (also needs `torch` and optionally `flash-attn` / `xformers`)
- For trying it out, use the provided checkpoint, open the game in 800x600 resolution, and place it at the top left corner of the screen.
- TODO: add simply detecting the window instead of careful placement
- Use hood camera, make your car invisible, set sunny weather
- Add a waypoint where you want the model to drive, and run main.py
- Try out with cars, without cars, in the city, on the highway, etc.

## Train yourself
This code is very multi purpose, as long as you want to predict some class or sequence. In case your data is different you will obviously need to modify the dataloaders. If you want to collect your own driving data, you can use the provided data collection script (not updated so don't know if it works). I would recommend increasing the resolution (if your gpu is big or you don't mind waiting), and obviously also recording every button press...  
Make sure that each time you start recording you use a new file, as you need to know when a sequence starts and ends.

## Contribute
If you have any ideas, improvements, or want to contribute in any way, feel free to open an issue or a pull request.
