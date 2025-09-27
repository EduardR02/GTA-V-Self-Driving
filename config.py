import os

width = 240
height = 180
sequence_len = 3
sequence_stride = 20     # Period between successive output sequences
fps_at_recording_time = 80      # check by using main with fps only set to true, while having the game running
fps_at_test_time = 34    # check by running model in main, while game is running!
use_dinov3 = True
monitor = {'top': 27, 'left': 0, 'width': 800, 'height': 600}
outputs = {"w": 0, "a": 1, "s": 2, "d": 3, "wa": 4, "wd": 5, "nothing": 6}
outputs_base = {"w": 0, "a": 1, "s": 2, "d": 3, "nothing": 4}
amt_remove_after_pause = 300

data_dir_name = "data"
model_dir_name = "models"
lstm_dir = os.path.join(model_dir_name, "lstm")
cnn_dir = os.path.join(model_dir_name, "cnn")
ckpt_name_cnn = "ckpt.pt"
ckpt_name_rnn = "ckpt.pt"
new_data_dir_name = os.path.join(data_dir_name, "new_data")
turns_data_dir_name = os.path.join(data_dir_name, "turns")
stuck_data_dir_name = os.path.join(data_dir_name, "stuck")
data_name = f"{width}x{height}_rgb"
