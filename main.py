from collections import deque
import numpy as np
import mss
import cv2
import time
from keys import PressKey, ReleaseKey, W, A, S, D
import config
from grabkeys import key_check
from threading import Thread as Worker
from training_new import load_model, ctx
from dataloader import transform
import torch
import psutil
import os


# Get the current process
process = psutil.Process(os.getpid())
# Set the priority to "High Priority" class
process.nice(psutil.HIGH_PRIORITY_CLASS)


# 0 - 1. Button presses can only be decided each time the model predicts, this is the max fraction of the full duration
# a turn button will be pressed until the next model prediction. This helps the model to not wiggle from oversteering,
# which is an "artifact" of low fps at inference time, but high fps when creating data. Meaning a turn key pressed when
# making the data at higher fps is much less impactful than when the model predicts at a lower fps, because
# it's prediction lasts for much longer.
max_steer = 1.0
output_dict = {"w": 0, "a": 1, "s": 2, "d": 3}


def press_and_release(key, t_time):
    PressKey(key)
    time.sleep(t_time)
    ReleaseKey(key)


def release_all():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)


# this works so well that I deleted everything else, the other ones should legit not be used, this is that good
def proportional_output_key(prediction, t_since_last_press):
    """
    Using proportional values for the steering (left, right) works insanely well.
    Gas and break not bad for reasons (tapping makes the car slow)...
    So instead we make them binary because that works really well.
    """
    press_keys = True
    log_presses = True
    min_val_steer = 0.05  # 0 - 1 (0.01 - 0.05 seems best)       0.05 seems to be best for the best models, use 0.01 for more overfit models trained for very long with high batch size (512+) to cut out noise
    speed_threshold = 0.1  # 0 - 1
    prediction = prediction.float().numpy().squeeze()   # float in case bf16 selected
    # do this before thresholding to get the true values
    prediction = handle_opposite_keys(prediction, output_dict)

    press_durations = np.where(prediction < min_val_steer, 0, prediction)
    # Binary thresholding for speed keys (W, S): either 0 or 1
    press_durations[[output_dict['w'], output_dict['s']]] = np.where(
        prediction[[output_dict['w'], output_dict['s']]] >= speed_threshold, 1, 0)
    # don't care about speed key values because we just want to press or not press
    press_durations *= t_since_last_press
    if press_keys:
        # Iterate over the keys and their corresponding durations
        for key, duration in zip([W, A, S, D], press_durations):
            if duration > 0:
                if key in [A, D]:  # Steering keys
                    max_steer_duration = t_since_last_press * max_steer
                    steer_duration = float(min(duration, max_steer_duration))
                    worker = Worker(target=press_and_release, args=(key, steer_duration))
                else:  # Speed keys
                    worker = Worker(target=PressKey, args=(key,))
            else:
                worker = Worker(target=ReleaseKey, args=(key,))
            worker.start()
    if log_presses:
        if np.sum(press_durations) == 0:
            print("Nothing pressed, max val was:", prediction.max())
        else:
            pressed_keys = [key + "-" + str(prediction[value]) for key, value in output_dict.items() if press_durations[value] > 0]
            print("Keys pressed:", ", ".join(pressed_keys))


def handle_opposite_keys(prediction, output_dict):
    if prediction[output_dict["a"]] > prediction[output_dict["d"]]:
        prediction[output_dict["a"]] -= prediction[output_dict["d"]]
        prediction[output_dict["d"]] = 0
    else:
        prediction[output_dict["d"]] -= prediction[output_dict["a"]]
        prediction[output_dict["a"]] = 0
    return prediction


# noinspection PyTypeChecker
def show_screen():
    sct = mss.mss()
    monitor = config.monitor
    while 1:
        img = np.asarray(sct.grab(monitor))  # output is bgra
        """img = cv2.resize(img, (int(config.width), int(config.height)))
        img = cv2.resize(img, (800, 600))"""
        # img = get_screencap_img(sct)
        # img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        # no need to convert color as cv2 displays bgr as rgb, capture is in bgr
        cv2.imshow(f'Car View {config.width}x{config.height}', img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


@torch.no_grad()
def main_with_cnn():
    model = load_model(sample_only=True)
    sct = mss.mss()
    counter = 0
    t = time.time()
    t_since_last_press = time.time()
    while True:
        counter += 1
        img = get_screencap_img(sct)
        img = transform(image=img)["image"]
        img = img[None, ...]
        img = img.pin_memory().to("cuda")

        prediction, _ = model(img)
        prediction = torch.nn.functional.sigmoid(prediction)

        proportional_output_key(prediction, t_since_last_press)
        t_since_last_press = time.time()
        key = key_check()
        if (time.time() - t) >= 1:
            print("fps:", counter)
            counter = 0
            t = time.time()
        if "T" in key:
            release_all()
            break
        if "N" in key:
            release_all()
            time.sleep(5)


@torch.no_grad()
def main_with_lstm():
    model = load_model(sample_only=True)
    sct = mss.mss()
    t = time.time()
    last_press_time = time.time()
    counter = 0
    # Use a deque to store timestamped images
    max_stored_images = 20  # Adjust this value based on your memory constraints
    image_buffer = deque(maxlen=max_stored_images)

    desired_interval = config.sequence_stride / config.fps_at_recording_time
    while True:
        counter += 1
        current_time = time.time()
        img = get_screencap_img(sct)

        image_buffer.append((current_time, img))    # timestamp for selecting images with best stride
        selected_images = select_images(image_buffer, desired_interval, config.sequence_len)
        # display_selected_images(selected_images)

        img_tensor = torch.stack([transform(image=img)["image"] for img in selected_images], dim=0)   # first timedim, then batch
        img_tensor = img_tensor[None, ...]  # add dummy batch dim
        img_tensor = img_tensor.pin_memory().to("cuda")
        with ctx:
            prediction, _ = model(img_tensor)
            prediction = torch.nn.functional.sigmoid(prediction.cpu())
        proportional_output_key(prediction, time.time() - last_press_time)
        last_press_time = time.time()
        key = key_check()

        if (time.time() - t) >= 1:
            print("fps:", counter)
            counter = 0
            t = time.time()
        if "T" in key:
            release_all()
            break
        if "N" in key:
            image_buffer.clear()
            release_all()
            time.sleep(5)


def display_selected_images(selected_images):
    """
    Display the selected images side by side in a non-blocking window.
    The newest image is on the left, oldest on the right.
    
    Args:
        selected_images: List of images to display
    """
    if not hasattr(display_selected_images, "window_created"):
        # Create window only once
        cv2.namedWindow("Selected Images", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Selected Images", 1200, 300)  # Adjust size as needed
        display_selected_images.window_created = True
    
    if not selected_images:
        return
        
    # Create a single wide image to display all frames side by side
    vis_height = selected_images[0].shape[0]
    vis_width = selected_images[0].shape[1] * len(selected_images)
    visualization = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
    
    # Reverse the order for display (newest on left, oldest on right)
    reversed_images = selected_images[::-1]
    
    for i, img in enumerate(reversed_images):
        x_offset = i * img.shape[1]
        # Convert from RGB to BGR for proper display with OpenCV
        display_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        visualization[:, x_offset:x_offset+img.shape[1], :] = display_img
        
        # Labels for reversed order - newest is first, oldest is last
        label = "Newest" if i == 0 else "Oldest" if i == len(reversed_images)-1 else f"t-{i}"
        cv2.putText(visualization, label, (x_offset + 10, vis_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display the visualization
    cv2.imshow("Selected Images", visualization)
    cv2.waitKey(1)  # Update the window (non-blocking)


def select_images(image_buffer, desired_interval, sequence_len):
    if len(image_buffer) < 2:
        return [img for _, img in image_buffer]

    current_time = image_buffer[-1][0]
    selected_images = [image_buffer[-1][1]]  # Start with the most recent image

    # Create a copy of the buffer to avoid modifying the original
    available_images = list(image_buffer)[:-1]  # Exclude the most recent image

    for i in range(1, sequence_len):
        if not available_images:
            break

        target_time = current_time - i * desired_interval
        best_image_index = min(range(len(available_images)),
                               key=lambda i: abs(available_images[i][0] - target_time))

        selected_images.append(available_images[best_image_index][1])

        # Remove the selected image and all newer images
        available_images = available_images[:best_image_index]

    # Pad with the oldest selected image if we don't have enough
    while len(selected_images) < sequence_len:
        selected_images.append(selected_images[-1])
    return list(reversed(selected_images))


def get_screencap_img(sct):
    img = np.asarray(sct.grab(config.monitor))  # in bgra format
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    # cv2 in comparison to keras uses width, height order
    img = cv2.resize(img, (config.width, config.height))
    # ! check again with new data
    img = img.reshape(config.height, config.width, 3)
    return img


if __name__ == "__main__":
    main_with_lstm()
    # main_with_cnn()
    # show_screen()     # check alignment before running model
