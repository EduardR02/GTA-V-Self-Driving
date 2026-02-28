from collections import Counter
import numpy as np
import mss
import cv2
import time
import os
import pandas as pd
from grabkeys import key_check
import h5py
from threading import Thread as Worker
import config
import utils

curr_file_index = 0
gb_per_file = 2
current_data_dir = config.stuck_data_dir_name


def convert_output(key):
    output = np.zeros(config.output_classes)
    if "A" in key and "W" in key:
        output[config.outputs["wa"]] = 1
    elif "D" in key and "W" in key:
        output[config.outputs["wd"]] = 1
    elif "S" in key:
        output[config.outputs["s"]] = 1
    elif "D" in key:
        output[config.outputs["d"]] = 1
    elif "A" in key:
        output[config.outputs["a"]] = 1
    elif "W" in key:
        output[config.outputs["w"]] = 1
    else:
        output[config.outputs["nothing"]] = 1
    # one hot np array
    return output


def convert_output_raw(key):
    """
    Converts the input to a label where each button that was pressed is recorded, not one hot
    """
    output = np.zeros(len(config.outputs_base))
    if "W" in key:
        output[config.outputs_base["w"]] = 1
    if "A" in key:
        output[config.outputs_base["a"]] = 1
    if "S" in key:
        output[config.outputs_base["s"]] = 1
    if "D" in key:
        output[config.outputs_base["d"]] = 1
    # nothing output is not needed as each input is recorded individually, so if none are pressed all are 0
    return output


def start():
    for i in range(3, 0, -1):
        print(i)
        time.sleep(1)
    print("Go!")


def main(only_fps_test_mode=False, no_pause_remove=True):
    global curr_file_index
    correct_file_counter(True)  # start with new file
    train_images = []
    train_labels = []
    paused = True
    print("starting paused")
    counter = 0
    counter2 = 0
    n = 1
    t = time.time()
    sct = mss.mss()
    key_check()
    while True:
        if not paused:
            counter += 1
            counter2 += 1
            img, label = get_image_and_label(sct)
            train_images.append(img)
            train_labels.append(label)
            if len(train_labels) == 290 or len(train_labels) == (config.sequence_len - 1) * utils.get_fps_ratio() - 10:
                print(len(train_labels))
            if len(train_labels) % 1000 == 0:
                print(counter)
                print("Average Fps:", counter / n)
                if not only_fps_test_mode:
                    # reduces fps a little for the time while saving, but without would stop loop for time of exec,
                    # meaning gaps in the data, multiprocessing module works worse, halts execution for a short time.
                    temp_images = train_images
                    temp_labels = train_labels
                    t1 = Worker(target=save_with_hdf5, args=(temp_images, temp_labels))
                    t1.start()
                    train_images = []
                    train_labels = []
            if time.time() - t >= 1:
                t = time.time()
                n += 1
                counter2 = 0
        else:
            time.sleep(0.1)
        key = key_check()
        if "T" in key:
            if paused:
                print("unpaused")
            else:
                # if you press pause it probably means you want to discard last x frames
                if no_pause_remove or (len(train_labels) > config.amt_remove_after_pause and not only_fps_test_mode):
                    print("paused, wait for saving to finish")
                    if not no_pause_remove:
                        train_images = train_images[:-config.amt_remove_after_pause]
                        train_labels = train_labels[:-config.amt_remove_after_pause]
                    t1 = Worker(target=save_with_hdf5, args=(train_images, train_labels))
                    t1.start()
                    t1.join()
                    print("done saving")
                else:
                    print("No need to save, too little data after last save")
                curr_file_index = curr_file_index + 1
            paused = not paused
            time.sleep(0.1)
            train_images = []
            train_labels = []
            t = time.time()


def get_image_and_label(sct):
    img = np.asarray(sct.grab(config.monitor))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imshow("Car View", img)
    img = cv2.resize(img, (config.width, config.height))
    key = key_check()
    output = convert_output(key)
    return img, output


def correct_file_counter(called_on_start):
    global curr_file_index
    filename = current_data_dir + config.data_name + f"_{curr_file_index}.h5"
    # iterate  to writable file
    while os.path.isfile(filename) and (called_on_start or os.stat(filename).st_size > (1024 ** 3 * gb_per_file)):
        curr_file_index = curr_file_index + 1
        filename = current_data_dir + config.data_name + f"_{curr_file_index}.h5"


def save_with_hdf5(data_x, data_y):
    global curr_file_index
    correct_file_counter(False)
    filename = current_data_dir + config.data_name + f"_{curr_file_index}.h5"
    # both are lists of numpy array, retain structure but make it np array
    data_x = np.stack(data_x, axis=0)
    data_y = np.stack(data_y, axis=0)
    # don't do chunks=True, it makes it giga slow
    if not os.path.isfile(filename):
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset("images", data=data_x,
                              maxshape=(None, config.height, config.width, config.color_channels))
            hf.create_dataset("labels", data=data_y,
                              maxshape=(None, config.output_classes))
    else:
        with h5py.File(filename, 'a') as hf:
            hf["images"].resize((hf["images"].shape[0] + data_x.shape[0]), axis=0)
            hf["images"][-data_x.shape[0]:] = data_x

            hf["labels"].resize((hf["labels"].shape[0] + data_y.shape[0]), axis=0)
            hf["labels"][-data_y.shape[0]:] = data_y
    file_size = os.stat(filename).st_size
    print(file_size)
    if file_size > (1024 ** 3 * gb_per_file): curr_file_index += 1
    return


def display_data(data):
    df = pd.DataFrame(data)
    print(df.head())
    print(Counter(df[1].apply(str)))


def show_training_data_sequenced():
    # see what happens if you change height and width, don't mess up on reshaping for the neural net
    images, labels = utils.load_file(current_data_dir + "240x180_rgb_3.h5")
    classes = labels.shape[-1]
    print(images.shape, labels.shape)
    timeseries = generate_timeseries(images, labels, shuffle=True, incorporate_fps=True)
    images, labels = [], []
    # unpack and reshape
    for batch in timeseries:
        img, lb = batch
        images += [img]
        labels += [lb]
    images = np.concatenate(images)
    labels = np.concatenate(labels)
    # flatten time and batch dim
    images = images.reshape((-1, config.height, config.width, config.color_channels))
    labels = labels.reshape((-1, classes))

    print(images.shape)
    i = 0
    j = 0
    t = time.time()
    count_frames = 0
    while True:
        img = images[i]
        j += 1 if i != 0 and i % config.sequence_len == 0 else 0
        label = np.argmax(labels[j])
        for key, value in config.outputs.items():
            if value == label:
                label = key
                break
        print(label)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # because cv2 imshow uses bgr not rgb
        cv2.imshow("car_view", img)
        if time.time() - t > 1:
            t = time.time()
            print("fps:", count_frames)
            count_frames = 0
        count_frames += 1
        i += 1
        # time.sleep(0.2)
        if len(images) <= i:
            break
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def test_collection_correct():
    images, labels = utils.load_data(current_data_dir)
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)
    print(images.shape, labels.shape)
    labels = np.argmax(labels, axis=-1)
    print(labels)
    print(labels.shape)
    labels = np.unique(labels)
    print(labels)


def play_data(filepath, print_labels=True, fps=40):
    """
    Simple function to load and play back recorded data as a video.
    
    Args:
        filepath: Path to the .h5 file to load
        print_labels: Whether to print the label for each frame
        fps: Frames per second for playback (controls speed)
    """
    # Load the data
    images, labels = utils.load_file(filepath)
    print(f"Loaded {images.shape[0]} frames from {filepath}")
    
    # Calculate target time per frame in seconds
    target_frame_time = 1.0 / fps
    
    # FPS tracking variables
    start_time = time.time()
    frame_count = 0
    last_fps_print = time.time()
    next_frame_time = start_time
    
    # Play through each frame
    for i in range(len(images)):
        frame_start = time.time()
        
        img = images[i]
        label = labels[i]
        
        # Convert label to readable format
        label_idx = np.argmax(label)
        label_name = None
        for key, value in config.outputs.items():
            if value == label_idx:
                label_name = key
                break
        
        if print_labels:
            print(f"Frame {i}: {label_name}")
        
        # Convert from RGB to BGR for cv2
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Display the frame
        cv2.imshow("Data Playback", img_bgr)
        
        # Track FPS
        frame_count += 1
        if time.time() - last_fps_print >= 1.0:
            actual_fps = frame_count / (time.time() - start_time)
            print(f"Actual FPS: {actual_fps:.2f} (Target: {fps})")
            last_fps_print = time.time()
        
        # Calculate next frame time
        next_frame_time += target_frame_time
        
        # Always call waitKey(1) to update window (this is required for cv2.imshow)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Wait until it's time for the next frame with precise timing
        time_until_next = next_frame_time - time.time()
        if time_until_next > 0:
            time.sleep(time_until_next)
    
    # Print final FPS stats
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"\nPlayback finished: {frame_count} frames in {total_time:.2f}s")
    print(f"Average actual FPS: {avg_fps:.2f} (Target: {fps})")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # load_data()
    # normalize()
    # main(False, no_pause_remove=True)
    # test_collection_correct()
    # show_training_data_sequenced()
    play_data(current_data_dir + "\\240x180_rgb_33.h5", print_labels=True, fps=80)

