import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from multiprocessing import Pool


def init_cdf():
    base_dir = '../datasets/celeb-df-v2/versions/1'
    all_videos = []
    for subdir in ['Celeb-real', 'Celeb-synthesis', 'YouTube-real']:
        video_dir = os.path.join(base_dir, subdir)
        for root, _, files in os.walk(video_dir):
            for file in files:
                if file.endswith('.mp4'):
                    all_videos.append(os.path.join(root, file))

    video_list_txt = os.path.join(base_dir, 'List_of_testing_videos.txt')
    test_videos = set()
    labels = {}
    with open(video_list_txt, 'r') as f:
        for data in f:
            line = data.split()
            label = 1 - int(line[0])
            path = os.path.join(base_dir, line[1].replace('/', os.sep))
            test_videos.add(path)
            labels[path] = label

    for video in all_videos:
        if video not in labels:
            if 'Celeb-real' in video or 'YouTube-real' in video:
                labels[video] = 0
            elif 'Celeb-synthesis' in video:
                labels[video] = 1

    return all_videos, labels, test_videos


def extract_frames(org_path, num_frames=10, save_path=''):
    cap_org = cv2.VideoCapture(org_path)
    frame_count_org = int(cap_org.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count_org == 0:
        print(f"Error: {org_path} contains no frames.")
        return

    frame_idxs = np.linspace(0, frame_count_org - 1, num_frames, endpoint=True, dtype=np.int32)

    for cnt_frame in range(frame_count_org):
        ret_org, frame_org = cap_org.read()
        if not ret_org:
            tqdm.write(f"Error reading frame {cnt_frame} from {org_path}.")
            break

        if cnt_frame not in frame_idxs:
            continue

        os.makedirs(save_path, exist_ok=True)
        image_name = f"{os.path.basename(org_path).replace('.mp4', '')}_{str(cnt_frame).zfill(3)}.png"
        image_path = os.path.join(save_path, image_name)

        if not os.path.isfile(image_path):
            cv2.imwrite(image_path, frame_org)

    cap_org.release()


def assign_video_to_folder(video_path, label, test_videos, num_frames):
    real_save_path = '../datasets/Celeb-DF-v2/real'
    fake_save_path = '../datasets/Celeb-DF-v2/fake'
    real_test_save_path = '../datasets/Celeb-DF-v2/real_test'
    fake_test_save_path = '../datasets/Celeb-DF-v2/fake_test'

    if video_path in test_videos:
        if label == 0:
            save_path = real_test_save_path
        else:
            save_path = fake_test_save_path
    else:
        if label == 0:
            save_path = real_save_path
        else:
            save_path = fake_save_path

    extract_frames(video_path, num_frames=num_frames, save_path=save_path)
    return video_path


def Bar(arg):
    print(f"{arg} Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', dest='num_frames', type=int, default=32)
    args = parser.parse_args()

    all_videos, labels, test_videos = init_cdf()

    pool = Pool(processes=16)
    for video_path in all_videos:
        pool.apply_async(
            assign_video_to_folder,
            args=(video_path, labels[video_path], test_videos, args.num_frames),
            callback=Bar
        )

    pool.close()
    pool.join()

    print("Frame extraction and saving completed.")