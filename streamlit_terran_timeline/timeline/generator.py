import json
import numpy as np
import os

import streamlit as st
from scipy.spatial.distance import cosine

from terran.face import face_detection, extract_features
from terran.io import open_video, open_image

from .utils import (
    crop_expanded_pad,
    get_thumbnail,
    get_video_id,
    to_base64,
)


def generate_timeline(
    video_src,
    json_f,
    ref_directory=None,
    appearence_threshold=None,
    batch_size=16,
    duration=None,
    framerate=4,
    output_directory=None,
    similarity_threshold=0.5,
    start_time=0,
    thumbnail_rate=None,
):
    """Generates a face-recognition timeline from a video.

    Parameters
    ----------
    video_src : str
        A path to a local video file or a link to any video on a streaming
        platform. All streaming platforms supported by YoutubeDL are supported.
    ref_directory : str, pathlike or None
        A path to a folder containing images of faces to look for in the video. If the
        value is None, then it'll automatically collect the faces as we read the video
        and generate their timeline automatically.
    appearence_threshold : int
        If a face appears more then this amount it will be considered for the timeline
    batch_size : int
        How many frames to process at once
    duration : int
        How many seconds of the video should be processed. If equals to None then
        all the video is processed
    framerate : int
        How many frames per second we should process
    output_directory : str, pathlike or None
        Where to store the timeline results as a JSON file. If None, it won't save the
        results
    similarity_threshold : float
        A distance value for when two faces are the same
    start_time : int
        The starting time (in seconds) to beging the timeline generation
    thumbnail_rate : int or None
        Collect a thumbnail of the video for every X seconds. If None, it won't collect
        thumbnails.
    """
    progress_bar = st.progress(0)

    machine_by_track = {}

    video = open_video(
        video_src,
        batch_size=batch_size,
        framerate=framerate,
        read_for=duration,
        start_time=start_time,
    )

    st.info("🕰  Производим детекцию строительной техники в кадре")
    timestamps_by_track = {}
    thumbnails = []
    last_timestamp = 0

    video_lengh = len(video)
    #print("video_length", video_lengh)
    num_ids = 0
    machine_by_track_black = {}
    machine_by_track_white = {}
    for machine_id, sublist in json_f.items():
        timestamps_by_track[machine_id] = []
        num_ids+=1
        #print(machine_id)
        #machine_id = str(int(machine_id)-1)
        machine_by_track_black[machine_id] = []
        
        for subdict in sublist:
            #print(machine_id)
            bbox = subdict["object_coords"]
            frame_idx = subdict["frame_idx_start"]
            machine_by_track_black[machine_id].append({"frame_idx": frame_idx, "bbox": bbox})
            frame_timestamp_start = subdict["frame_idx_start"]
            frame_timestamp_end = subdict["frame_idx_end"]
            frame_list = [x for x in range(frame_timestamp_start, frame_timestamp_end+1)]
            timestamps_by_track[machine_id] += frame_list

    for bidx, frames in enumerate(video):
        #faces_per_frame = face_detection(frames)
        #features_per_frame = extract_features(frames, faces_per_frame)

        for fidx, frame in enumerate(frames):
            frame_idx = bidx * video.batch_size + fidx

            if thumbnail_rate is not None and frame_idx % thumbnail_rate == 0:
                thumbnails.append(get_thumbnail(frame))

            for machine_id, value in machine_by_track_black.items():
                #machine_id = str(int(machine_id)-1)
                for subdict in machine_by_track_black[machine_id]:
                    if frame_idx == subdict["frame_idx"]:
                        #print("machine_id:", machine_id, "frame_idx:", frame_idx)
                        bbox = machine_by_track_black[machine_id][0]["bbox"]
                        x_min, x_max, y_min, y_max = bbox.values()
                        machine_by_track_white[machine_id] = crop_expanded_pad(
                        frame, (x_min, y_min, x_max, y_max), factor=0.0
                    )
                        break
            last_timestamp = frame_idx

        progress = min(100, int(((bidx + 1) / video_lengh) * 100))
        progress_bar.progress(progress)
    print("last_timestamp:", last_timestamp)

    appearance = {}

    for i, (_, timestamps) in enumerate(timestamps_by_track.items()):
        if appearence_threshold and len(timestamps) / framerate < appearence_threshold:
            continue

        track_appearance = np.zeros((last_timestamp + 1), dtype=np.bool_)
        for ts in timestamps:   
            track_appearance[ts] = 1

        appearance[i] = track_appearance.tolist()

    track_ids = list(sorted(appearance.keys()))
    print(machine_by_track_white.keys())
    i = 0
    copy_of_dict = {}
    #print("sorted:",  list(sorted(machine_by_track_white.keys())))
    #print("sorted_2:", list({k: v for k, v in sorted(machine_by_track_white.items(), key=lambda item: [int(s) for s in item[0].split() if s.isdigit()][0])}))
    for key in list({k: v for k, v in sorted(machine_by_track_white.items(), key=lambda item: [int(s) for s in item[0].split() if s.isdigit()][0])}):
        copy_of_dict[i] = machine_by_track_white[key]
        i+=1
    machine_by_track_white = copy_of_dict
    print(machine_by_track_white.keys())
    video_id = get_video_id(video_src)
    timeline = dict(
        id=video_id,
        url=video_src,
        appearance=appearance,
        track_ids=track_ids,
        framerate=video.framerate,
        start_time=video.start_time,
        end_time=video.start_time + video.duration,
        track_faces={
            machine_id: to_base64(machine) for machine_id, machine in machine_by_track_white.items()
        },
        thumbnail_rate=thumbnail_rate,
        thumbnails=[to_base64(th) for th in thumbnails],
    )
    if output_directory is not None:
        os.makedirs(output_directory, exist_ok=True)
        with open(os.path.join(output_directory, f"{video_id}.json"), "w") as f:
            json.dump(timeline, f)

    #st.success(f"💿  Successfully generated timeline for video {video_src}")
    return timeline

