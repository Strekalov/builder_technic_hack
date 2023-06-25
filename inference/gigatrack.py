# https://github.com/ultralytics/ultralytics/issues/1429#issuecomment-1519239409

from pathlib import Path
import torch
import argparse
import numpy as np
import cv2
from types import SimpleNamespace
import json

from ultralytics import YOLO
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS
from boxmot.utils.checks import TestRequirements
from boxmot.utils import logger as LOGGER
from boxmot.utils.torch_utils import select_device

tr = TestRequirements()
tr.check_packages(("ultralytics",))  # install

from ultralytics.yolo.engine.model import YOLO, TASK_MAP

from ultralytics.yolo.utils import (
    SETTINGS,
    colorstr,
    ops,
    is_git_dir,
    IterableSimpleNamespace,
)
from ultralytics.yolo.utils.checks import check_imgsz, print_args
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.engine.results import Boxes
from ultralytics.yolo.data.utils import VID_FORMATS
from ultralytics.yolo.utils.plotting import save_one_box

from copy import copy, deepcopy

from multi_yolo_backend import MultiYolo
from math import dist


# from utils import write_MOT_results
import sys

LOGGER.remove()
LOGGER.add(sys.stdout, level="DEBUG")

FPS = 6
JSON_STORAGE = dict()
TEMP_DB = dict()

WORKED_TEMP_DB = dict()
WORKED_JSON_STORAGE = dict()


THRESHOLD_FRAMES = 2

ID_TO_TITLE = {
    0: "асфальтоукладчик",
    1: "бетономешалка",
    2: "экскаватор",
    3: "подъемный кран",
    4: "трактор",
    5: "грузовик",
}


def has_moved(old_center, new_center, tresh):
    
    distance = dist(old_center, new_center)

    if distance > tresh:
        print(old_center, new_center, distance)
    return distance > tresh


def on_predict_start(predictor):
    predictor.trackers = []
    predictor.tracker_outputs = [None] * predictor.dataset.bs
    predictor.args.tracking_config = (
        ROOT
        / "boxmot"
        / opt.tracking_method
        / "configs"
        / (opt.tracking_method + ".yaml")
    )
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.args.tracking_method,
            predictor.args.tracking_config,
            predictor.args.reid_model,
            predictor.device,
            predictor.args.half,
        )
        predictor.trackers.append(tracker)


# def che


@torch.no_grad()
def run(args):
    
    args.conf = 0.8
    args.device = "0"
    args.half = True
    args.imgsz = [1440, 2560]
    args.iou = 0.45
    args.save = True
    args.tracking_method = "deepocsort"
    args.vid_stride = 1
    args.yolo_model = Path("./yolov8x6.pt")
    args.reid_model = Path("./osnet_x1_0_imagenet.pt")
    args.kp_model = Path("./keipointv8.pt")
    args.tracking_method = "deepocsort"

    model = YOLO(args.yolo_model)

    model_kp_exc = YOLO(args.kp_model)

    overrides = model.overrides.copy()
    model.predictor = TASK_MAP[model.task][3](
        overrides=overrides, _callbacks=model.callbacks
    )

    # extract task predictor
    predictor = model.predictor

    # combine default predictor args with custom, preferring custom
    combined_args = {**predictor.args.__dict__, **args}
    # overwrite default args
    predictor.args = IterableSimpleNamespace(**combined_args)
    predictor.args.device = select_device(args["device"])
    # LOGGER.info(args)

    # setup source and model
    if not predictor.model:
        predictor.setup_model(model=model.model, verbose=False)
    predictor.setup_source(predictor.args.source)

    predictor.args.imgsz = check_imgsz(
        predictor.args.imgsz, stride=model.model.stride, min_dim=2
    )  # check image size
    predictor.save_dir = increment_path(
        Path(predictor.args.project) / predictor.args.name,
        exist_ok=predictor.args.exist_ok,
    )

    # Check if save_dir/ label file exists
    if predictor.args.save or predictor.args.save_txt:
        (
            predictor.save_dir / "labels"
            if predictor.args.save_txt
            else predictor.save_dir
        ).mkdir(parents=True, exist_ok=True)
    # Warmup model
    if not predictor.done_warmup:
        predictor.model.warmup(
            imgsz=(
                1
                if predictor.model.pt or predictor.model.triton
                else predictor.dataset.bs,
                3,
                *predictor.imgsz,
            )
        )
        predictor.done_warmup = True
    predictor.seen, predictor.windows, predictor.batch, predictor.profilers = (
        0,
        [],
        None,
        (ops.Profile(), ops.Profile(), ops.Profile(), ops.Profile()),
    )
    predictor.add_callback("on_predict_start", on_predict_start)
    predictor.run_callbacks("on_predict_start")

    model = MultiYolo(
        model=model.predictor.model
        if "v8" in str(args["yolo_model"])
        else args["yolo_model"],
        device=predictor.device,
        args=predictor.args,
    )

    for frame_idx, batch in enumerate(predictor.dataset):
        predictor.run_callbacks("on_predict_batch_start")
        predictor.batch = batch
        path, im0s, vid_cap, s = batch
        visualize = (
            increment_path(
                predictor.save_dir / Path(path[0]).stem, exist_ok=True, mkdir=True
            )
            if predictor.args.visualize and (not predictor.dataset.source_type.tensor)
            else False
        )

        n = len(im0s)
        predictor.results = [None] * n

        # Preprocess
        with predictor.profilers[0]:
            im = predictor.preprocess(im0s)

        # Inference
        with predictor.profilers[1]:
            preds = model(im, im0s)

        # Postprocess moved to MultiYolo
        with predictor.profilers[2]:
            predictor.results = model.postprocess(path, preds, im, im0s, predictor)
        predictor.run_callbacks("on_predict_postprocess_end")

        # Visualize, save, write results
        n = len(im0s)
        for i in range(n):
            if (
                predictor.dataset.source_type.tensor
            ):  # skip write, show and plot operations if input is raw tensor
                continue
            p, im0 = path[i], im0s[i].copy()
            p = Path(p)

            with predictor.profilers[3]:
                # get raw bboxes tensor
                dets = predictor.results[i].boxes.data
                # get tracker predictions
                predictor.tracker_outputs[i] = predictor.trackers[i].update(
                    dets.cpu().detach().numpy(), im0
                )
            predictor.results[i].speed = {
                "preprocess": predictor.profilers[0].dt * 1e3 / n,
                "inference": predictor.profilers[1].dt * 1e3 / n,
                "postprocess": predictor.profilers[2].dt * 1e3 / n,
                "tracking": predictor.profilers[3].dt * 1e3 / n,
            }

            # filter boxes masks and pose results by tracking results
            model.filter_results(i, predictor)
            # overwrite bbox results with tracker predictions
            model.overwrite_results(i, im0.shape[:2], predictor)
            # print(predictor.tracker_outputs)
            # print(.shape)
            boxes = predictor.tracker_outputs[0]
            # print(predictor.results)
            kps_preds = model_kp_exc(im)
            keypoints = kps_preds[0].keypoints
            technic_in_frame = set(boxes[..., 4].astype(int).tolist())

            technic_in_history = set(TEMP_DB.keys())

            old_technic = technic_in_history - technic_in_frame
            # print(technic_in_history, technic_in_frame)
            # worked_technic_in_history = set(WORKED_TEMP_DB.keys())
            # # print(technic_in_history, technic_in_frame)
            # worked_old_technic = worked_technic_in_history - technic_in_frame

            for uid in old_technic:
                cls_id = TEMP_DB[uid]["technic_type_id"]
                title = TEMP_DB[uid]["title"]
                old_frame_idx = frame_idx - TEMP_DB[uid]["frame_idx"]

                if old_frame_idx < THRESHOLD_FRAMES:
                    # del JSON_STORAGE[uid]
                    JSON_STORAGE[uid].pop()
                    del TEMP_DB[uid]
                    continue

                tmp = {
                    "object_uid": uid,
                    "technic_type_id": cls_id,
                    "title": title,
                    "event_type": 0,
                    "event_title": "выход из кадра",
                    # "confidence": conf,
                    "timestamp": int(frame_idx / FPS),
                    # "worked_start_timestamp": frame_idx/FPS,
                    "worked_end_timestamp": int(frame_idx / FPS),
                    "frame_idx": frame_idx,
                }

                # TEMP_DB[uid].update(tmp)

                event_dict = deepcopy(TEMP_DB[uid])
                event_dict.update(tmp)

                JSON_STORAGE[uid].append(event_dict)

                del TEMP_DB[uid]

                tmp = {
                    "object_uid": uid,
                    "technic_type_id": cls_id,
                    "title": title,
                    "event_type": 3,
                    "event_title": "конец работы",
                    # "confidence": conf,
                    # "timestamp": int(frame_idx / FPS),
                    # "worked_start_timestamp": frame_idx/FPS,
                    # "worked_end_timestamp": int(frame_idx / FPS),
                    "frame_idx": frame_idx,
                }

                # TEMP_DB[uid].update(tmp)

                event_dict = deepcopy(WORKED_TEMP_DB[uid])
                event_dict.update(tmp)

                WORKED_JSON_STORAGE[uid].append(event_dict)
                del WORKED_TEMP_DB[uid]

            for box in boxes:
                x1, y1, x2, y2, uid, conf, cls = box
                x1, y1, x2, y2, uid, cls = map(int, [x1, y1, x2, y2, uid, cls])

                history_object = WORKED_TEMP_DB.get(uid)

                if history_object is not None:
                    # print(history_object["status"])
                    old_coords = history_object["object_coords"]
                    
                    
                    if cls in [2, 3]:
                        kp_x, kp_y =keypoints[0, :2].astype(int)
                        old_kp_x, old_kp_y = history_object["kp_x"], history_object["old_kp_y"]
                        
                        if has_moved((old_kp_x, old_kp_y), (kp_x, kp_y), 20):
                            if (
                                history_object["status"] == 0
                                and frame_idx - history_object["frame_idx"] > 5
                            ):
                                tmp = {
                                    "status": 1,
                                    "object_uid": uid,
                                    "technic_type_id": cls,
                                    "title": ID_TO_TITLE.get(cls),
                                    "event_type": 3,
                                    "event_title": "конец работы",
                                    # "confidence": conf,
                                    "timestamp": int(frame_idx / FPS),
                                    # "worked_start_timestamp": int(frame_idx / FPS),
                                    "worked_end_timestamp": int(frame_idx / FPS),
                                    "frame_idx": frame_idx,

                                    "object_coords": {
                                        "x1": x1,
                                        "x2": x2,
                                        "y1": y1,
                                        "y2": y2,
                                    },
                            }

                            event_dict = deepcopy(WORKED_TEMP_DB[uid])
                            event_dict.update(tmp)

                            WORKED_JSON_STORAGE[uid].append(event_dict)
                            WORKED_TEMP_DB[uid] = event_dict
                        elif (
                        history_object["status"] == 1
                        and (frame_idx - history_object["frame_idx"]) > 5
                    ):
                            tmp = {
                            "status": 0,
                            "object_uid": uid,
                            "technic_type_id": cls,
                            "title": ID_TO_TITLE.get(cls),
                            "event_type": 3,
                            "event_title": "конец работы",
                            # "confidence": conf,
                            "timestamp": int(frame_idx / FPS),
                            # "worked_start_timestamp": int(frame_idx / FPS),
                            "worked_end_timestamp": int(frame_idx / FPS),
                            "frame_idx": frame_idx,
                            "object_coords": {
                                "x1": x1,
                                "x2": x2,
                                "y1": y1,
                                "y2": y2,
                            },
                        }
                        
                    old_x1 = old_coords["x1"]
                    old_y1 = old_coords["y1"]
                    old_x2 = old_coords["x2"]
                    old_y2 = old_coords["y2"]

                    old_center = (
                        int((old_x2 - old_x1) / 2),
                        int((old_y2 - old_y1) / 2),
                    )
                    new_center = (int((x2 - x1) / 2), int((y2 - y1) / 2))

                    if has_moved(old_center, new_center, 10):
                        # print(old_center, new_center)

                        if (
                            history_object["status"] == 0
                            and frame_idx - history_object["frame_idx"] > 5
                        ):
                            tmp = {
                                "object_uid": uid,
                                "status": 1,
                                "technic_type_id": cls,
                                "title": ID_TO_TITLE.get(cls),
                                "event_type": 2,
                                "event_title": "начало работы",
                                "confidence": conf,
                                "timestamp": int(frame_idx / FPS),
                                "worked_start_timestamp": int(frame_idx / FPS),
                                "worked_end_timestamp": None,
                                "frame_idx": frame_idx,
                                "object_coords": {
                                    "x1": x1,
                                    "x2": x2,
                                    "y1": y1,
                                    "y2": y2,
                                },
                            }

                            event_dict = deepcopy(WORKED_TEMP_DB[uid])
                            event_dict.update(tmp)

                            WORKED_JSON_STORAGE[uid].append(event_dict)
                            WORKED_TEMP_DB[uid] = tmp

                    elif (
                        history_object["status"] == 1
                        and (frame_idx - history_object["frame_idx"]) > 5
                    ):
                        tmp = {
                            "status": 0,
                            "object_uid": uid,
                            "technic_type_id": cls,
                            "title": ID_TO_TITLE.get(cls),
                            "event_type": 3,
                            "event_title": "конец работы",
                            # "confidence": conf,
                            "timestamp": int(frame_idx / FPS),
                            # "worked_start_timestamp": int(frame_idx / FPS),
                            "worked_end_timestamp": int(frame_idx / FPS),
                            "frame_idx": frame_idx,
                            "object_coords": {
                                "x1": x1,
                                "x2": x2,
                                "y1": y1,
                                "y2": y2,
                            },
                        }

                        event_dict = deepcopy(WORKED_TEMP_DB[uid])
                        event_dict.update(tmp)

                        WORKED_JSON_STORAGE[uid].append(event_dict)
                        WORKED_TEMP_DB[uid] = event_dict

                        # tmp = {
                        #     "object_uid": uid,
                        #     "technic_type_id": cls_id,
                        #     "title": title,
                        #     "event_type": 0,
                        #     "event_title": "выход из кадра",
                        #     # "confidence": conf,
                        #     "timestamp": int(frame_idx / FPS),
                        #     # "worked_start_timestamp": frame_idx/FPS,
                        #     "worked_end_timestamp": int(frame_idx / FPS),
                        #     "frame_idx": frame_idx,
                        # }

                        # # TEMP_DB[uid].update(tmp)
                    

                    
                        # tmp = {
                        #         "frame_idx": frame_idx,
                        #         "object_coords": {
                        #             "x1": x1,
                        #             "x2": x2,
                        #             "y1": y1,
                        #             "y2": y2,
                        #         }
                        # }

                        # WORKED_TEMP_DB[uid].update(tmp)

                if TEMP_DB.get(uid) is None:
                    TEMP_DB[uid] = {
                        "object_uid": uid,
                        "technic_type_id": cls,
                        "title": ID_TO_TITLE.get(cls),
                        "event_type": 1,
                        "event_title": "появление в кадре",
                        "confidence": conf,
                        "timestamp": int(frame_idx / FPS),
                        "worked_start_timestamp": int(frame_idx / FPS),
                        "worked_end_timestamp": None,
                        "frame_idx": frame_idx,
                        "object_coords": {"x1": x1, "x2": x2, "y1": y1, "y2": y2},
                    }



                    JSON_STORAGE[uid] = list()
                    JSON_STORAGE[uid].append(TEMP_DB[uid])

                    WORKED_TEMP_DB[uid] = {
                        "object_uid": uid,
                        "technic_type_id": cls,
                        "title": ID_TO_TITLE.get(cls),
                        "status": 0,
                        "confidence": conf,
                        "timestamp": int(frame_idx / FPS),
                        # "worked_start_timestamp": int(frame_idx / FPS),
                        # "worked_end_timestamp": None,
                        "frame_idx": frame_idx,
                        "object_coords": {"x1": x1, "x2": x2, "y1": y1, "y2": y2},
                        
                    }
                    if cls in [2, 3]:
                        kp_x, kp_y =keypoints[0, :2].astype(int)

                        WORKED_TEMP_DB[uid].update({"kp_x": kp_x, "kp_y": kp_y})
                    WORKED_JSON_STORAGE[uid] = list()
                    # JSON_STORAGE[uid].append(TEMP_DB[uid])


            
                
                    
                    
            # for uid in old_technic:
            #     cls_id = WORKED_TEMP_DB[uid]["technic_type_id"]
            #     title = WORKED_TEMP_DB[uid]["title"]
            #     old_frame_idx = frame_idx - WORKED_TEMP_DB[uid]["frame_idx"]

            #     if old_frame_idx < THRESHOLD_FRAMES:
            #         del WORKED_JSON_STORAGE[uid]
            #         del WORKED_TEMP_DB[uid]
            #         continue

            #     tmp = {
            #         "object_uid": uid,
            #         "technic_type_id": cls_id,
            #         "title": title,
            #         "event_type": 3,
            #         "event_title": "конец работы",
            #         # "confidence": conf,
            #         "timestamp": int(frame_idx / FPS),
            #         # "worked_start_timestamp": frame_idx/FPS,
            #         "worked_end_timestamp": int(frame_idx / FPS),
            #         "frame_idx": frame_idx,
            #     }

            #     # TEMP_DB[uid].update(tmp)

            #     event_dict = deepcopy(WORKED_TEMP_DB[uid])
            #     event_dict.update(tmp)

            #     WORKED_JSON_STORAGE[uid].append(event_dict)

            #     del WORKED_TEMP_DB[uid]

            # for box in boxes:
            #     x1, y1, x2, y2, uid, conf, cls = box
            #     x1, y1, x2, y2, uid, cls = map(int, [x1, y1, x2, y2, uid, cls])

            #     if WORKED_JSON_STORAGE.get(uid) is not None:
            #         # print(uid, old_technic)
            #         pass

            #     else:
            #         WORKED_TEMP_DB[uid] = {
            #             "object_uid": uid,
            #             "technic_type_id": cls,
            #             "title": ID_TO_TITLE.get(cls),
            #             "event_type": 2,
            #             "event_title": "начало работы",
            #             "worked"
            #             "confidence": conf,
            #             "timestamp": int(frame_idx / FPS),
            #             "worked_start_timestamp": int(frame_idx / FPS),
            #             "worked_end_timestamp": None,
            #             "frame_idx": frame_idx,
            #             "object_coords": {"x1": x1, "x2": x2, "y1": y1, "y2": y2},
            #         }

            #         WORKED_JSON_STORAGE[uid] = list()
            #         WORKED_JSON_STORAGE[uid].append(WORKED_TEMP_DB[uid])

            # write inference results to a file or directory
            if (
                predictor.args.verbose
                or predictor.args.save
                or predictor.args.save_txt
                or predictor.args.show
                or predictor.args.save_id_crops
            ):
                s += predictor.write_results(i, predictor.results, (p, im, im0))
                predictor.txt_path = Path(predictor.txt_path)

                # write MOT specific results
                if predictor.args.source.endswith(VID_FORMATS):
                    predictor.MOT_txt_path = predictor.txt_path.parent / p.stem
                else:
                    # append folder name containing current img
                    predictor.MOT_txt_path = predictor.txt_path.parent / p.parent.name

                if predictor.tracker_outputs[i].size != 0 and predictor.args.save_mot:
                    # needed if txt save is not activated, otherwise redundant
                    predictor.MOT_txt_path.mkdir(
                        parents=True, exist_ok=predictor.args.exist_ok
                    )
                    # write_MOT_results(
                    #     predictor.MOT_txt_path,
                    #     predictor.results[i],
                    #     frame_idx,
                    #     i,
                    # )

                if predictor.args.save_id_crops:
                    for d in predictor.results[i].boxes:
                        save_one_box(
                            d.xyxy,
                            im0.copy(),
                            file=predictor.save_dir
                            / "crops"
                            / str(int(d.cls.cpu().numpy().item()))
                            / str(int(d.id.cpu().numpy().item()))
                            / f"{p.stem}.jpg",
                            BGR=True,
                        )

            # display an image in a window using OpenCV imshow()
            if predictor.args.show and predictor.plotted_img is not None:
                predictor.show(p.parent)

            # save video predictions
            if predictor.args.save and predictor.plotted_img is not None:
                predictor.save_preds(vid_cap, i, str("outputs_with_boxes.webm"))

        predictor.run_callbacks("on_predict_batch_end")

        # print time (inference-only)
        if predictor.args.verbose:
            LOGGER.info(
                f"{s}YOLO {predictor.profilers[1].dt * 1E3:.1f}ms, TRACKING {predictor.profilers[3].dt * 1E3:.1f}ms"
            )

    # Release assets
    if isinstance(predictor.vid_writer[-1], cv2.VideoWriter):
        predictor.vid_writer[-1].release()  # release final video writer

    # Print results
    if predictor.args.verbose and predictor.seen:
        t = tuple(
            x.t / predictor.seen * 1e3 for x in predictor.profilers
        )  # speeds per image
        LOGGER.info(
            f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess, %.1fms tracking per image at shape "
            f"{(1, 3, *predictor.args.imgsz)}" % t
        )
    if predictor.args.save or predictor.args.save_txt or predictor.args.save_crop:
        nl = len(list(predictor.save_dir.glob("labels/*.txt")))  # number of labels
        s = (
            f"\n{nl} label{'s' * (nl > 1)} saved to {predictor.save_dir / 'labels'}"
            if predictor.args.save_txt
            else ""
        )
        LOGGER.info(f"Results saved to {colorstr('bold', predictor.save_dir)}{s}")

    predictor.run_callbacks("on_predict_end")

    # NEW_JSON_STORAGE = {}

    # for k, v in JSON_STORAGE.items():
    # with open("worked_data.json", "w") as f:
    #
    #     json.dump(WORKED_JSON_STORAGE, f, ensure_ascii=False, indent=4)

    NEW_JSON_STORAGE = dict()
    NEW_WORKED_JSON_STORAGE = dict()

    for k, v in JSON_STORAGE.items():
        NEW_JSON_STORAGE[k] = list()

        result = list(zip(v[::2], v[1::2]))

        for res in result:
            tmp = {
                "object_uid": res[0]["object_uid"],
                "technic_type_id": res[0]["technic_type_id"],
                "title": res[0]["title"],
                "confidence": res[0]["confidence"],
                "worked_start_timestamp": res[0]["worked_start_timestamp"],
                "worked_end_timestamp": res[1]["worked_end_timestamp"],
                "frame_idx_start": res[0]["frame_idx"],
                "frame_idx_end": res[1]["frame_idx"],
                "object_coords": res[0]["object_coords"],
            }

            NEW_JSON_STORAGE[k].append(tmp)

        if len(v) % 2 == 1:
            last_elem = v.pop()
            tmp = {
                "object_uid": last_elem["object_uid"],
                "technic_type_id": last_elem["technic_type_id"],
                "title": last_elem["title"],
                "confidence": last_elem["confidence"],
                "worked_start_timestamp": last_elem["worked_start_timestamp"],
                "worked_end_timestamp": int(frame_idx / FPS),
                "frame_idx_start": last_elem["frame_idx"],
                "frame_idx_end": frame_idx,
                "object_coords": last_elem["object_coords"],
            }
            NEW_JSON_STORAGE[k].append(tmp)

    for k, v in WORKED_JSON_STORAGE.items():
        NEW_WORKED_JSON_STORAGE[k] = list()

        result = list(zip(v[::2], v[1::2]))

        for res in result:
            tmp = {
                "object_uid": res[0]["object_uid"],
                "technic_type_id": res[0]["technic_type_id"],
                "title": res[0]["title"],
                "confidence": res[0]["confidence"],
                "worked_start_timestamp": res[0]["worked_start_timestamp"],
                "worked_end_timestamp": res[1]["worked_end_timestamp"],
                "frame_idx_start": res[0]["frame_idx"],
                "frame_idx_end": res[1]["frame_idx"],
                "object_coords": res[0]["object_coords"],
            }

            NEW_WORKED_JSON_STORAGE[k].append(tmp)

        if len(v) % 2 == 1:
            last_elem = v.pop()
            tmp = {
                "object_uid": last_elem["object_uid"],
                "technic_type_id": last_elem["technic_type_id"],
                "title": last_elem["title"],
                "confidence": last_elem["confidence"],
                # "appearance_time": res[0]["appearance_time"],
                # "disappearance": res[0]["disappearance"],
                "worked_start_timestamp": last_elem["worked_start_timestamp"],
                "worked_end_timestamp": int(frame_idx / FPS),
                "frame_idx_start": last_elem["frame_idx"],
                "frame_idx_end": frame_idx,
                "object_coords": last_elem["object_coords"],
            }
            NEW_WORKED_JSON_STORAGE[k].append(tmp)

    with open("old_worked.json", "w") as f:
        # Записываем словарь в файл с отступами
        json.dump(NEW_JSON_STORAGE, f, ensure_ascii=False, indent=4)
    with open("new_worked.json", "w") as f:
        # Записываем словарь в файл с отступами
        json.dump(NEW_WORKED_JSON_STORAGE, f, ensure_ascii=False, indent=4)

    return "old_worked.json","new_worked.json", "outputs_with_boxes.webm"