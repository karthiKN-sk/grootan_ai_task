from cv.global_state import detections_state
from cv.annotate_video_frame import annotate_frame
import supervision as sv
from typing import Dict, Any
import numpy as np

def process_frame(frame: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    result = config["model"](frame, verbose=False, conf=config["conf_threshold"], iou=config["iou_threshold"])[0]
    detections = sv.Detections.from_ultralytics(result)
    detections.class_id = np.zeros(len(detections))
    detections = config["tracker"].update_with_detections(detections)
    detections_in_zones, detections_out_zones = [], []
    for zone_in, zone_out in zip(config["zones_in"].values(), config["zones_out"].values()):
        in_zone = detections[zone_in.trigger(detections)]
        out_zone = detections[zone_out.trigger(detections)]
        detections_in_zones.append(in_zone)
        detections_out_zones.append(out_zone)

    filtered = config["detections_manager"](detections, detections_in_zones, detections_out_zones,config)
    return annotate_frame(frame, filtered, config)