from ultralytics import YOLO
import supervision as sv
from typing import Dict, List, Any
import numpy as np

from constants.app_constants import TURN_MAPPING



YOLO_MODEL_PATH = "models/YoloFineTunedV1.pt"
YOLO_MODEL_PATH2 = "models/YoloFineTunedV2.pt"

# Load a COCO-pretrained YOLO model
trained_model = YOLO(YOLO_MODEL_PATH)

# Display model information (optional)
trained_model.info()

# Create a dedicated tracker for vehicles
tracker = sv.ByteTrack()




detections_state = {
    "tracker_id_to_zone_id": {},
    "vehicle_paths": {},
    "vehicle_turns": {},
}



def update_detections_state(
    detections_all: sv.Detections,
    detections_in_zones: List[sv.Detections],
    detections_out_zones: List[sv.Detections],
    config: Dict[str, Any],
    state: Dict[str, Any] = detections_state
) -> sv.Detections:
    tracker_id_to_zone_id = state["tracker_id_to_zone_id"]
    vehicle_paths = state["vehicle_paths"]
    vehicle_turns = state.setdefault("vehicle_turns", {})

    # --- Assign entry zones and track vehicle IN zone names ---
    zone_in_names = list(config["zones_in"].keys())  # cache the keys list
    for zone_in_id, detections_in_zone in enumerate(detections_in_zones):
        zone_name = zone_in_names[zone_in_id]  
        for tracker_id in detections_in_zone.tracker_id:
            tracker_id_to_zone_id.setdefault(tracker_id, zone_in_id)

            # Initialize vehicle path if not already present
            vehicle_paths.setdefault(tracker_id, {"in": None, "out": None})
            if vehicle_paths[tracker_id]["in"] is None:
                vehicle_paths[tracker_id]["in"] = zone_name

    # --- Count exits grouped by entry zone and track vehicle OUT zone names ---
    zone_out_names = list(config["zones_out"].keys())  
    for zone_out_id, detections_out_zone in enumerate(detections_out_zones):
        zone_name = zone_out_names[zone_out_id]  
        for tracker_id in detections_out_zone.tracker_id:
            if tracker_id in tracker_id_to_zone_id:
                vehicle_paths.setdefault(tracker_id, {"in": None, "out": None})
                if vehicle_paths[tracker_id]["out"] is None:
                    vehicle_paths[tracker_id]["out"] = zone_name

    # --- Detect turns ---
    for tracker_id, path in vehicle_paths.items():
        in_zone = path["in"]
        out_zone = path["out"]
        if in_zone and out_zone and tracker_id not in vehicle_turns:
            turn_type = TURN_MAPPING.get(in_zone, {}).get(out_zone)
            if turn_type:
                vehicle_turns[tracker_id] = turn_type

    # Assign class_id for drawing/annotation
    if len(detections_all) > 0:
        detections_all.class_id = np.vectorize(
            lambda x: tracker_id_to_zone_id.get(x, -1)
        )(detections_all.tracker_id)
    else:
        detections_all.class_id = np.array([], dtype=int)

    return detections_all[detections_all.class_id != -1]
