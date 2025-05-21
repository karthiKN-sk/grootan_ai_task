from typing import List, Dict, Any
import numpy as np
import supervision as sv  
from constants.app_constants import TURN_MAPPING

class DetectionStateTracker:
    def __init__(self):
        self.tracker_id_to_zone_id: Dict[int, int] = {}
        self.vehicle_paths: Dict[int, Dict[str, str]] = {}
        self.vehicle_turns: Dict[int, str] = {}

    def update(
        self,
        detections_all: sv.Detections,
        detections_in_zones: List[sv.Detections],
        detections_out_zones: List[sv.Detections],
        config: Dict[str, Any]
    ) -> sv.Detections:
        zone_in_names = list(config["zones_in"].keys())
        zone_out_names = list(config["zones_out"].keys())

        # --- Track vehicles entering IN zones ---
        for zone_in_id, detections_in_zone in enumerate(detections_in_zones):
            zone_name = zone_in_names[zone_in_id]
            for tracker_id in detections_in_zone.tracker_id:
                self.tracker_id_to_zone_id.setdefault(tracker_id, zone_in_id)
                self.vehicle_paths.setdefault(tracker_id, {"in": None, "out": None})
                if self.vehicle_paths[tracker_id]["in"] is None:
                    self.vehicle_paths[tracker_id]["in"] = zone_name

        # --- Track vehicles exiting OUT zones ---
        for zone_out_id, detections_out_zone in enumerate(detections_out_zones):
            zone_name = zone_out_names[zone_out_id]
            for tracker_id in detections_out_zone.tracker_id:
                if tracker_id in self.tracker_id_to_zone_id:
                    self.vehicle_paths.setdefault(tracker_id, {"in": None, "out": None})
                    if self.vehicle_paths[tracker_id]["out"] is None:
                        self.vehicle_paths[tracker_id]["out"] = zone_name

        # --- Detect turns ---
        for tracker_id, path in self.vehicle_paths.items():
            in_zone = path["in"]
            out_zone = path["out"]
            if in_zone and out_zone and tracker_id not in self.vehicle_turns:
                turn_type = TURN_MAPPING.get(in_zone, {}).get(out_zone)
                if turn_type:
                    self.vehicle_turns[tracker_id] = turn_type

        # --- Annotate detections for visualization ---
        if len(detections_all) > 0:
            detections_all.class_id = np.vectorize(
                lambda x: self.tracker_id_to_zone_id.get(x, -1)
            )(detections_all.tracker_id)
        else:
            detections_all.class_id = np.array([], dtype=int)

        return detections_all[detections_all.class_id != -1]

    def get_vehicle_turns(self) -> Dict[int, str]:
        return self.vehicle_turns

    def reset(self):
        self.tracker_id_to_zone_id.clear()
        self.vehicle_paths.clear()
        self.vehicle_turns.clear()
