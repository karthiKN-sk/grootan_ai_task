from cv.global_state import trained_model,tracker, update_detections_state
from cv.draw_polygen import initiate_polygon_zones
from constants.app_constants import COLORS
from typing import Optional, Dict, Any
import supervision as sv
from constants.app_constants import ZONE_IN_NAMES,ZONE_IN_POLYGONS,ZONE_OUT_NAMES,ZONE_OUT_POLYGONS


def setup_video_processor(
    source_video_path: str,
    target_video_path: Optional[str] = None,
    confidence_threshold: float = 0.4,
    iou_threshold: float = 0.7,
) -> Dict[str, Any]:
    return {
        "conf_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "source_video_path": source_video_path,
        "target_video_path": target_video_path,
        "model": trained_model,
        "tracker": tracker,
        "video_info": sv.VideoInfo.from_video_path(source_video_path),
        "zones_in": initiate_polygon_zones(ZONE_IN_NAMES,ZONE_IN_POLYGONS),
        "zones_out": initiate_polygon_zones(ZONE_OUT_NAMES,ZONE_OUT_POLYGONS),
        "box_annotator": sv.BoxAnnotator(color=COLORS),
        "label_annotator": sv.LabelAnnotator(color=COLORS, text_color=sv.Color.BLACK),
        "trace_annotator": sv.TraceAnnotator(
            color=COLORS, position=sv.Position.CENTER, trace_length=100, thickness=2
        ),
        "detections_manager": update_detections_state,
    }