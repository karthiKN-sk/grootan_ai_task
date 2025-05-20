import numpy as np
from constants.app_constants import COLORS
from cv.draw_polygen import compute_centroid
from cv.global_state import detections_state
import supervision as sv
from typing import Dict, Any

def annotate_frame(frame: np.ndarray, detections: sv.Detections, config: Dict[str, Any]) -> np.ndarray:
    frame_ = frame.copy()

    # Draw zones
    for i, ((zin_name, zin), (zout_name, zout)) in enumerate(zip(config["zones_in"].items(), config["zones_out"].items())):
        color = COLORS.colors[i % len(COLORS.colors)]
        zin_anchor = compute_centroid(zin.polygon)  
        zout_anchor = compute_centroid(zout.polygon)
        frame_ = sv.draw_polygon(frame_, zin.polygon, color)
        frame_ = sv.draw_text(frame_, text=zin_name, text_anchor=zin_anchor, text_color=color)
        frame_ = sv.draw_polygon(frame_, zout.polygon, color)
        frame_ = sv.draw_text(frame_, text=zout_name, text_anchor=zout_anchor, text_color=color)

    labels = [f"Car #{id_}" for id_ in detections.tracker_id]

    frame_ = config["trace_annotator"].annotate(frame_, detections)
    frame_ = config["box_annotator"].annotate(frame_, detections)
    frame_ = config["label_annotator"].annotate(frame_, detections, labels)

    # Count the different types of turns
    vehicle_turns= detections_state["vehicle_turns"]
    total_vehicles = len(vehicle_turns)
    right_turns = sum(1 for turns in vehicle_turns.values() if turns == "right_turn" )
    left_turns = sum(1 for turns in vehicle_turns.values() if turns == "left_turn" )
    u_turns = sum(1 for turns in vehicle_turns.values() if turns == "u_turn" )
    no_turns = sum(1 for turns in vehicle_turns.values() if turns == "straight")

    # Add detection count info
    total_count = len(detections)
    frame_ = sv.draw_text(
        frame_,
        f"Detected: {total_count}",
        sv.Point(50, 50),
        background_color=sv.Color.from_hex("#FF7F50")
    )
    # Draw fixed turn statistics on the center-left of the frame
    start_x = 80
    start_y = 350
    line_spacing = 40
    text_color = sv.Color(r=255, g=255, b=255) 

    # Line 1: Total vehicles tracked
    frame_ = sv.draw_text(
        frame_,
        text=f"Total vehicles tracked: {total_vehicles}",
        text_anchor=sv.Point(start_x + 30, start_y),
        background_color=sv.Color.from_hex("#DDDDDD"),
    )

    # Line 2: Right turns (Red)
    frame_ = sv.draw_text(
        frame_,
        text=f"Right turns: {right_turns}",
        text_anchor=sv.Point(start_x + 10, start_y + line_spacing),
        background_color=sv.Color(r=255, g=0, b=0),
        text_color=text_color
    )

    # Line 3: Left turns (Green)
    frame_ = sv.draw_text(
        frame_,
        text=f"Left turns: {left_turns}",
        text_anchor=sv.Point(start_x + 10, start_y + 2 * line_spacing),
        background_color=sv.Color(r=0, g=255, b=0),
       
    )

    # Line 4: U-turns (Black)
    frame_ = sv.draw_text(
        frame_,
        text=f"U-turns: {u_turns}",
        text_anchor=sv.Point(start_x + 10, start_y + 3 * line_spacing),
        background_color=sv.Color(r=0, g=0, b=0),
        text_color=text_color
    )

    # Line 5: No turns (Blue)
    frame_ = sv.draw_text(
        frame_,
        text=f"No turns: {no_turns}",
        text_anchor=sv.Point(start_x + 10, start_y + 4 * line_spacing),
        background_color=sv.Color(r=0, g=0, b=255),
        text_color=text_color
    )


    return frame_