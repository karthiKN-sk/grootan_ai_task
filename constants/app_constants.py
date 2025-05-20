import numpy as np
import supervision as sv


COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])

ZONE_IN_POLYGONS = [
    np.array([[1220, 360], [1500,600], [1600, 420], [1350, 200]]),
    np.array([[680, 260], [820, 380], [1020, 220], [880, 80]]),
    np.array([[480, 700], [680, 900], [800, 800], [660, 600]]),
    np.array([[1180, 1060], [1380, 880], [1200, 740], [1050, 1010]]),
]

ZONE_OUT_POLYGONS = [
    np.array([[950, 120], [1200, 350], [1340, 180], [1160, 20]]),
    np.array([[650, 260], [420, 480], [560, 620], [820, 400]]),
    np.array([[680, 920], [850, 1050], [1050, 1000], [820, 800]]),
    np.array([[1380, 860],[1220, 720],[1500, 620],[1640, 720],]),
]

ZONE_IN_NAMES = ["In1", "In2", "In3", "In4"]
ZONE_OUT_NAMES = ["Out1", "Out2", "Out3", "Out4"]

def get_color_for_turn(tracker_id, vehicle_turns):
    turn = vehicle_turns.get(tracker_id)
    if turn == "right_turn":
        return sv.Color.RED
    elif turn == "left_turn":
        return sv.Color.GREEN
    elif turn == "u_turn":
        return sv.Color.BLACK
    else:
        return sv.Color.BLUE

TURN_MAPPING = {
    "In1": {"Out2": "right_turn", "Out4": "left_turn", "Out3": "straight", "Out1": "u_turn"},
    "In2": {"Out3": "right_turn", "Out1": "left_turn", "Out4": "straight", "Out2": "u_turn"},
    "In3": {"Out4": "right_turn", "Out2": "left_turn", "Out1": "straight", "Out3": "u_turn"},
    "In4": {"Out1": "right_turn", "Out3": "left_turn", "Out2": "straight", "Out4": "u_turn"},
}

