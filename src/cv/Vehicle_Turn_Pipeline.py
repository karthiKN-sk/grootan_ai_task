import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any
from rich.progress import Progress
import supervision as sv
from ultralytics import YOLO
from constants.app_constants import ZONE_IN_NAMES,ZONE_IN_POLYGONS,ZONE_OUT_NAMES,ZONE_OUT_POLYGONS,COLORS
from cv.Draw_Polygon import initiate_polygon_zones,compute_centroid
from cv.Detection_State_Tracker import DetectionStateTracker

class VehicleTurnPipeline:
    def __init__(
        self,
        source_video_path: str,
        target_video_path: Optional[str] = "output_traced.mp4",
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.7,
    ):
        self.config = {
            "conf_threshold": confidence_threshold,
            "iou_threshold": iou_threshold,
            "source_video_path": source_video_path,
            "target_video_path": target_video_path,
            "model": YOLO("/content/grootan_ai_task/models/YoloFineTunedV1.pt"),
            "tracker": sv.ByteTrack(),
            "video_info": sv.VideoInfo.from_video_path(source_video_path),
            "zones_in": initiate_polygon_zones(ZONE_IN_NAMES, ZONE_IN_POLYGONS),
            "zones_out": initiate_polygon_zones(ZONE_OUT_NAMES, ZONE_OUT_POLYGONS),
            "box_annotator": sv.BoxAnnotator(color=COLORS),
            "label_annotator": sv.LabelAnnotator(color=COLORS, text_color=sv.Color.BLACK),
            "trace_annotator": sv.TraceAnnotator(
                color=COLORS, position=sv.Position.CENTER, trace_length=100, thickness=2
            ),
        }
        self.detections_state_tracker = DetectionStateTracker()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        result = self.config["model"](
            frame, verbose=False, conf=self.config["conf_threshold"], iou=self.config["iou_threshold"]
        )[0]
        detections = sv.Detections.from_ultralytics(result)
        detections.class_id = np.zeros(len(detections))
        detections = self.config["tracker"].update_with_detections(detections)

        detections_in_zones, detections_out_zones = [], []
        for zone_in, zone_out in zip(self.config["zones_in"].values(), self.config["zones_out"].values()):
            in_zone = detections[zone_in.trigger(detections)]
            out_zone = detections[zone_out.trigger(detections)]
            detections_in_zones.append(in_zone)
            detections_out_zones.append(out_zone)

        filtered = self.detections_state_tracker.update(
            detections, detections_in_zones, detections_out_zones, self.config
        )
        return self.annotate_frame(frame, filtered)

    def annotate_frame(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        frame_ = frame.copy()

        for i, ((zin_name, zin), (zout_name, zout)) in enumerate(
            zip(self.config["zones_in"].items(), self.config["zones_out"].items())
        ):
            color = COLORS.colors[i % len(COLORS.colors)]
            frame_ = sv.draw_polygon(frame_, zin.polygon, color)
            frame_ = sv.draw_text(frame_, zin_name, compute_centroid(zin.polygon), text_color=color)
            frame_ = sv.draw_polygon(frame_, zout.polygon, color)
            frame_ = sv.draw_text(frame_, zout_name, compute_centroid(zout.polygon), text_color=color)

        labels = [f"Car #{id_}" for id_ in detections.tracker_id]

        frame_ = self.config["trace_annotator"].annotate(frame_, detections)
        frame_ = self.config["box_annotator"].annotate(frame_, detections)
        frame_ = self.config["label_annotator"].annotate(frame_, detections, labels)

        vehicle_turns = self.detections_state_tracker.get_vehicle_turns()
        turn_stats = self._get_turn_statistics(vehicle_turns)
        start_x, start_y, spacing = 80, 350, 40

        text_lines = [
            ("Total vehicles tracked", turn_stats["total"], "#DDDDDD"),
            ("Right turns", turn_stats["right_turn"], "#FF0000"),
            ("Left turns", turn_stats["left_turn"], "#00FF00"),
            ("U-turns", turn_stats["u_turn"], "#000000"),
            ("No turns", turn_stats["straight"], "#0000FF"),
        ]

        for i, (label, count, color) in enumerate(text_lines):
            frame_ = sv.draw_text(
                frame_,
                text=f"{label}: {count}",
                text_anchor=sv.Point(start_x + 10, start_y + i * spacing),
                background_color=sv.Color.from_hex(color),
                text_color=sv.Color(r=255, g=255, b=255) if color != "#DDDDDD" else sv.Color.BLACK
            )

        return frame_

    def _get_turn_statistics(self, vehicle_turns: Dict[int, str]) -> Dict[str, int]:
        return {
            "total": len(vehicle_turns),
            "right_turn": sum(1 for t in vehicle_turns.values() if t == "right_turn"),
            "left_turn": sum(1 for t in vehicle_turns.values() if t == "left_turn"),
            "u_turn": sum(1 for t in vehicle_turns.values() if t == "u_turn"),
            "straight": sum(1 for t in vehicle_turns.values() if t == "straight"),
        }

    def process_video(self) -> Dict[int, str]:
        frame_generator = sv.get_video_frames_generator(self.config["source_video_path"])
        total_frames = self.config["video_info"].total_frames

        with Progress() as progress:
            task = progress.add_task("[green]Processing video...", total=total_frames)

            if self.config["target_video_path"]:
                with sv.VideoSink(self.config["target_video_path"], self.config["video_info"]) as sink:
                    saved_sample = False
                    for frame in frame_generator:
                        annotated = self.process_frame(frame)
                        sink.write_frame(annotated)
                        if not saved_sample:
                            cv2.imwrite("annotated_output.png", annotated)
                            saved_sample = True
                        progress.advance(task)
            else:
                for frame in frame_generator:
                    annotated = self.process_frame(frame)
                    cv2.imshow("Processed Video", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    progress.advance(task)
                cv2.destroyAllWindows()

        return self.detections_state_tracker.get_vehicle_turns()

    def analyze_turns(self, vehicle_turns: Dict[int, str]) -> Dict[str, Any]:
        stats = self._get_turn_statistics(vehicle_turns)

        print("\n--- Turn Analysis Results ---")
        for key, value in stats.items():
            label = key.replace("_", " ").capitalize()
            print(f"{label}: {value}")

        plt.figure(figsize=(10, 6))
        plt.bar(
            ["Right Turn", "Left Turn", "U-Turn", "No Turn"],
            [stats["right_turn"], stats["left_turn"], stats["u_turn"], stats["straight"]],
            color=["red", "green", "black", "blue"]
        )
        plt.title('Vehicle Turn Analysis')
        plt.ylabel('Number of Vehicles')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        for i, value in enumerate([stats["right_turn"], stats["left_turn"], stats["u_turn"], stats["straight"]]):
            plt.text(i, value + 0.3, str(value), ha='center')
        plt.savefig('turn_analysis.png')
        plt.show()
        plt.close()

        return {
            "message": "Turn Analysis Results completed.",
            "total_vehicles": stats["total"],
            "turn_counts": stats,
            "turn_details": [{"tracker_id": k, "turn": v} for k, v in vehicle_turns.items()]
        }

    def append_summary_to_video(self, vehicle_turns: Dict[int, str], output_path: str = "final_output.mp4") -> Dict[str, Any]:
        analysis = self.analyze_turns(vehicle_turns)
        cap = cv2.VideoCapture(self.config["target_video_path"])
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        chart_img = cv2.imread("turn_analysis.png")
        if chart_img is None:
            raise FileNotFoundError("turn_analysis.png not found.")
        chart_img = cv2.resize(chart_img, (width, height))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        for _ in range(int(fps * 5)):
            out.write(chart_img)

        cap.release()
        out.release()
        print(f"Final video with chart saved as '{output_path}'")
        return analysis

    def run_pipeline(self, final_output_path: str = "final_output.mp4") -> Dict[str, Any]:
        vehicle_turns = self.process_video()
        return self.append_summary_to_video(vehicle_turns, final_output_path)
