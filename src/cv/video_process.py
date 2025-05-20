from rich.progress import Progress
from typing import Dict, Any
import supervision as sv
import cv2
from cv.global_state import detections_state
from cv.process_video_frame import process_frame

def process_video(config: Dict[str, Any]) -> None:
    frame_generator = sv.get_video_frames_generator(config["source_video_path"])
    total = config["video_info"].total_frames

    with Progress() as progress:
        task = progress.add_task("[green]Processing video...", total=total)

        if config["target_video_path"]:
            with sv.VideoSink(config["target_video_path"], config["video_info"]) as sink:
                saved_sample = False
                for frame in frame_generator:
                    annotated = process_frame(frame, config)
                    sink.write_frame(annotated)
                    if not saved_sample:
                        cv2.imwrite("annotated_output.png", annotated)
                        saved_sample = True
                    progress.advance(task)
        else:
            for frame in frame_generator:
                annotated = process_frame(frame, config)
                cv2.imshow("Processed Video", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                progress.advance(task)
            cv2.destroyAllWindows()
  
    return detections_state["vehicle_turns"]