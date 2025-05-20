from cv.add_final_summary_to_video import add_final_summary_to_video
from cv.setup_video_processer import setup_video_processor
from cv.video_process import process_video

def run_full_vehicle_turn_pipeline(
    source_video_path: str,
    final_output_path: str = "final_output-sat1.mp4"
):
    """
    Runs the full pipeline: processes video, tracks turns, and appends summary.
    """
    # Step 1: Setup and process the video
    config = setup_video_processor(
        source_video_path=source_video_path,
        target_video_path="output_traced.mp4"
    )
    vehicle_turns_state = process_video(config)

    # Step 2: Append summary chart to the traced video
    vehicle_turn_json = add_final_summary_to_video(
        video_path="output_traced.mp4",
        vehicle_turns=vehicle_turns_state,
        output_path=final_output_path
    )
    return vehicle_turn_json
