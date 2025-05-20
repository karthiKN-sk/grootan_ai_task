from cv.analysis_turn import analyze_turns
import cv2

def add_final_summary_to_video(video_path, vehicle_turns, output_path="final_output.mp4"):
    """Add a final summary frame to the end of the video"""

    # First analyze the turns
    vehicle_turn_json = analyze_turns(vehicle_turns)

    # Load the bar chart image
    chart_img = cv2.imread("turn_analysis.png")
    if chart_img is None:
        raise FileNotFoundError("turn_analysis.png not found.")

    # Read the original video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Resize chart to match video resolution
    chart_img = cv2.resize(chart_img, (width, height))

    # Create the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Copy all frames from the original video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    # Append chart image as 5 seconds of frames
    for _ in range(int(fps * 5)):
        out.write(chart_img)

    # Release resources
    cap.release()
    out.release()
    print(f"Final video with chart saved as '{output_path}'")
    
    return vehicle_turn_json