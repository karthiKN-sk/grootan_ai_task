import tempfile
import streamlit as st
from src.cv.run_pipline import run_full_vehicle_turn_pipeline  
from src.llm.user_query_handling import convert_turn_stats_to_text,create_pipeline 
import os
import gradio as gr
import subprocess

# Persistent state for document store and pipeline
global_pipeline = None
global_turn_json = None

def analyze_video(video_file_path):
    global global_pipeline, global_turn_json

    if not video_file_path:
        return None, "Please upload a video file."

    with open(video_file_path, "rb") as source_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(source_file.read())
            tmp_video_path = tmp_file.name

    raw_output_path = "raw_output.mp4"
    browser_safe_path = "final_output_with_summary.mp4"

    # Run the full vehicle turn detection pipeline
    global_turn_json = run_full_vehicle_turn_pipeline(
        source_video_path=tmp_video_path,
        final_output_path=raw_output_path
    )
     
    # Re-encode video
    encode_to_browser_safe_mp4(raw_output_path, browser_safe_path)
    

    # Create document store and pipeline for QA
    text_data = convert_turn_stats_to_text(global_turn_json)
    global_pipeline = create_pipeline(text_data)
    

    return browser_safe_path, "Video analyzed successfully! You can now ask questions."

def answer_question(user_question):
    if not global_pipeline or not global_turn_json:
        return "Please analyze a video first."

    return global_pipeline(user_question)

# Gradio UI
video_input = gr.Video(label="Upload a video")
analyze_btn = gr.Button("Submit & Analyze")
video_output = gr.Video(label="Processed Video")
status_output = gr.Textbox(label="Status")

question_input = gr.Textbox(label="Ask a question (e.g., How many U-turns were made?)")
answer_output = gr.Textbox(label="Answer")

with gr.Blocks(title="Vehicle Turn Detection and QA") as demo:
    gr.Markdown("## Vehicle Turn Detection and Q&A")
    with gr.Row():
        with gr.Column():
            video_input.render()
            analyze_btn.render()
        with gr.Column():
            video_output.render()
            status_output.render()

    analyze_btn.click(fn=analyze_video, inputs=video_input, outputs=[video_output, status_output])

    gr.Markdown("### Ask a question after analysis:")
    question_input.render()
    answer_output.render()

    question_input.change(fn=answer_question, inputs=question_input, outputs=answer_output)


def encode_to_browser_safe_mp4(input_path: str, output_path: str):
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vcodec", "libx264",
        "-acodec", "aac",
        "-movflags", "+faststart",  
        output_path
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("Error: ffmpeg failed to convert video to browser-safe format.")


if __name__ == "__main__":
    demo.launch(debug=True)





