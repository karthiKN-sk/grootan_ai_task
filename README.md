# Grootan AI Task

This project is designed to detect, track, and analyze vehicle turns (left, right, U-turn, straight) using a custom-trained YOLO model, entry/exit zones, and frame-by-frame analysis. It outputs annotated videos, turn statistics, and optional user query handling via an LLM interface.

---

## 📁 Project Structure

```plaintext
├── models/
│   ├── YoloFineTunedV1.pt                               # Custom-trained YOLO model
    └── YoloFineTunedV2.pt   
├── annotated_output.png                                 # Sample annotated video frame (output)
├── turn_analysis.png                                    # Visualization of turn statistics (output)
├── main.py                                              # Main entry point (Start UI)
├── src/  
│     └── cv/
│     │    ├── Detection_State_Tracker.py                # Global Detection State and Vehicle Entry-Exit Tracking and Turn Classification Logic
│     │    ├── Draw_Polygen.py                           # Helper functions (centroid, zone initiation).
│     │    └── Vehicle_Turn_Pipeline.py                  # Full Video-Based Vehicle Turn Detection and Pipeline for Vehicle Turn Detection
│     │ 
│     └── llm/
│          └── user_query_handling.py                    # Vehicle Turn Detection Summary & AI-Powered Question Answering.
│
├── notebook/
│       ├── video_vehicle_analysis_grootan_task.ipynb    # Main Notebook contain Code We Can run directly in Google Colab
│       └── FineTuning_Yolo.ipynb.ipynb                  # Notebook Used to FineTuning Yolo Model
│
└──  constants/
        └── app_constants.pt                             # Configuration for Zone Definitions, Color Palette, and Vehicle Turn Classification

```


**▶️ How to Run**

1. Git Clone from Branch 

```
!git clone -b Class_Code_branch  https://github.com/karthiKN-sk/grootan_ai_task.git
```

2. Install dependencies

```
!pip install -r /content/grootan_ai_task/requirement.txt
```
2. Run UI Gradio

```
!python main.py
```


