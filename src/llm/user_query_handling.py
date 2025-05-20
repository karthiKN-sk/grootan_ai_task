from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def convert_turn_stats_to_text(analysis_result):
    turn_counts = analysis_result.get("turn_counts", {})
    turn_details = analysis_result.get("turn_details", [])

    summary_text = (
        f"Total Vehicles Tracked: {analysis_result.get('total_vehicles', 0)}\n"
        f"Vehicles making right turns: {turn_counts.get('Vehicles making right turns', 0)}\n"
        f"Vehicles making left turns: {turn_counts.get('Vehicles making left turns', 0)}\n"
        f"Vehicles making U-turns: {turn_counts.get('Vehicles making U-turns', 0)}\n"
        f"Vehicles with no detected turns (Straight): {turn_counts.get('Vehicles with no detected turns (Straight)', 0)}\n"
    )

    detail_lines = [
        f"Tracker/Vehicle ID {item['tracker_id']}: {item['turn'].replace('_', ' ').capitalize()}"
        for item in turn_details
    ]

    details_text = "\n".join(detail_lines)

    return f"{summary_text}\nIndividual Vehicle Turns:\n{details_text}"



# Load Qwen model and tokenizer (only once globally)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
generation_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)


def create_pipeline(text_data):
    """
    Create a simple function to handle QA using Qwen with the full text_data
    """
    def qa_pipeline(question):
        messages = [
            {"role": "system", "content": "You are an expert on vehicle turn analysis."},
            {"role": "user", "content": f"Here is the analysis report:\n{text_data}"},
            {"role": "user", "content": question},
        ]
        response = generation_pipe(messages, max_new_tokens=100)[0]
        assistant_response = ""
        for msg in response['generated_text']:
            if msg.get("role") == "assistant":
                assistant_response = msg.get("content", "")
                break 
        return assistant_response

    return qa_pipeline
