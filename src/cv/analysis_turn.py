
import matplotlib.pyplot as plt

def analyze_turns(vehicle_turns):
    """Analyze the turns and create summary statistics"""
    total_vehicles = len(vehicle_turns)
    if total_vehicles == 0:
        print("No vehicles were detected or tracked.")
        return
    # Count the different types of turns
    right_turns = sum(1 for turns in vehicle_turns.values() if turns == "right_turn" )
    left_turns = sum(1 for turns in vehicle_turns.values() if turns == "left_turn" )
    u_turns = sum(1 for turns in vehicle_turns.values() if turns == "u_turn" )
    no_turns = sum(1 for turns in vehicle_turns.values() if turns == "straight")

    print("\n--- Turn Analysis Results ---")
    print(f"Total unique vehicles tracked: {total_vehicles}")
    print(f"Vehicles making right turns: {right_turns} ")
    print(f"Vehicles making left turns: {left_turns}")
    print(f"Vehicles making U-turns: {u_turns}")
    print(f"Vehicles with no detected turns: {no_turns}")


    # Create a visualization
    turn_counts = {
        'Right Turn': right_turns,
        'Left Turn': left_turns,
        'U-Turn': u_turns,
        'No Turn': no_turns
    }

    turn_message = {
    "message": "Turn Analysis Results completed.",
    "total_vehicles": total_vehicles,
    "turn_counts": {
        "Vehicles making right turns" : right_turns,
        "Vehicles making left turns": left_turns,
        "Vehicles making U-turns": u_turns,
        "Vehicles with no detected turns (Straight)": no_turns
    },
     "turn_details": [
        {"tracker_id": tracker_id, "turn": turn}
        for tracker_id, turn in vehicle_turns.items()
    ]
    }

    plt.figure(figsize=(10, 6))
    colors = ['red', 'green', 'black', 'blue']
    plt.bar(turn_counts.keys(), turn_counts.values(), color=colors)
    plt.title('Vehicle Turn Analysis')
    plt.ylabel('Number of Vehicles')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add count labels on top of each bar
    for i, (key, value) in enumerate(turn_counts.items()):
        plt.text(i, value + 0.3, str(value), ha='center')

    plt.savefig('turn_analysis.png')
    # plt.show()
    return turn_message