import json
import statistics


def calculate_beat_times(offsets, total_beats, use_most_common_interval=True):
    """
    Given a list of offsets (beats 0 to N) and the total number of beats,
    this function calculates the absolute times for each beat.

    If use_most_common_interval is True, the function finds the most common step
    and applies it uniformly. Otherwise, it cycles through the original interval pattern.
    """
    base = abs(offsets[0])
    abs_times = [round(base + off, 3) for off in offsets]

    # Calculate differences (intervals) between consecutive offsets
    intervals = [round(offsets[i + 1] - offsets[i], 3) for i in range(len(offsets) - 1)]

    # Store original intervals (commented in JSON output for later use)
    # print("Intervals:", intervals)  # Uncomment to check intervals manually

    # Find the most frequent interval
    most_common_interval = statistics.mode(intervals)

    # Start with the first actual beat
    first_beat_time = abs_times[-1]
    beat_times = [first_beat_time]

    pattern_index = 0

    while len(beat_times) < total_beats:
        if use_most_common_interval:
            next_time = round(beat_times[-1] + most_common_interval, 3)
        else:
            next_time = round(beat_times[-1] + intervals[pattern_index], 3)
            pattern_index = (pattern_index + 1) % len(intervals)

        beat_times.append(next_time)

    return beat_times, intervals, most_common_interval


def assign_scenes(beat_times, scenes):
    """Assigns scenes to beats, cycling through the given scenes list."""
    output = []
    for i, t in enumerate(beat_times):
        scene = scenes[i % len(scenes)]
        output.append({"time": t, "scene": scene})
    return output


if __name__ == "__main__":
    # Nine step offsets (beat positions relative to an offset reference)
    offsets = [-4.036, -3.591, -3.147, -2.702, -2.258, -1.813, -1.369, -0.924, -0.480, -0.036, 0.408]

    # Total number of beats in the song (example: 272 beats)
    total_beats = 456

    # Scene names (modify or extend this list as needed)
    scenes = ["straight_left", "straight_right"]

    # Calculate beat times using the offsets and the repeating pattern of intervals.
    beat_times, intervals, most_common_interval = calculate_beat_times(offsets, total_beats, use_most_common_interval=True)

    # Assign a scene to each beat (here, we simply cycle through the scenes list)
    beat_data = assign_scenes(beat_times, scenes)

    output_data = {
        "original_intervals": intervals,  # This is for reference, not used in calculations
        "most_common_interval": most_common_interval
    }

    print(output_data)

    # Write the result to a JSON file
    with open("beats.json", "w") as json_file:
        json.dump(beat_data, json_file, indent=4)

    # Print the JSON to the console (optional)
    print(json.dumps(beat_data, indent=4))
