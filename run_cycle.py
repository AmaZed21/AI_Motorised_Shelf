import csv
import random
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from simulator import (
    Compartment,
    LABEL_MANUAL_STOP,
    LABEL_NORMAL,
    LABEL_OBSTRUCTION,
    LABEL_OVERLOAD,
    STATE_FAULT,
    STATE_STOPPED,
)

FIELDS = [
    "compartment_no",
    "position_cm",
    "speed_cm_s",
    "motor_current_a",
    "sensor_distance_cm",
    "weight_kg",
    "state",
    "actual_condition",
    "event_type",
]


def generate_training_csv(
    output_csv: str = "data/training_sensor_data.csv",
    episodes_per_class: int = 200,
    dt: float = 0.1,
    seed: int = 42,
) -> dict[str, int]:
    if episodes_per_class < 1:
        raise ValueError("episodes_per_class must be at least 1")
    if dt <= 0:
        raise ValueError("dt must be greater than 0")

    rng = random.Random(seed)
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)

    compartments = [
        Compartment(com_no=1, weight=0.4, contents=["inhaler"]),
        Compartment(com_no=2, weight=0.2, contents=["bottle"]),
        Compartment(com_no=3, weight=0.5, contents=["keys", "medicine"]),
        ]
    simulated_time = datetime.now().replace(microsecond=0)
    counts: Counter[str] = Counter()

    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()

        def log_sample(event_type: str) -> None:
            nonlocal simulated_time
            writer.writerow({
                "compartment_no": compartment.com_no,
                "position_cm": round(compartment.position, 3),
                "speed_cm_s": round(compartment.speed, 3),
                "motor_current_a": round(compartment.motor_current, 3),
                "sensor_distance_cm": round(compartment.sensor_distance, 3),
                "weight_kg": round(compartment.weight, 3),
                "state": compartment.state,
                "actual_condition": compartment.label,
                "event_type": event_type,
            })
            counts[compartment.label] += 1
            simulated_time += timedelta(seconds=dt)

        def tick(steps: int, event_type: str = "TICK") -> None:
            for _ in range(steps):
                compartment.update(dt)
                log_sample(event_type)

        def prepare_episode() -> None:
            compartment.clear_fault()
            compartment.position = rng.uniform(20.0, compartment.MAX_HEIGHT)
            compartment.sensor_distance = compartment.position + 2.0
            compartment.weight = rng.uniform(0.05, 0.9)
            compartment.label = LABEL_NORMAL
            compartment.move_down()

        scenarios = [
        (compartment, condition)
        for compartment in compartments
        for condition in (
            LABEL_NORMAL,
            LABEL_OBSTRUCTION,
            LABEL_OVERLOAD,
            LABEL_MANUAL_STOP,
        )
        for _ in range(episodes_per_class)
        ]
        rng.shuffle(scenarios)

        for compartment, condition in scenarios:
            prepare_episode()
            warmup_steps = rng.randint(5, 25)
            tick(warmup_steps)

            if condition == LABEL_NORMAL:
                tick(rng.randint(10, 50))
                compartment.stop(label=LABEL_NORMAL)
                log_sample("NORMAL_STOP")

            elif condition == LABEL_OBSTRUCTION:
                compartment.inject_obstruction()
                log_sample("OBSTRUCTION_INJECTED")
                while compartment.state != STATE_FAULT:
                    tick(1, "OBSTRUCTION_RESPONSE")
                tick(5, "OBSTRUCTION_FAULT")

            elif condition == LABEL_OVERLOAD:
                compartment.inject_overload()
                log_sample("OVERLOAD_INJECTED")
                while compartment.state != STATE_FAULT:
                    tick(1, "OVERLOAD_RESPONSE")
                tick(5, "OVERLOAD_FAULT")

            else:  # LABEL_MANUAL_STOP
                compartment.stop(label=LABEL_MANUAL_STOP)
                log_sample("MANUAL_STOP")
                # Include a few stationary samples after a commanded stop.
                tick(5, "MANUAL_STOPPED")

            # Do not carry a fault state into the next episode.
            if compartment.state == STATE_FAULT:
                compartment.clear_fault()
            elif compartment.state != STATE_STOPPED:
                compartment.stop(label=LABEL_NORMAL)

    return dict(counts)


if __name__ == "__main__":
    summary = generate_training_csv()
    print("Training CSV created: data/training_sensor_data.csv")
    print("Rows by condition:", summary)