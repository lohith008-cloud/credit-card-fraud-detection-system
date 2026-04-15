import csv
import os

def log_prediction(data, prob, pred):
    os.makedirs("logs", exist_ok=True)

    file = "logs/predictions.csv"
    write_header = not os.path.exists(file)

    with open(file, "a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow(list(data.keys()) + ["probability", "prediction"])

        writer.writerow(list(data.values()) + [prob, pred])