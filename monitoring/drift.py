from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import pandas as pd

df = pd.read_csv("data/creditcard.csv")

train = df.sample(1000, random_state=42)
test = df.sample(1000, random_state=1)

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=train, current_data=test)

report.save_html("monitoring/drift_report.html")

print("✅ Drift report generated")