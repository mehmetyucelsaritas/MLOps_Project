import wandb
import json
import os
from dotenv import load_dotenv

# --------------------
# Load files
# --------------------
evaluation_results_path = "outputs/evaluation_results/evaluation_results_2026-01-12_M7.json"
response_time_path = "outputs/evaluation_results/response_times_2026-01-12_M7.json"

load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))

with open(evaluation_results_path, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(response_time_path, "r", encoding="utf-8") as f:
    timing = json.load(f)

# --------------------
# Init run
# --------------------
wandb.init(
    project="FireGPT",
    name="RAG-eval",
    config={
        "model": "gemini-2.0-flash-lite",
        "temperature": 0.0,
        "top_p": 1.0,
    },
)


# --------------------
# Helper function
# --------------------
def log_bar_chart(metric_name: str, metric_dict: dict):
    table = wandb.Table(columns=["query_id", metric_name])

    for qid, value in metric_dict.items():
        table.add_data(int(qid), float(value))

    wandb.log(
        {
            f"{metric_name}_bar": wandb.plot.bar(
                table,
                "query_id",
                metric_name,
                title=f"{metric_name} per query",
            )
        }
    )


# --------------------
# Log metrics
# --------------------
log_bar_chart("llm_context_precision_without_reference", data["llm_context_precision_without_reference"])

log_bar_chart("faithfulness", data["faithfulness"])

log_bar_chart("answer_relevancy", data["answer_relevancy"])

# --------------------
# Response time bar chart
# --------------------
rt_table = wandb.Table(columns=["query_id", "response_time_sec"])
for i, item in enumerate(timing):
    rt_table.add_data(i, item["response_time_sec"])

wandb.log({"response_time_bar": wandb.plot.bar(rt_table, "query_id", "response_time_sec", title="Response Time per Query (sec)")})

wandb.finish()
