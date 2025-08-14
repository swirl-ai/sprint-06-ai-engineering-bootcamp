from langsmith import Client
from time import sleep

from src.api.rag.agents import coordinator_agent_node
from src.api.rag.graph import State
from src.api.core.config import config


ACC_THRESHOLD = 0.9
SLEEP_TIME = 5
EVAL_DATASET_NAME = "coordinator-evaluation-dataset-01"

ls_client = Client(api_key=config.LANGSMITH_API_KEY)


def next_agent_evaluator_gpt_4_1(run, example):

    next_agent_match = run.outputs["next_agent"] == example.outputs["next_agent"]
    final_answer_match = run.outputs["coordinator_final_answer"] == example.outputs["coordinator_final_answer"]

    return next_agent_match and final_answer_match


def next_agent_evaluator_gpt_4_1_mini(run, example):

    next_agent_match = run.outputs["next_agent"] == example.outputs["next_agent"]
    final_answer_match = run.outputs["coordinator_final_answer"] == example.outputs["coordinator_final_answer"]

    return next_agent_match and final_answer_match


def next_agent_evaluator_groq_llama_3_3_70b_versatile(run, example):

    next_agent_match = run.outputs["next_agent"] == example.outputs["next_agent"]
    final_answer_match = run.outputs["coordinator_final_answer"] == example.outputs["coordinator_final_answer"]

    return next_agent_match and final_answer_match


results_gpt_4_1 = ls_client.evaluate(
    lambda x: coordinator_agent_node(State(messages=x["messages"]), models=["gpt-4.1"]),
    data=EVAL_DATASET_NAME,
    evaluators=[
        next_agent_evaluator_gpt_4_1
    ],
    experiment_prefix="gpt-4.1",
    num_repetitions=1
)

results_gpt_4_1_mini = ls_client.evaluate(
    lambda x: coordinator_agent_node(State(messages=x["messages"]), models=["gpt-4.1-mini"]),
    data=EVAL_DATASET_NAME,
    evaluators=[
        next_agent_evaluator_gpt_4_1_mini
    ],
    experiment_prefix="gpt-4.1-mini",
    num_repetitions=1
)


results_groq_llama_3_3_70b_versatile = ls_client.evaluate(
    lambda x: coordinator_agent_node(State(messages=x["messages"]), models=["groq/llama-3.3-70b-versatile"]),
    data=EVAL_DATASET_NAME,
    evaluators=[
        next_agent_evaluator_groq_llama_3_3_70b_versatile
    ],
    experiment_prefix="groq/llama-3.3-70b-versatile",
    num_repetitions=1
)

print(f"Sleeping for {SLEEP_TIME} seconds...")
sleep(SLEEP_TIME)

results_resp_gpt_4_1 = ls_client.read_project(
    project_name=results_gpt_4_1.experiment_name, include_stats=True
)

results_resp_gpt_4_1_mini = ls_client.read_project(
    project_name=results_gpt_4_1_mini.experiment_name, include_stats=True
)

results_resp_groq_llama_3_3_70b_versatile = ls_client.read_project(
    project_name=results_groq_llama_3_3_70b_versatile.experiment_name, include_stats=True
)


output_message = "\n"

avg_metrics = []
error_count = 0

for result in zip(
    [results_resp_gpt_4_1, results_resp_gpt_4_1_mini, results_resp_groq_llama_3_3_70b_versatile],
    ["next_agent_evaluator_gpt_4_1", "next_agent_evaluator_gpt_4_1_mini", "next_agent_evaluator_groq_llama_3_3_70b_versatile"]
):

    avg_metric = result[0].feedback_stats[result[1]]["avg"]
    avg_metrics.append(avg_metric)
    error_count += result[0].feedback_stats[result[1]]["errors"]

    if avg_metric >= ACC_THRESHOLD:
        output_message += f"✅ {result[1]} - Success: {avg_metric}\n"
    else:
        output_message += f"❌ {result[1]} - Failure: {avg_metric}\n"

if error_count > 0:
    raise AssertionError(f"There were {error_count} errors while running evaluations.")
elif all(metric >= ACC_THRESHOLD for metric in avg_metrics):
    print(output_message, flush = True)
else:
    raise AssertionError(output_message)

