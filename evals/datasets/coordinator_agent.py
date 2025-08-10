from langsmith import Client
import os

DATASET_NAME = "coordinator-evaluation-dataset"

coordinator_eval_dataset = [
    {
        "inputs": {
            "messages": [
                {"role": "user", "content": "Whats is the weather today?"}
            ]
        },
        "outputs": {
            "next_agent": "",
            "coordinator_final_answer": True
        }
    },
    {
        "inputs": {
            "messages": [
                {"role": "user", "content": "Can I get some earphones?"}
            ]
        },
        "outputs": {
            "next_agent": "product_qa_agent",
            "coordinator_final_answer": False
        }
    },
    {
        "inputs": {
            "messages": [
                {"role": "user", "content": "Can you add an item with ID B09NLTDHQ6 to my cart?"}
            ]
        },
        "outputs": {
            "next_agent": "shopping_cart_agent",
            "coordinator_final_answer": False
        }
    },
    {
        "inputs": {
            "messages": [
                {"role": "user", "content": "Can you add those earphones to my cart?"}
            ]
        },
        "outputs": {
            "next_agent": "",
            "coordinator_final_answer": True
        }
    },
    {
        "inputs": {
            "messages": [
                {"role": "user", "content": "Can you add the best items to my cart? I am looking for laptop bags."}
            ]
        },
        "outputs": {
            "next_agent": "product_qa_agent",
            "coordinator_final_answer": False
        }
    },
    {
        "inputs": {
            "messages": [
                {"role": "user", "content": "Can you find some good reviews for items in my cart?"}
            ]
        },
        "outputs": {
            "next_agent": "shopping_cart_agent",
            "coordinator_final_answer": False
        }
    },
    {
        "inputs": {
            "messages": [
                {"role": "user", "content": "Can you put the items with the most positive user reviews to my cart?"}
            ]
        },
        "outputs": {
            "next_agent": "product_qa_agent",
            "coordinator_final_answer": False
        }
    },
    {
        "inputs": {
            "messages": [
                {"role": "user", "content": "What kind of stuff do you sell?"}
            ]
        },
        "outputs": {
            "next_agent": "",
            "coordinator_final_answer": True
        }
    },
    {
        "inputs": {
            "messages": [
                {"role": "user", "content": "Can you help me with my order?"}
            ]
        },
        "outputs": {
            "next_agent": "",
            "coordinator_final_answer": True
        }
    },
    {
        "inputs": {
            "messages": [
                {"role": "user", "content": "Can you add two, ideally red tablets to my cart?"}
            ]
        },
        "outputs": {
            "next_agent": "product_qa_agent",
            "coordinator_final_answer": False
        }
    }
]


if __name__ == "__main__":

    client = Client(api_key=os.environ["LANGSMITH_API_KEY"])

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Dataset for evaluating routing of the coordinator agent"
    )

    for item in coordinator_eval_dataset:
        client.create_example(
            dataset_id=dataset.id,
            inputs={"messages": item["inputs"]["messages"]},
            outputs={
                "next_agent": item["outputs"]["next_agent"],
                "coordinator_final_answer": item["outputs"]["coordinator_final_answer"]
            }
        )