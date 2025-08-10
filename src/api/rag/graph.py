from pydantic import BaseModel, Field
from typing import List, Dict, Any, Annotated, Optional
from operator import add

from api.rag.agents import ToolCall, RAGUsedContext, coordinator_agent_node, product_qa_agent_node, MCPToolCall, shopping_cart_agent_node, Delegation
from api.rag.utils.utils import mcp_tool_node, get_tool_descriptions_from_mcp_servers, get_tool_descriptions_from_node
from api.rag.tools import add_to_shopping_cart, remove_from_cart, get_shopping_cart
from api.core.config import config

import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.prebuilt import ToolNode


class State(BaseModel):
    messages: Annotated[List[Any], add] = []
    answer: str = ""
    product_qa_iteration: int = Field(default=0)
    shopping_cart_iteration: int = Field(default=0)
    coordinator_iteration: int = Field(default=0)
    product_qa_final_answer: bool = Field(default=False)
    shopping_cart_final_answer: bool = Field(default=False)
    coordinator_final_answer: bool = Field(default=False)
    product_qa_available_tools: List[Dict[str, Any]] = []
    shopping_cart_available_tools: List[Dict[str, Any]] = []
    mcp_tool_calls: Optional[List[MCPToolCall]] = Field(default_factory=list)
    tool_calls: Optional[List[ToolCall]] = Field(default_factory=list)
    retrieved_context_ids: List[RAGUsedContext] = []
    trace_id: str = ""
    user_id: str = ""
    cart_id: str = ""
    plan: list[Delegation] = Field(default_factory=list)
    next_agent: str = ""


#### ROUTERS ####

def product_qa_tool_router(state) -> str:
    """Decide whether to continue or end"""
    
    if state.product_qa_final_answer:
        return "end"
    elif state.product_qa_iteration > 3:
        return "end"
    elif len(state.mcp_tool_calls) > 0:
        return "tools"
    else:
        return "end"


def shopping_cart_tool_router(state) -> str:
    """Decide whether to continue or end"""
    
    if state.shopping_cart_final_answer:
        return "end"
    elif state.shopping_cart_iteration > 3:
        return "end"
    elif len(state.tool_calls) > 0:
        return "tools"
    else:
        return "end"


def coordinator_router(state) -> str:
    """Decide whether to continue or end"""
    
    if state.coordinator_final_answer:
        return "end"
    elif state.coordinator_iteration > 4:
        return "end"
    elif state.next_agent == "product_qa_agent":
        return "product_qa_agent"
    elif state.next_agent == "shopping_cart_agent":
        return "shopping_cart_agent"
    else:
        return "end"


#### WORKFLOW ####

shopping_cart_agent_tools = [add_to_shopping_cart, remove_from_cart, get_shopping_cart]
shopping_cart_tool_node = ToolNode(shopping_cart_agent_tools)
shopping_cart_tool_descriptions = get_tool_descriptions_from_node(shopping_cart_tool_node)

workflow = StateGraph(State)

workflow.add_edge(START, "coordinator_agent_node")

workflow.add_node("shopping_cart_agent_node", shopping_cart_agent_node)
workflow.add_node("coordinator_agent_node", coordinator_agent_node)
workflow.add_node("product_qa_agent_node", product_qa_agent_node)
workflow.add_node("product_qa_tool_node", mcp_tool_node)
workflow.add_node("shopping_cart_tool_node", shopping_cart_tool_node)

workflow.add_conditional_edges(
    "coordinator_agent_node",
    coordinator_router,
    {
        "product_qa_agent": "product_qa_agent_node",
        "shopping_cart_agent": "shopping_cart_agent_node",
        "end": END
    }
)

workflow.add_conditional_edges(
    "product_qa_agent_node",
    product_qa_tool_router,
    {
        "tools": "product_qa_tool_node",
        "end": "coordinator_agent_node"
    }
)

workflow.add_conditional_edges(
    "shopping_cart_agent_node",
    shopping_cart_tool_router,
    {
        "tools": "shopping_cart_tool_node",
        "end": "coordinator_agent_node"
    }
)

workflow.add_edge("product_qa_tool_node", "product_qa_agent_node")
workflow.add_edge("shopping_cart_tool_node", "shopping_cart_agent_node")


async def run_agent(question: str, thread_id: str):

    mcp_servers = ["http://items_mcp_server:8000/mcp", "http://reviews_mcp_server:8000/mcp"]

    product_qa_tool_descriptions = await get_tool_descriptions_from_mcp_servers(mcp_servers)

    initial_state = {
        "messages": [{"role": "user", "content": question}],
        "user_id": thread_id,
        "cart_id": thread_id,
        "product_qa_iteration": 0,
        "shopping_cart_iteration": 0,
        "coordinator_iteration": 0,
        "product_qa_available_tools": product_qa_tool_descriptions,
        "shopping_cart_available_tools": shopping_cart_tool_descriptions,
        "product_qa_final_answer": False,
        "shopping_cart_final_answer": False,
        "coordinator_final_answer": False
    }

    configuration = {"configurable": {"thread_id": thread_id}}

    async with AsyncPostgresSaver.from_conn_string(config.POSTGRES_CONN_STRING) as checkpointer:

        graph = workflow.compile(checkpointer=checkpointer)

        result = await graph.ainvoke(initial_state, config=configuration)

    return result


async def run_agent_wrapper(question: str, thread_id: str):

    qdrant_client = QdrantClient(url=config.QDRANT_URL)

    result = await run_agent(question, thread_id)

    image_url_list = []
    dummy_vector = np.zeros(1536).tolist()
    for id in result.get("retrieved_context_ids", []):
        payload = qdrant_client.query_points(
            collection_name=config.QDRANT_COLLECTION_NAME_ITEMS,
            query=dummy_vector,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="parent_asin",
                        match=MatchValue(value=id.id)
                    )
                ]
            ),
            with_payload=True,
            limit=1
        ).points[0].payload
        image_url = payload.get("first_large_image")
        price = payload.get("price")
        if image_url:
            image_url_list.append({"image_url": image_url, "price": price, "description": id.description})

    shopping_cart = get_shopping_cart(thread_id, thread_id)
    shopping_cart_items = [
            {
                "price": item.get("price"),
                "quantity": item.get("quantity"),
                "currency": item.get("currency"),
                "product_image_url": item.get("product_image_url"),
                "total_price": item.get("total_price")
            } 
        for item in shopping_cart
    ]

    return {
        "answer": result.get("answer"),
        "retrieved_images": image_url_list,
        "trace_id": result.get("trace_id"),
        "shopping_cart": shopping_cart_items
    }
