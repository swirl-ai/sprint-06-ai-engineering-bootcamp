from fastapi import APIRouter, Request
import logging

from api.rag.graph import run_agent_wrapper
from api.processors.submit_feedback import submit_feedback

from api.api.models import RAGRequest, RAGResponse, RAGUsedImage, FeedbackRequest, FeedbackResponse, ShoppingCartItem


logger = logging.getLogger(__name__)

rag_router = APIRouter()
feedback_router = APIRouter()


@rag_router.post("/rag")
async def rag(
    request: Request,
    payload: RAGRequest
) -> RAGResponse:

    result = await run_agent_wrapper(payload.query, payload.thread_id)

    used_image_urls = [
        RAGUsedImage(
            image_url=image["image_url"],
            price=image["price"],
            description=image["description"]
        ) 
        for image in result["retrieved_images"]
    ]

    shopping_cart = [
        ShoppingCartItem(
            price=item["price"],
            quantity=item["quantity"],
            currency=item["currency"],
            product_image_url=item["product_image_url"],
            total_price=item["total_price"]
        )
        for item in result["shopping_cart"]
    ]

    return RAGResponse(
        request_id=request.state.request_id,
        answer=result["answer"],
        used_image_urls=used_image_urls,
        trace_id=result["trace_id"],
        shopping_cart=shopping_cart
    )


@feedback_router.post("/submit_feedback")
async def send_feedback(
    request: Request,
    payload: FeedbackRequest
) -> FeedbackResponse:

    submit_feedback(payload.trace_id, payload.feedback_score, payload.feedback_text, payload.feedback_source_type)

    return FeedbackResponse(
        request_id=request.state.request_id,
        status="success"
    )


api_router = APIRouter()
api_router.include_router(rag_router, tags=["rag"])
api_router.include_router(feedback_router, tags=["feedback"])