from fastmcp import FastMCP
from src.reviews_mcp_server.utils import retrieve_review_context, process_review_context

mcp = FastMCP("reviews")

@mcp.tool()
def get_formatted_review_context(query: str, item_list: list[str], top_k: int = 20) -> str:

    """Get the top k reviews matching a query for a list of prefiltered items.
    
    Args:
        query: The query to get the top k reviews for
        item_list: The list of item IDs to prefilter for before running the query
        top_k: The number of reviews to retrieve, this should be at least 20 if multipple items are prefiltered
    
    Returns:
        A string of the top k context chunks with IDs prepending each chunk, each representing an inventory item for a given query.
    """

    context = retrieve_review_context(query, item_list, top_k)
    formatted_context = process_review_context(context)

    return formatted_context


if __name__ == "__main__":
    mcp.run(transport='http', host="0.0.0.0", port=8000)