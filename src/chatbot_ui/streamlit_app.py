import streamlit as st
import requests
import uuid
import logging
from src.chatbot_ui.core.config import settings


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


st.set_page_config(
    page_title="Ecommerce Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

def get_session_id():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

session_id = get_session_id()

def api_call(method, url, **kwargs):
    def _show_error_popup(message):
        """Show error message as a popup in the top-right corner."""
        st.session_state["error_popup"] = {
            "visible": True,
            "message": message,
        }

    try:
        response = getattr(requests, method)(url, **kwargs)

        try:
            response_data = response.json()
        except requests.exceptions.JSONDecodeError:
            response_data = {"message": "Invalid response format from server"}

        if response.ok:
            return True, response_data

        return False, response_data

    except requests.exceptions.ConnectionError:
        _show_error_popup("Connection error. Please check your network connection.")
        return False, {"message": "Connection error"}
    except requests.exceptions.Timeout:
        _show_error_popup("The request timed out. Please try again later.")
        return False, {"message": "Request timeout"}
    except Exception as e:
        _show_error_popup(f"An unexpected error occurred: {str(e)}")
        return False, {"message": str(e)}

def submit_feedback(feedback_type=None, feedback_text=""):
    """Submit feedback to the API endpoint"""

    def _feedback_score(feedback_type):
        if feedback_type == "positive":
            return 1
        elif feedback_type == "negative":
            return 0
        else:
            return None 
    
    feedback_data = {
        "feedback_score": _feedback_score(feedback_type),
        "feedback_text": feedback_text,
        "trace_id": st.session_state.trace_id,
        "thread_id": session_id,
        "feedback_source_type": "api"
    }

    logger.info(f"Feedback data: {feedback_data}")
    
    status, response = api_call("post", f"{settings.API_URL}/submit_feedback", json=feedback_data)
    return status, response



# Initialize session state variables
if "retrieved_items" not in st.session_state:
    st.session_state.retrieved_items = []

if "shopping_cart" not in st.session_state:
    st.session_state.shopping_cart = []

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]

if "query_counter" not in st.session_state:
    st.session_state.query_counter = 0

if "sidebar_key" not in st.session_state:
    st.session_state.sidebar_key = 0

if "sidebar_placeholder" not in st.session_state:
    st.session_state.sidebar_placeholder = None

# Initialize feedback states (simplified)
if "latest_feedback" not in st.session_state:
    st.session_state.latest_feedback = None

if "show_feedback_box" not in st.session_state:
    st.session_state.show_feedback_box = False

if "feedback_submission_status" not in st.session_state:
    st.session_state.feedback_submission_status = None

if "trace_id" not in st.session_state:
    st.session_state.trace_id = None

# Sidebar with Tabs
with st.sidebar:
    # Create tabs in the sidebar
    suggestions_tab, cart_tab = st.tabs(["🔍 Suggestions", "🛒 Shopping Cart"])
    
    # Suggestions Tab
    with suggestions_tab:
        if st.session_state.retrieved_items:
            for idx, item in enumerate(st.session_state.retrieved_items):
                st.caption(item.get('description', 'No description'))
                if 'image_url' in item:
                    st.image(item["image_url"], width=250)
                st.caption(f"Price: {item['price']} USD")
                st.divider()
        else:
            st.info("No suggestions yet")
    
    # Shopping Cart Tab
    with cart_tab:
        if st.session_state.shopping_cart:
            
            for idx, item in enumerate(st.session_state.shopping_cart):
                st.caption(item.get('description', 'No description'))
                if 'product_image_url' in item:
                    st.image(item["product_image_url"], width=250)
                st.caption(f"Price: {item['price']} {item['currency']}")
                st.caption(f"Quantity: {item['quantity']}")
                st.caption(f"Total price: {item['total_price']} {item['currency']}")
                st.divider()
        else:
            st.info("Your cart is empty")

# Main content - Chat interface

# Display all messages
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Add feedback buttons only for the latest assistant message (excluding the initial greeting)
        is_latest_assistant = (
            message["role"] == "assistant" and 
            idx == len(st.session_state.messages) - 1 and 
            idx > 0
        )
        
        if is_latest_assistant:
            # Use Streamlit's built-in feedback component
            feedback_result = st.feedback("thumbs", key="feedback_latest")
            
            # Handle feedback selection
            if feedback_result is not None:
                feedback_type = "positive" if feedback_result == 1 else "negative"
                
                # Only submit if this is a new/different feedback
                if st.session_state.latest_feedback != feedback_type:
                    with st.spinner("Submitting feedback..."):
                        status, response = submit_feedback(feedback_type=feedback_type)
                        if status:
                            st.session_state.latest_feedback = feedback_type
                            st.session_state.feedback_submission_status = "success"
                            st.session_state.show_feedback_box = (feedback_type == "negative")
                        else:
                            st.session_state.feedback_submission_status = "error"
                            st.error("Failed to submit feedback. Please try again.")
                    st.rerun()
            
            # Show feedback status message
            if st.session_state.latest_feedback and st.session_state.feedback_submission_status == "success":
                if st.session_state.latest_feedback == "positive":
                    st.success("✅ Thank you for your positive feedback!")
                elif st.session_state.latest_feedback == "negative" and not st.session_state.show_feedback_box:
                    st.success("✅ Thank you for your feedback!")
            elif st.session_state.feedback_submission_status == "error":
                st.error("❌ Failed to submit feedback. Please try again.")
            
            # Show feedback text box if thumbs down was pressed
            if st.session_state.show_feedback_box:
                st.markdown("**Want to tell us more? (Optional)**")
                st.caption("Your negative feedback has already been recorded. You can optionally provide additional details below.")
                
                # Text area for detailed feedback
                feedback_text = st.text_area(
                    "Additional feedback (optional)",
                    key="feedback_text_latest",
                    placeholder="Please describe what was wrong with this response...",
                    height=100
                )
                
                # Send additional feedback button
                col_send, col_spacer, col_close = st.columns([3, 5, 2])
                with col_send:
                    if st.button("Send Additional Details", key="send_additional_feedback"):
                        if feedback_text.strip():  # Only send if there's actual text
                            with st.spinner("Submitting additional feedback..."):
                                status, response = submit_feedback(feedback_text=feedback_text)
                                if status:
                                    st.success("✅ Thank you! Your additional feedback has been recorded.")
                                    st.session_state.show_feedback_box = False
                                else:
                                    st.error("❌ Failed to submit additional feedback. Please try again.")
                        else:
                            st.warning("Please enter some feedback text before submitting.")
                        st.rerun()
                
                with col_close:
                    if st.button("Close", key="close_feedback_latest"):
                        st.session_state.show_feedback_box = False
                        st.rerun()

# Chat input
if prompt := st.chat_input("Hello! How can I assist you today?"):
    # Reset feedback state when new message is sent
    st.session_state.latest_feedback = None
    st.session_state.show_feedback_box = False
    st.session_state.feedback_submission_status = None
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.spinner("Thinking..."):
        status, output = api_call("post", f"{settings.API_URL}/rag", json={"query": prompt, "thread_id": session_id})
        # Update retrieved items and shopping cart
        st.session_state.retrieved_items = output.get("used_image_urls", [])
        st.session_state.shopping_cart = output.get("shopping_cart", [])
        st.session_state.trace_id = output.get("trace_id", None)
        
        response_content = output.get("answer", str(output))
    
    st.session_state.messages.append({"role": "assistant", "content": response_content})
    st.rerun()