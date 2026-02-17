import requests
import json
import gradio as gr

url = "http://localhost:11434/api/generate"

headers = {
    'Content-Type': 'application/json',
}

def generate_response(message, history):
    """Generate response using Ollama API with conversation history."""
    # Convert Gradio chat history format to prompt format
    history_prompt = []
    
    # Handle history - Gradio ChatInterface passes history as list of tuples (user_msg, assistant_msg)
    # Use indexing instead of unpacking to avoid ValueError
    if history:
        for entry in history:
            if isinstance(entry, (tuple, list)) and len(entry) >= 1:
                user_msg = entry[0] if len(entry) > 0 else ""
                assistant_msg = entry[1] if len(entry) > 1 else ""
                
                if user_msg:
                    history_prompt.append(f"User: {user_msg}")
                if assistant_msg:
                    history_prompt.append(f"Assistant: {assistant_msg}")
    
    # Add current user message
    history_prompt.append(f"User: {message}")
    
    # Create the full context string
    final_prompt = "\n".join(history_prompt)
    
    data = {
        "model": "codeguru",
        "prompt": final_prompt,
        "stream": False
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()  # Check for HTTP errors
        
        # Parse the JSON response
        response_data = response.json()
        actual_response = response_data.get("response", "Error: No response field found.")
        
        return actual_response
    
    except requests.exceptions.RequestException as e:
        return f"Request failed: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"

# Setting up the Gradio Chat Interface
demo = gr.ChatInterface(
    fn=generate_response,
    title="CodeGuru Chat",
    description="Chat with CodeGuru AI assistant",
    examples=["Hello!", "Write a Python function to calculate factorial", "Explain recursion"],
)

if __name__ == "__main__":
    demo.launch()