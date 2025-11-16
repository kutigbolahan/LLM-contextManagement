import sys
from typing import Dict, List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def initialize_client(use_ollama: bool= False)->OpenAI:
    """Initialize the OpenAI client for either OpenAi or OLLAMA"""
    if use_ollama:
        return OpenAI(base_url="http://localhost:11434/v1",api_key="ollama")
    return OpenAI()

def create_initial_messages() -> List[Dict[str,str]]:
     """Create the initial messages for the context memory."""
     return [
         {"role":"system", "content": "You are a helpful assistant."}
     ]
     
def chat(user_input: str, messages:List[Dict[str,str]], client:OpenAI, model_name:str)-> str:
    """Handles user input and generate responses"""
    messages.append({"role":"user", "content":user_input})
    
    try:
        response = client.chat.completions.create(model=model_name, messages=messages)
        assistant_response = response.choices[0].message.content
        messages.append({"role":"assistant", "content":assistant_response})
        return assistant_response
        
    except Exception as e:   
        return f"Error with API:{str(e)}"