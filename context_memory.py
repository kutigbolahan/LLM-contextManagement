import json
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

def summarize_messages(messages: List[Dict[str,str]])->List[Dict[str,str]]:
    """Summarize older messages to save tokens"""
    summary = "Previous conversation summarized:" + "   ".join([
        m["content"][:50]+ "..." for m in messages[-5:] 
    ])   
    return [{"role":"system", "content": summary}] + messages[-5:]

def save_conversation(messages:List[Dict[str,str]], filename: str = "conversation.json"):
    """Save conversation to a file"""
    with open(filename,"w") as f:
        json.dump(messages,f)
        
def load_conversation(filname:str="conversation.json")-> List[Dict[str,str]]:
    """Loads previous conversations from file"""
    try:
        with open(filname, "r") as f:
            return   json.load(f)
    except FileNotFoundError:
        print(f"No Conversation file found at{filname}")
        return create_initial_messages()
    

def main():
    print("Select model type:")
    print("1. OpenAI GPT 4")
    print("2. Llama (local)")
    
    choice = input("Enter choice(1 or 2): ")
    use_ollama = choice == '2'
    
    client = initialize_client(use_ollama)
    model_name = "llama3.2" if use_ollama else "gpt-4o-mini"
    messages= create_initial_messages()
    
    print(f"\nUsing {'Ollama' if use_ollama else 'OpenAI'} model.")
    print("Available commands:")
    print("-'save': Save conversation")
    print("-'load': Load conversation")
    print("-'summary': Summarize conversation")
    print("-'quit': Quit conversation")
    
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() =="quit":
            break
        elif user_input.lower() =="save":
            save_conversation(messages)
            print("Conversation Saved!")
            continue
        elif user_input.lower() =="load":
            messages = load_conversation()
            print("Conversation loaded")     
        elif user_input.lower() == "summary":
            messages = summarize_messages()
            print("Summarized messages")  
            continue  
        
        response = chat(user_input, messages,client, model_name)
        print(f"\nAssitant: {response}")
        
        if len(messages) > 10:
            messages = summarize_messages(messages)
            print("\n(Conversation automatically summarized)")
            
if __name__== "__main__": 
    main()           