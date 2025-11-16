# 🧠 Simple Python Chatbot with Memory, Context & JSON Conversation Storage

This project is a lightweight chatbot built in Python using the **OpenAI API** (or **Ollama** locally).  
It supports:

✅ Conversation memory  
✅ Long-term JSON conversation saving  
✅ Automatic summarization to reduce token usage  
✅ Load & continue past chats  
✅ Switch between OpenAI and local LLaMA via Ollama  
✅ Command-based terminal interaction  

---

## ✨ Features

### **1. Memory + Context**
The chatbot stores each user and assistant message inside a `messages` list, allowing it to maintain conversation flow naturally.

### **2. JSON Conversation Saving**
Conversations can be saved and loaded anytime using simple commands:
```
save     → saves to conversation.json  
load     → loads previous conversation  
summary  → summarizes old messages  
quit     → exit program
```

### **3. Supports Both OpenAI & Ollama**
You choose the engine at runtime:

- **GPT-4o-mini** (OpenAI cloud)
- **LLaMA 3.2** (local via Ollama)

### **4. Automatic Token Optimization**
When messages become long, the system auto-summarizes the conversation to save tokens while preserving context.

---

## 🚀 How It Works

### **Initialization**
The chatbot creates initial system instructions:
```python
def create_initial_messages():
    return [{"role": "system", "content": "You are a helpful assistant."}]
```

### **Real-time Conversation**
Your input is appended and sent to the model:
```python
response = client.chat.completions.create(model=model_name, messages=messages)
```

### **Saving Conversations**
```python
with open("conversation.json", "w") as f:
    json.dump(messages, f)
```

### **Loading Conversations**
```python
return json.load(f)
```

### **Summarizing Conversation**
The last few messages are compressed into a shorter summary to reduce token usage.

---

## 🖥️ Running the Chatbot

1. Install dependencies:
```
pip install openai python-dotenv
```

2. Add your OpenAI key to `.env`:
```
OPENAI_API_KEY=your_key_here
```

3. Run the script:
```
python main.py
```

4. Choose model:
```
1 → OpenAI GPT-4o-mini  
2 → Local LLaMA via Ollama
```

5. Start chatting!

---

## 📁 File: `conversation.json`
This file stores your past chats so the assistant can remember them later.

Example:
```json
[
  {"role": "user", "content": "Hello!"},
  {"role": "assistant", "content": "Hi, how can I help you today?"}
]
```

---

## 📌 Commands You Can Use
Inside the chatbot:

| Command | Action |
|---------|---------|
| `save` | Save conversation to JSON |
| `load` | Load conversation from JSON |
| `summary` | Summarize long conversation |
| `quit` | Exit program |

---

## 🧩 Code Overview
The project includes:

- **initialize_client()** – chooses OpenAI or Ollama  
- **chat()** – main conversation logic  
- **summarize_messages()** – memory compression  
- **save_conversation()** – saves chat  
- **load_conversation()** – loads chat  
- **main()** – command-line interface  

---

## 📜 License
Open source — feel free to modify or extend.

---

## ⭐ Future Improvements
- Embedding-based long-term memory  
- Vector database (Chroma/FAISS)  
- UI version using Gradio or Streamlit  
- Voice input support  
- Agent-style tool usage

---


