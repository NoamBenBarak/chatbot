import uvicorn
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel
from qdrant_client.http.exceptions import ResponseHandlingException

from config import OPENAI_KEY, QDRANT_URL
from services.chatbot_service import Chatbot
from services.neural_search_service import NeuralSearcher

app = FastAPI()
neural_searcher = NeuralSearcher(collection_name="startups")
openai_client = OpenAI(api_key=OPENAI_KEY)


class Query(BaseModel):
    message: str


# Declaration of a message array to save the last messages in the conversation
messages = []

@app.post("/query")
def search_query(query: Query):
    """ 
    I thought maybe to send the combined_message instead of the query.message in the user_message - 
    and then to concatenating the msgs that have no chatbot response. 
    i know it not good enough, so it's in a comment 
    """
    #combined_message = combine_user_messages(query.message)

    """ 
    Another option that i thought about is this one.
    This means if the 'role' field for the last element in the array of messages is 'user' - 
    so the CHATBOT has not answered yet, so I am returning a generic message.
    """
    # if len(messages)>0:
    #     if messages[-1]["role"] == "user":
    #         return {"output": "another user msg"}
        

    # Check if there are already messages to summerize
    if len(messages) > 0:
        summerize_history = summerize()
    else:
        summerize_history = ""


    # Add the summerize (context) to the current query.
    user_message = {"role": "user", "content": summerize_history + " " + query.message}

    add_to_history(user_message)

    try:
        # We call the Neural Search service
        retrieved_data = neural_searcher.search(text=query.message)
    except ResponseHandlingException:
        # In case the Neural Search service is not running
        raise HTTPException(status_code=418, detail=f"Neural Search service is not running. Please run it in: {QDRANT_URL}")

    # We call the Chatbot service
    output = Chatbot.search(openai_client, retrieved_data, user_message)
    chatbot_res = {"role": "chatbot", "content": output}

    add_to_history(chatbot_res)

    return {"output": output}


@app.get("/summerize")
def summerize():
    last_msgs = get_last_msgs()
    last_msgs_text = [
        f"{msg['role']}: {msg['content']}" 
        for msg in reversed(list(last_msgs))
    ]
    # Messages array into a single paragraph (string)
    paragraph = ', '.join(last_msgs_text)

    try:
        response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
            "role": "system",
            "content": "Summarize content you are provided and asked with."
            },
            {
            "role": "user",
            "content": paragraph
            }
        ],
        temperature=0.4,
        top_p=1
        )
        content =response.choices[0].message.content
        return content
    except Exception as e:
        return {'error': str(e)}
    
def get_last_msgs():
    # Get the last msgs from the messages array (if exists)
    return messages[-min(10, len(messages)):]

def add_to_history(new_msg):
    if len(messages) == 10:
        messages.pop(0)    # According to assignment instructions, there is no need to save more than 10 recent msgs
    messages.append(new_msg)  # Append new element

def combine_user_messages(current_message):
    combined_message = current_message
    for msg in reversed(messages):
        print(msg)
        if msg["role"] == "user":
            combined_message = msg["content"] + " " + combined_message
        elif msg["role"] != "user":
            break
    return combined_message

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
