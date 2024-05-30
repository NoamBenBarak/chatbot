# Start-Ups Q&A AI Chatbot

This Chatbot provides information on Startups all over the world.

## How to use it?

1. Create a virtual environment and activate into it
2. run `pip install -r requirements.txt`
3. run `python main.py`

Example query:

```
POST request to http://localhost:8000/query
Payload:
{
    "message": "Are there startups about wine?"
}
```

You will have a Docker container with the Neural Search running on port 6333:6333.

## 1. Neural Search Service

The first step was to build the Neural Search service. In the encoding_model folder you will find:

- `startups_demo.json` (raw data of companies from https://storage.googleapis.com/generall-shared-data/startups_demo.json)
- `vector_encoding.py` (script for encoding the raw startups' data)
- `startup_vectors.npy` (encoded vectors)
- `qdrant.py` (script for loading the vectors to Qdrant)

Steps to set run Qdrant service locally:

1. Make sure Docker is installed and running on your system
2. In the terminal, go to `chatbot-main` directory
3. Pull Qdrant imgae: `docker pull qdrant/qdrant`
4. Paste the following command in the terminal:

```
docker run -p 6333:6333 -p 6334:6334 \
-v "$(pwd)/encoding_model":/qdrant/storage:z \
qdrant/qdrant
```

5. In the terminal Go to `chatbot-main` directory
6. Run `python -m encoding_model.qdrant`

## 2. Chatbot Service

- A basic implementation that receives a data from the Neural Search Service, and a user message
- The service returns an answer generated by OpenAI model based on the data and the user message
