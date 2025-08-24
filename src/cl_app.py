import chainlit as cl
from rag_pipeline import RagPipeline
from config import API_KEY, SYSTEM_PROMPT_PATH, REWRITE_QUERY_PROMPT_PATH
import logging
from config import COLLECTION, GET_NEIGHBORS, MODEL, COLLECTION_RESOURCES
from docstore import SessionLocal


# # Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
    handlers=[logging.StreamHandler()]
)

# Create a module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Initialize your RAG pipeline once (singleton)
pipeline = RagPipeline(
    llm_api_key=API_KEY,
    llm_model=MODEL,
    collection_name=COLLECTION,
    rerank_top_k=5,
    system_prompt_path=SYSTEM_PROMPT_PATH,
    rewrite_query_prompt_path=REWRITE_QUERY_PROMPT_PATH,
    get_neighbors=GET_NEIGHBORS,
    collection_resources=COLLECTION_RESOURCES,
    docstore_session=SessionLocal
    
)

@cl.on_chat_start
async def start_chat():
    await cl.Message("Welcome! Ask me anything, and I'll answer using my knowledge base.").send()

@cl.on_message
async def main(message: cl.Message):
    # Stream the LLM's response
    stream, contexts = pipeline.chat(message.content)

    # show context
    if contexts:
        contexts = "\n".join(contexts)
        async with cl.Step(name="Context") as step:
            step.output = contexts

    # Create a Chainlit message for streaming
    msg = cl.Message(content="")

    # Stream chunks to UI (simulated - replace with actual streaming if available)
    for chunk in stream:  # If your LLM.stream() yields chunks
        print(chunk, end="#########", flush=True)
        await msg.stream_token(chunk)

    # Update final message
    await msg.send()

# Run with: chainlit run app.py -w