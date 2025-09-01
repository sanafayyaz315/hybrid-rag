import chainlit as cl
from sqlalchemy import select
import base64, json
from io import BytesIO
from fastapi import UploadFile
from src.builder import pipeline
from src.docstore.session import AsyncSessionLocal, SessionLocal
from src.api.routes import upload_files
from src.docstore.files_crud import (async_list_all_files, 
                                     async_list_files, 
                                     async_ensure_unique_filenames, 
                                     async_stage_file_rows, 
                                     async_delete_file_row
                                    )
from src.rag_utils import rewrite_query, check_context_relevance
from src.builder import pipeline, rewrite_query_prompt
import logging

history_window = 5
# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@cl.set_starters
async def set_starters():
    return []

@cl.on_stop
def on_stop():
    cl.user_session.set("stop", True)

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if (username, password) == ("admin", "admin"):
        user = cl.User(identifier="admin", display_name="Admin User", metadata={})
    else:
        user = None
    return user

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("history", [])
    cl.user_session.set("pipeline", pipeline)

    actions = [
        cl.Action(
            name="get_files_action",
            payload={},
            label="Get Files",
            icon="file-text"
        ),
        cl.Action(
            name="upload_file_action",
            payload={},
            label="Upload File",
            icon="upload"
        ),
        cl.Action(
            name="delete_file_action",
            payload={},
            label="Delete File",
            icon="trash-2"
        )
    ]
    await cl.Message(
        content=f"## File Management Options",
        actions=actions
    ).send()

@cl.action_callback("get_files_action")
async def get_files_action(action: cl.Action):  
    async with AsyncSessionLocal() as session:
        file_names = await async_list_files(session) 
    
    if not file_names:
        file_names = ["No files found."]
    
    logger.debug(f"List of files: {file_names}")
    props = {"files": file_names}
    print(props)
    file_element = cl.CustomElement(name="GetFiles", props=props)
    print(file_element)
    await cl.Message(
        content="Here is the files information!",
        elements=[file_element]
    ).send()

@cl.action_callback("upload_file_action")
async def upload_file_action(action: cl.Action):
    element = cl.CustomElement(name="UploadFileModal", props={})
    await cl.Message(
        content="Select a file to upload:",
        elements=[element]
    ).send()


@cl.action_callback("confirm_upload_file")
async def confirm_upload_file(action: cl.Action):
    filename = action.payload.get("filename")
    file_data = action.payload.get("file")

    if not filename or not file_data:
        await cl.Message(content="No file selected.").send()
        return

    # Decode base64 to bytes
    content = base64.b64decode(file_data.split(",")[1])
    size_kb = len(content) / 1024

    # Create UploadFile-like object
    temp_file = BytesIO(content)
    upload_file = UploadFile(filename=filename, file=temp_file)

    # Use async DB session
    async with AsyncSessionLocal() as session:
        try:
            # Call your async upload_files function
            response = await upload_files(file=upload_file, session=session)
            await cl.Message(
                content=f"File '{filename}' uploaded successfully ({size_kb:.2f} KB)\nResponse: {response}"
            ).send()
        except Exception as e:
            await cl.Message(content=f"Error uploading file: {e}").send()

@cl.action_callback("delete_file_action")
async def delete_file_action(action: cl.Action):
    element = cl.CustomElement(name="DeleteFileModal", props={"filename": ""})
    await cl.Message(
        content="Enter the filename to delete:",
        elements=[element]
    ).send()


@cl.action_callback("confirm_delete_file")
async def confirm_delete_file(action: cl.Action):
    filename = action.payload.get("filename")   
    if not filename:
        await cl.Message(content="No filename provided.").send()
        return
    async with AsyncSessionLocal() as session:
        try:
            # Call async delete_file endpoint 
            res = await async_delete_file_row(filename, session=session)
            if res:
                await cl.Message(content=f"File '{filename}' and related docstore entries deleted.").send()
            else:
                await cl.Message(content=f"{filename} not available").send()
        
        except Exception as e:
            await cl.Message(content=f"Error deleting file: {e}").send()

@cl.on_message
async def on_message(message: cl.Message):
    history = cl.user_session.get("history")
    pipeline = cl.user_session.get("pipeline")
    chat_history = "\n".join(f"{item['role'].capitalize()}: {item['content']}" for item in history[history_window:])
    user_message = message.content
    history.append(
        {
            "role": "user",
            "content": user_message
        }
    )

    async with cl.Step("Query Rewrite") as step:
        query, sources = await rewrite_query(user_query=user_message, rewrite_query_prompt=rewrite_query_prompt, llm=pipeline.llm)       
        step.output = query

    logger.debug(f"Actual query: {user_message}\nRewritten query: {query}\n Sources: {sources}")

    retrieved_docs = await pipeline.retrieve(query=query,
                                             sources=sources                                      
                                      )
    contexts = retrieved_docs
    formatted_context = pipeline.build_context(retrieved_docs)
    print("retrieved_docs:", retrieved_docs)
    logger.debug(f"Retrieved {len(retrieved_docs)} docs")
    stream = await pipeline.generate_response(query, formatted_context)

    # stream response
    response = cl.Message(content="")
    res = ""
    for chunk in stream:
        res += chunk
        await response.stream_token(chunk)    

    # output sources
    elements = []
    names = []
   
    for n, context in enumerate(contexts):
        name = f"Source_{n}: {context.get('metadata', {}).get('source', 'Unknown Source')}"
        content = context.get("text", "")
        names.append(name)
        element = cl.Text(
            content=content, name=name, display="side"
        )
        elements.append(element)
    
    names = "\n".join(names)
    result = f"{res}\n\nSources:\n{names}"
    response.content = result
    response.elements = elements

    history.append(
        {
            "role": "assistant",
            "content": res
        }
    )
    await response.update()
    await response.send()
