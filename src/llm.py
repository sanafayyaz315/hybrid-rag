from openai import OpenAI, AsyncOpenAI
import asyncio
from typing import List, AsyncGenerator

class LLM:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """Initialize with required API key and model name. 
        Args:
            api_key (str): OpenAI API key
            model (str): Default model to use (e.g., "gpt-4", "gpt-3.5-turbo"). Default set to gpt-3.5-turbo
        """  
        self.llm = OpenAI(api_key=api_key)
        self.model = model

    def invoke(self, messages: list) -> str:
        """Get complete response."""
        response = self.llm.chat.completions.create( 
        model=self.model,
        messages=messages,
        seed=42
    )
        return response.choices[0].message.content

    def stream(self, messages: list):
        """Stream response token by token."""
        stream = self.llm.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            seed=42
        )
        for chunk in stream:
            if content := chunk.choices[0].delta.content:
                yield content

    async def async_invoke(self, messages: List[dict]) -> str:
        loop = asyncio.get_event_loop()
        # Run blocking invoke in threadpool
        return await loop.run_in_executor(None, self.invoke, messages)

    async def async_stream(self, messages: List[dict]) -> AsyncGenerator[str, None]:
        loop = asyncio.get_event_loop()
        # Create a queue to buffer the streaming results
        queue = asyncio.Queue()

        def _stream():
            for chunk in self.stream(messages):
                asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)  # signal completion

        # Run blocking stream in executor
        asyncio.get_running_loop().run_in_executor(None, _stream)

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk

if __name__ == "__main__":
    from config import API_KEY
    llm = LLM(api_key=API_KEY)
    # Regular invocation
    print("\nRegular response:")
    message = [{"role": "user", "content": "Explain quantum computing simply"}]
    response = llm.invoke(message)
    print(response)

    # Streaming
    print("\nStreaming response:")
    message = [{"role": "user", "content": "Explain photosynthesis in detail"}]
    for chunk in llm.stream(message):
        print(chunk, end="", flush=True)
