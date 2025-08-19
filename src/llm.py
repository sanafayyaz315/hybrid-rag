from openai import OpenAI, AsyncOpenAI

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
