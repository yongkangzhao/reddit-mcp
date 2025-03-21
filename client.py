import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv
from openai import AsyncOpenAI
import json

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
        self.system_prompt = """
            You are a helpful assistant, and you will try your best to help the user.
            You won't give up on the first try, and you will be creative on solving the user's problem.
            When the first thing you tried fails, you should come up with alternatives.
            Before and after you do anything, make sure to thing about it first.

            <think>think about the situation</think>
            before doing anything think the situation.

            <action>what you should do</action>
            once fully reasoned about the situation, execute accordingly.

            <reflect>reflect on what you did and the result of your action</reflect>
            everytime you do something, always reflect on the result of your action, is it what you expected? If not, always think about what else you could have done.

            continue the cycle until completion.
            """
        self.messages = []

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
            
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query_claude(self, query: str) -> str:
        print("process query using Claude")
        """Process a query using Claude and available tools"""
        self.messages.append(
            {
                "role": "user",
                "content": query
            }
        )

        response = await self.session.list_tools()
        available_tools = [{ 
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        # Initial Claude API call
        response = self.anthropic.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1000,
            system=self.system_prompt,
            messages=self.messages,
            tools=available_tools
        )

        # Process response and handle tool calls
        final_text = []

        for content in response.content:
            if content.type == 'text':
                final_text.append(content.text)
            elif content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input
                
                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                # Continue conversation with tool results
                if hasattr(content, 'text') and content.text:
                    self.messages.append({
                      "role": "assistant",
                      "content": content.text
                    })
                self.messages.append({
                    "role": "user", 
                    "content": result.content
                })

                # Get next response from Claude
                response = self.anthropic.messages.create(
                    model="claude-3-7-sonnet-20250219",
                    max_tokens=10000,
                    messages=self.messages,
                )

                final_text.append(response.content[0].text)

        return "\n".join(final_text)



    async def process_query_gpt(self, query: str) -> str:
        print("Process query using GPT")
        """Process a query using OpenAI and available tools"""
        if not self.messages:
            self.messages.append({
                "role": "system",
                "content": self.system_prompt
            })
        self.messages.append(
            {
                "role": "user", 
                "content": query
            }
        )
        response = await self.session.list_tools()
        available_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in response.tools
        ]

        client = AsyncOpenAI()

        # Initial OpenAI API call
        response = await client.chat.completions.create(
            model="o3-mini",
            messages=self.messages,
            tools=available_tools,
            tool_choice="auto",
        )

        final_text = []

        message = response.choices[0].message
        
        while True:
            if message.content:
                final_text.append(message.content)
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = tool_call.function.arguments

                    # Ensure tool_args is a dictionary
                    if isinstance(tool_args, str):
                        tool_args = json.loads(tool_args)

                    # Execute tool call
                    result = await self.session.call_tool(tool_name, tool_args)
                    final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                    # Continue conversation with tool results
                    self.messages.append(message)
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result.content,
                        }
                    )

                response = await client.chat.completions.create(
                    model="o3-mini",
                    messages=self.messages,
                    tools=available_tools,
                    tool_choice="auto",
                )
                message = response.choices[0].message
            else:
                break

        return "\n".join(final_text)


    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                    
                response = await self.process_query_claude(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def query_loop(self, query):
        """Run an non-interactive chat loop"""
        while True:
            try:                    
                response = await self.process_query_claude(query)
                print("\n" + response)
                return response
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)
        
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())