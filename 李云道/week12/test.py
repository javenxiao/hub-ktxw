import asyncio
from fastmcp import Client


async def call_tool():
    for i in range(5):
        try:
            client = Client(f"http://localhost:{8900 + i}/sse")
            async with client:
                result = await client.list_tools()
                print(result[0])
                print([t.name for t in result])
        except:
            print(f"Failed to connect to service on port {8900 + i}")


asyncio.run(call_tool())


async def call_sentiment_classification(text: str):
    client = Client("http://localhost:8903/sse")
    async with client:
        result = await client.call_tool("sentiment_classification", arguments={"text": text})
        print(result)

text="你他妈的有病吧"
asyncio.run(call_sentiment_classification(text))