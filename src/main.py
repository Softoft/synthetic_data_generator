import asyncio

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

client = AsyncOpenAI()


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


if __name__ == '__main__':
    completion = asyncio.run(client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            { "role": "system", "content": "Extract the event information." },
            { "role": "user", "content": "Alice and Bob are going to a science fair on Friday." },
        ],
        response_format=CalendarEvent,
    ))

    event = completion.choices[0].message.parsed

    print(f"Event: {event}")
    print(f"{completion.choices[0]}")
    print(f"{completion}")
