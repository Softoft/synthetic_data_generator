import asyncio
from typing import Optional

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

client = AsyncOpenAI()


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


if __name__ == '__main__':
    response_format: Optional[type(BaseModel)] = CalendarEvent
    completion = asyncio.run(client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            { "role": "system", "content": "Extract the event information." },
            { "role": "user", "content": "Alice and Bob are going to a science fair on Friday." },
        ],
    ))

    print(f"{completion.choices[0]}")
    print(f"{completion}")
