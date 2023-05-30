import openai
import asyncio
from config import (
    dev_name,
    relevant_messages_length,
    recent_messages_length,
    system_message,
)
from retrieval import (
    alt_retrieve_relevant_messages,
    retrieve_relevant_messages,
    retrieve_relevant_messages,
    retrieve_recent_messages,
)


# defines a function for asynchronously handling chat completion
async def async_chat_completion_create(
    model_for_completion, messages_for_completion, response_tokens
):
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: openai.ChatCompletion.create(
            model=model_for_completion,
            messages=messages_for_completion,
            max_tokens=response_tokens,
        ),
    )
    return response


# defines a function for handling chat completion that isn't asynchronous. this is useful for situations where we NEED the response before continuing. database handling etc.
def chat_completion_create(
    model_for_completion, messages_for_completion, response_tokens
):
    response = openai.ChatCompletion.create(
        model=model_for_completion,
        messages=messages_for_completion,
        max_tokens=response_tokens,
    )
    return response


# defines a helper function that handles creation of the messages block of the chat completion.
async def generate_completion_messages(message):
    if message.author.name == dev_name:
        print("dev name detected, generating alt relevant messages")
        relevant_messages = await alt_retrieve_relevant_messages(
            message, relevant_messages_length
        )
        print(relevant_messages)
    else:
        relevant_messages = retrieve_relevant_messages(
            message, relevant_messages_length
        )

    # previous_relevant_messages = retrieve_previous_relevant_messages(message)
    recent_messages = await retrieve_recent_messages(message, recent_messages_length)
    timestamp = str(message.created_at)[:-16]
    assistant_message = f"I am planning a response to {message.author.name}'s upcoming message. I will first take in contextual information so that I can respond appropriately. These are the recent messages in this discord channel: {recent_messages}. These are retrieved messages from the message database that may be relevant to the conversation: {relevant_messages}. The current time is {timestamp}. With this information taken into account, I will respond to {message.author.name} appropriately."
    # print(assistant_message)
    messages = [
        {"role": "system", "content": system_message},
        {"role": "assistant", "content": assistant_message},
        {"role": "user", "content": f"{message.clean_content}"},
    ]

    return messages
