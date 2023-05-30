from database_storage import store_document, is_dm
from discord_commands import handle_command
from chat_completion import generate_completion_messages, async_chat_completion_create
from config import model, max_discord_message_length, max_response_tokens, client
import traceback
import asyncio


# defines a function that handles messages sent on discord that the bot can see.
async def on_message(message):
    store_document(message)
    try:
        # first we need to determine whether there is any commands in the message.
        # if there is, we need to handle them and return.
        if message.content.lower().startswith(f"!{bot_name.lower()}"):  # type: ignore
            await handle_command(message)
        else:
            # Check if the message should be responded to
            should_respond = (
                client.user in message.mentions and message.author != client.user
            ) or (is_dm(message) and message.author != client.user)
            # If the message should be responded to, send a response
            if should_respond:
                # fetch and organise the messages for chat completion.
                messages = await generate_completion_messages(message)

                # Send the messages to the OpenAI API
                response = await async_chat_completion_create(
                    model, messages, max_response_tokens
                )
                response = response["choices"][0]["message"]["content"]  # type: ignore

                # Send the response to the channel
                await send_long_discord_message(message, response)

                # print("Message was responded to.")
            else:
                # print("Message not responded to.")
                pass

        # print the alt_relevant_messages to the console if the message author is the dev.

    # If an error occurs, print it to the console.
    except Exception:
        # print(f"Error occurred: {e} \n")
        traceback.print_exc()


# defines a helper function that handles messages bigger than discord handles by default (nitro makes this redundant)
async def send_long_discord_message(message, response):
    if len(response) <= max_discord_message_length:
        await message.channel.send(response)
    else:
        parts = [
            response[i : i + max_discord_message_length]
            for i in range(0, len(response), max_discord_message_length)
        ]
        for part in parts:
            await message.channel.send(part)
            await asyncio.sleep(1)


# defines a function that prints a message to the console when the discord bot is ready
async def on_ready():
    print(f"{client.user} has connected to Discord!")
    # Add a print statement to display connected servers
    print(f"Connected servers: {', '.join([guild.name for guild in client.guilds])}")
