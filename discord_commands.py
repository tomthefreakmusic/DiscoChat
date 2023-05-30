from bot_logic import client
from database_storage import store_document, channel_database_count


# Defines a helper function that checks if the message is a command, if it is, it runs the relevant function.
async def handle_command(message):
    if message.author == client.user:
        return
    else:
        command = message.content[11:]
        if command == "populate database":
            await populate_database(message)
        else:
            await message.channel.send("Command not found")


# in this function we are doing our initial populating of the database for the channel. this involves iterating through all prior messages,
async def populate_database(message):
    # get the channel that the message was sent in
    async for message in message.channel.history(limit=500):
        store_document(message)
    await message.channel.send(
        f"Database populated. There are {channel_database_count(message)} messages stored from this channel."
    )
    # print(message_bank.get(where={'channel': str(message.channel.id)}))
    return
