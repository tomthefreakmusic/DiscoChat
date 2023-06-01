# import relevant modules
import openai
import discord
import os
import asyncio
import chromadb
from chromadb.config import Settings
from chromadb.errors import IDAlreadyExistsError
from dotenv import load_dotenv
from rake_nltk import Rake
import tiktoken
import traceback
import atexit
import nltk

# Download nltk data
nltk.download("stopwords")
nltk.download("punkt")
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# Load and set environment variables from .env file
load_dotenv()

required_variables = [
    "DISCORD_TOKEN",
    "OPENAI_API_KEY",
    "BOT_NAME",
    "DATABASE_DIRECTORY",
    "DEV_NAME",
    "SERVER_WHITELIST"
]

for variable in required_variables:
    if os.getenv(variable) is None:
        print(f"{variable} environment variable not set.")
        exit(1)

TOKEN = os.getenv("DISCORD_TOKEN")

# Set OpenAI API key, also set a seperate variable for chroma to use.
openai.api_key = os.getenv("OPENAI_API_KEY")

# establishes the bot name. this is a string.
if os.getenv("BOT_NAME") is None:
    bot_name = "Discochat"
else:
    bot_name = os.getenv("BOT_NAME")

# sets the developer name. this is a string.
if os.getenv("DEV_NAME") is None:
    dev_name = ""
else:
    dev_name = os.getenv("DEV_NAME")

# sets the server whitelist. this is a string.
if os.getenv("SERVER_WHITELIST") is None:
    server_whitelist = ""
else:
    server_whitelist = os.getenv("SERVER_WHITELIST")


# sets the database directory.
if os.getenv("DATABASE_DIRECTORY") is None:
    database_directory = "/database/"
else:
    database_directory = os.getenv("DATABASE_DIRECTORY")

# sets the recent messages section of message history in token length.
recent_messages_length = 1000

# sets the relevant messages section of message history in token length.
relevant_messages_length = 1000

# sets the system message.
system_message = f"You are a helpful AI system named {bot_name}. You are a combination of a vector database (Chroma) and OpenAI's GPT 3.5 Turbo model, integrated into Discord. Recent messages are fetched from Discord, whereas relevant messages are fetched from the vector database. These messages are found in the first message from yourself, seperated by HTML-style formatting tags. It is important to take into consideration both recent messages and relevant messages in your response."

# sets the model.
model = "gpt-3.5-turbo"

# sets the chat completion variables.
max_response_tokens = 1000
temperature = 1.2
frequency_penalty = 0.2
presence_penalty = 0.2

# sets the minimum messages required to be stored before relevant messages can be retrieved.
min_messages_threshold = 5

# creates a ephemeral dictionary for storing the previous relevant messages, these are currently summarized before storage.
previous_relevant_messages = {}

# sets maximum message length (if you have nitro this could be increased)
max_discord_message_length = 2000

# Set discord intents
intents = discord.Intents.default()
intents.messages = True
intents.guild_messages = True
intents.message_content = True

# Sets the client variable
client = discord.Client(intents=intents)

# Sets the token encoder to match the model we are using.
token_encoder = tiktoken.encoding_for_model(model)

# setup chroma and the collection (message_bank)
chromadb_client = chromadb.Client(
    Settings(chroma_db_impl="duckdb+parquet", persist_directory=f"{database_directory}")
)

message_bank = chromadb_client.get_or_create_collection(
    "message_bank", metadata={"hnsw:space": "cosine"}
)


# Defines the store_document function, for storing the discord messages in the chroma database.
def store_document(message):
    if message and message.content:
        message_id, content, metadata = extract_message_data(message)

        try:
            message_bank.add(
                documents=[content],
                metadatas=[metadata],  # type: ignore
                ids=[message_id],
            )
        except IDAlreadyExistsError:
            # If the document with the same id already exists in the database, skip it
            pass
        except Exception as e:
            # Handle other types of exceptions
            print(f"Error adding message to database: {e}")
            traceback.print_exc()


# This function extracts the relevant data from a discord message and returns it in a format that can be stored in the database.
def extract_message_data(message):
    message_id = str(message.id)

    if message.author is client.user:
        author = bot_name
    else:
        author = str(message.author)
    created_at = str(message.created_at)

    content = message.clean_content  # This should be a string, not a list
    # if the content includes mention of bot, we need to remove the mention itself

    if client.user in message.mentions:
        bot_mentioned = "True"
    else:
        bot_mentioned = "False"
    if is_dm:
        server = "DM"
    else:
        server = str(message.guild)
    if message.content.lower().startswith(f"!{bot_name.lower()}"):  # type: ignore
        is_command = "True"
    else:
        is_command = "False"

        # Extracts keywords from the message content.
    r = Rake()
    r.extract_keywords_from_text(message.clean_content)
    keywords = r.get_ranked_phrases()[0:5]

    # Convert keywords list to a single string
    keywords_str = ", ".join(keywords)

    metadata = {
        "message_id": message_id,
        "channel": str(message.channel.id),
        "server": server,
        "author": author,
        "created_at": created_at,
        "keywords": keywords_str,
        "bot_mentioned": bot_mentioned,
        "is_command": is_command,
    }

    return message_id, content, metadata


async def retrieve_relevant_messages(message, token_length, recent_message_ids):
    query = message.clean_content
    channel = str(message.channel.id)
    distance_threshold = 0.8
    bot_penalty = 0.1  # Adjust this to control how much bot messages are penalized

    # Split the query into sentences
    sentences = nltk.tokenize.sent_tokenize(query)

    # Set the where conditions to only search in the channel
    where_conditions = {"$and": [{"channel": channel}, {"is_command": "False"}]}

    relevant_messages = message_bank.query(
        query_texts=sentences,
        n_results=2,
        where=where_conditions,  # type: ignore
    )

    result_string = ""
    seen_messages = set()

    for i in range(len(sentences)):
        ids = relevant_messages["ids"][i]
        documents = relevant_messages["documents"][i]  # type: ignore
        metadatas = relevant_messages["metadatas"][i]  # type: ignore
        distances = relevant_messages["distances"][i]  # type: ignore

        # print(f"\nQuery Sentence: {sentences[i]}")

        for j in range(len(ids)):
            if ids[j] in seen_messages or ids[j] in recent_message_ids:
                continue

            seen_messages.add(ids[j])
            
            distance = distances[j]
            if metadatas[j]["author"] == bot_name:
                distance += bot_penalty  # Increase distance for bot messages

            if distance <= distance_threshold:
                message_id = ids[j]  # Discord message ID is stored in ids
                
                # Fetch message object by id
                message_around = await message.channel.fetch_message(message_id)
                near_messages = [msg async for msg in message.channel.history(limit=3, around=message_around, oldest_first=True)]
                
                for msg in near_messages:
                    if msg.id in seen_messages or msg.id in recent_message_ids:
                        continue

                    seen_messages.add(msg.id)

                    temp_string = f"[{str(msg.created_at)[:-16]}] {msg.author.name}: {msg.clean_content}, "
                    current_message_tokens = len(
                    token_encoder.encode(result_string + temp_string)
                    )

                    if current_message_tokens <= token_length:
                        result_string += temp_string
                        # print(f"Retrieved Message: {temp_string}")
                    else:
                        break  # If adding next message would exceed token limit, break the loop

    result_string = result_string[:-2]

    print(f"{result_string} is of length {len(token_encoder.encode(result_string))}")

    return result_string


# Defines a function that stores relevant messages in a dictionary.
def store_relevant_messages(message, result_string):
    channel = str(message.channel.id)

    messages = [
        {"role": "user", "content": f"summarize these messages: {result_string}"},
    ]

    summary = chat_completion_create(model, messages, max_response_tokens)
    summary = summary["choices"][0]["message"]["content"]  # type: ignore
    previous_relevant_messages[channel] = summary


# Defines a helper function that retrieves the strings from "previous_relevant_messages" for the channel and returns the string.
def retrieve_previous_relevant_messages(message):
    channel = str(message.channel.id)
    if channel in previous_relevant_messages:
        result_string = previous_relevant_messages[channel]
        return result_string
    else:
        return ""


# Defines a helper function that checks if the message is a DM.
def is_dm(message):
    is_dm = isinstance(message.channel, discord.DMChannel)
    return is_dm


# Defines a function that fetches most recent messages from discord based on the token length bounds.
async def retrieve_recent_messages(message, token_length):
    # defines a list to store the history
    recent_messages = []
    recent_message_ids = [message.id]

    message_number = 0
    async for message in message.channel.history(limit=11):
        if message_number == 0:
            message_number += 1
            continue
        message_number += 1

        # for timestamp, we want to strip it back to a useful format
        timestamp = str(message.created_at)[:-16]

        formatted_message = (
            f"[{timestamp}] {message.author.name}: {message.clean_content} "
        )

        current_message_tokens = len(token_encoder.encode(formatted_message))
        if token_length - current_message_tokens < 0:
            break

        # append formatted message to history
        recent_messages.append(formatted_message)
        recent_message_ids.append(message.id)
        token_length -= current_message_tokens

    # reverse the history list so that the messages are in chronological order.
    recent_messages.reverse()
    # print(recent_messages)
    # returns the recent messages from the channel upto the length requested.
    return recent_messages, recent_message_ids



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


# defines a helper function for counting the messages in a channel.
def channel_database_count(message):
    # Query the database for all documents where the channel matches the current channel.
    # As we are not interested in the documents themselves, we only retrieve the metadata.
    channel_id = str(message.channel.id)
    channel_messages = message_bank.get(where={"channel": channel_id})

    # Count the number of messages
    num_messages = len(channel_messages["ids"])

    return num_messages

def check_permissions(message):
    # check if the user has the required role to use the bot, {bot_name} being the role name.
    # if the user is in a whitelisted server, we don't need to check for roles.
    # if the user is messaging in DMs, we don't need to check for roles.
    if is_dm(message):
        return True
    if message.guild.name in server_whitelist:
        return True
    else:
        # iterate through the roles of the user, checking if they have the required role.
        for role in message.author.roles:
            if role.name == bot_name:
                return True
        return False



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
            temperature=temperature,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        ),
    )
    return response


# defines a helper function that handles creation of the messages block of the chat completion.
async def generate_completion_messages(message):

    recent_messages, recent_message_ids = await retrieve_recent_messages(message, recent_messages_length)
    relevant_messages = await retrieve_relevant_messages(message, relevant_messages_length, recent_message_ids)
    timestamp = str(message.created_at)[:-16]

    assistant_message = f"I am responding to the user: {message.author.name}. <recent messages> {recent_messages} </recent messages>, <recalled messages> {relevant_messages} </recalled messages> The time is {timestamp}."
    messages = [
        {"role": "system", "content": system_message},
        {"role": "assistant", "content": assistant_message},
        {"role": "user", "content": f"{message.clean_content}"},
    ]
    #print(f"The user said: {message.clean_content}")
    #print("start of assistant context message.")
    #print(assistant_message)
    #print("end of assistant context message.")
    return messages


# defines a function that handles messages sent on discord that the bot can see.
async def on_message(message):
    # First we'll check roles, if the user doesn't have the required role, we'll return.
    if not check_permissions(message):
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
    else:
        pass


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


# Add the event handlers to the client
client.event(on_ready)
client.event(on_message)

# Run the bot
try:
    if TOKEN is not None:
        client.run(TOKEN)
    else:
        raise ValueError("TOKEN is not set.")
except ValueError as e:
    print(str(e))


# defines a function that saves the chroma database to disk.
def save_database():
    chromadb_client.persist()
    pass


# saves the database on exit (workaround for https://github.com/chroma-core/chroma/issues/622)
atexit.register(save_database)
