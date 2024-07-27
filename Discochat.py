import json
import random
import re
import traceback
import anthropic
import discord
from discord.ui import View, Button
import os
import asyncio
import chromadb
from chromadb.config import Settings
from chromadb.errors import IDAlreadyExistsError
from dotenv import load_dotenv
from rake_nltk import Rake
import nltk
import textwrap
import atexit
import ast

# Download nltk data
nltk.download("stopwords")
nltk.download("punkt")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Load and set environment variables from .env file
load_dotenv()

required_variables = [
    "DISCORD_TOKEN",
    "ANTHROPIC_API_KEY",
    "BOT_NAME",
    "DATABASE_DIRECTORY",
    "DEV_NAME",
    "SERVER_WHITELIST",
]

for variable in required_variables:
    if os.getenv(variable) is None:
        print(f"{variable} environment variable not set.")
        exit(1)

TOKEN = os.getenv("DISCORD_TOKEN")

# Set Anthropic API key
anthropic_client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))

bot_name = os.getenv("BOT_NAME")
assert bot_name is not None, "Environment variable BOT_NAME is not set"
if not bot_name:
    bot_name = "Discochat"

# sets the developer name. this is a string.
if os.getenv("DEV_NAME") is None:
    dev_name = ""
else:
    dev_name = os.getenv("DEV_NAME")

# sets the server whitelist. this is a string.
if os.getenv("SERVER_WHITELIST") is None:
    server_whitelist = []
else:
    server_whitelist = os.getenv("SERVER_WHITELIST")

# sets the database directory.
if os.getenv("DATABASE_DIRECTORY") is None:
    database_directory = "/database/"
else:
    database_directory = os.getenv("DATABASE_DIRECTORY")

# sets the model.
model = "claude-3-5-sonnet-20240620"

CHAT_MODE_PRESETS = {
    "Default": {
        "recent_messages_length": 2000,
        "relevant_messages_length": 0,
        "sporadic_messages_length": 0,
        "max_response_tokens": 200,
        "system_message": "You are a helpful AI assistant.",
        "temperature": 0.8
    },
    "Memory": {
        "recent_messages_length": 2000,
        "relevant_messages_length": 2000,
        "sporadic_messages_length": 0,
        "max_response_tokens": 500,
        "system_message": "You are a helpful AI assistant with access to conversation history.",
        "temperature": 0.8
    },
    "Day Dream": {
        "recent_messages_length": 1000,
        "relevant_messages_length": 0,
        "sporadic_messages_length": 3000,
        "max_response_tokens": 200,
        "system_message": "You are a creative AI assistant, feel free to be imaginative in your responses.",
        "temperature": 1.0
    }
}

# sets the minimum messages required to be stored before relevant messages can be retrieved.
min_messages_threshold = 5

# creates a ephemeral dictionary for storing the previous relevant messages, these are currently summarized before storage.
previously_relevant_messages = {}

# sets maximum message length (if you have nitro this could be increased)
max_discord_message_length = 2000

# Set discord intents
intents = discord.Intents.default()
intents.messages = True
intents.guild_messages = True
intents.message_content = True

# Sets the client variable
client = discord.Client(intents=intents)

# Initialize Rake
r = Rake()

# setup chroma and the collection (message_bank)
chromadb_client = chromadb.Client(
    Settings(chroma_db_impl="duckdb+parquet", persist_directory=f"{database_directory}")
)

message_bank = chromadb_client.get_or_create_collection(
    "message_bank", metadata={"hnsw:space": "cosine"}
)

# Defines the store_message function, for storing the discord messages in the chroma database.
def store_message(message):
    if message and message.content:
        message_id, content, metadata = extract_message_data(message)

        try:
            message_bank.add(
                documents=[content],
                metadatas=[metadata],
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

    content = message.clean_content

    if client.user in message.mentions:
        bot_mentioned = "True"
    else:
        bot_mentioned = "False"
    if is_dm(message):
        server = "DM"
    else:
        server = str(message.guild)
    if message.content.lower().startswith(f"!{bot_name.lower()}"):
        is_command = "True"
    else:
        is_command = "False"

    # Extracts keywords from the message content.
    keywords = get_keywords(message.clean_content)

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

def get_keywords(text, num_keywords=5):
    r.extract_keywords_from_text(text)
    return r.get_ranked_phrases()[0:num_keywords]

async def retrieve_relevant_messages(
    message, query_terms, token_length, recent_message_ids=None
):
    if recent_message_ids is None:
        recent_message_ids = []
    if not query_terms:
        return ""
    query = []
    # get keywords from message.clean_content, add these to the query list
    query.extend(query_terms)
    channel = str(message.channel.id)
    distance_threshold = 0.7
    bot_penalty = 0.4  # Adjust this to control how much bot messages are penalized

    # Set the where conditions to only search in the channel
    where_conditions = {"$and": [{"channel": channel}, {"is_command": "False"}]}

    relevant_messages = message_bank.query(
        query_texts=query,
        n_results=5,
        where=where_conditions,
    )

    relevant_messages_result = ""

    seen_messages = set()

    for i in range(len(query)):
        ids = relevant_messages["ids"][i]
        documents = relevant_messages["documents"][i]
        metadatas = relevant_messages["metadatas"][i]
        distances = relevant_messages["distances"][i]

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
                message_around = None
                try:
                    message_around = await message.channel.fetch_message(message_id)

                # Process the message
                except discord.errors.NotFound:
                    # Handle the error, skip this message, or perform any necessary action
                    pass

                near_messages = [
                    msg
                    async for msg in message.channel.history(
                        limit=5, around=message_around, oldest_first=True
                    )
                ]

                for msg in near_messages:
                    if msg.id in seen_messages or msg.id in recent_message_ids:
                        continue

                    # we are going to split the message into words, and if any word is longer than 28 characters, we will truncate the word to that limit and add a "..." to the end
                    # this is to prevent the model from crashing due to too many tokens
                    message_content = msg.clean_content

                    for word in message_content.split():
                        if len(word) > 28:
                            message_content = message_content.replace(
                                word, word[:28] + "..."
                            )

                    seen_messages.add(msg.id)

                    temp_string = f"[{str(msg.created_at)[:-16]}] {msg.author.name}: {message_content}, "
                    current_message_tokens = len(temp_string)

                    if current_message_tokens <= token_length:
                        relevant_messages_result += temp_string
                        token_length -= current_message_tokens
                    else:
                        break  # If adding next message would exceed token limit, break the loop

    relevant_messages_result = relevant_messages_result[:-2]

    return relevant_messages_result

async def retrieve_sporadic_messages(message, token_length):
    channel = str(message.channel.id)
    where_conditions = {"$and": [{"channel": channel}, {"is_command": "False"}]}

    sporadic_messages = message_bank.get(
        where=where_conditions, include=["metadatas", "documents"]
    )

    sporadic_messages_result = ""
    message_indices = list(range(len(sporadic_messages["ids"])))
    random.shuffle(message_indices)

    for i in message_indices:
        id = sporadic_messages["ids"][i]
        document = sporadic_messages["documents"][i]
        metadata = sporadic_messages["metadatas"][i]

        message_content = document

        # Truncate long words
        for word in message_content.split():
            if len(word) > 28:
                message_content = message_content.replace(word, word[:28] + "...")

        created_at = str(metadata["created_at"])[:-16]
        author = metadata["author"]

        temp_string = f"[{created_at}] {author}: {message_content}, "
        current_message_tokens = len(temp_string)

        if current_message_tokens <= token_length:
            sporadic_messages_result += temp_string
            token_length -= current_message_tokens
        else:
            break  # If adding next message would exceed token limit, break the loop

    sporadic_messages_result = sporadic_messages_result[:-2]
    return sporadic_messages_result

# Defines a function that stores relevant messages in a dictionary.
def summarize(input_text, summary_length=500):
    messages = [
        anthropic.Message(role="user", content=f"summarize these messages: {input_text}"),
    ]

    summary = anthropic_client.messages.create(
        model=model,
        max_tokens=summary_length,
        messages=messages
    )
    return summary.content

# Defines a function that stores relevant messages in a dictionary.
def summarize_for_context(recent_messages, relevant_messages, summary_length=500):
    messages = [
        anthropic.Message(
            role="user",
            content=f"summarize the recalled messages based on what is relevant to the conversation in recent messages. \
        <recalled messages> {relevant_messages} </recalled messages> <recent messages> {recent_messages} </recent messages>."
        ),
    ]

    summary = anthropic_client.messages.create(
        model=model,
        max_tokens=summary_length,
        messages=messages
    )
    return summary.content

# Defines a helper function that retrieves the strings from "previous_relevant_messages" for the channel and returns the string.
def retrieve_previously_relevant_messages(message):
    channel = message.channel.id
    if channel in previously_relevant_messages:
        result_string = previously_relevant_messages[channel]
        return result_string
    else:
        return ""

# Defines a helper function that checks if the message is a DM.
def is_dm(message):
    is_dm = isinstance(message.channel, discord.DMChannel)
    return is_dm

# Defines a function that fetches most recent messages from discord based on the token length bounds.
async def retrieve_recent_messages(message, token_length, limit=151):
    # defines a list to store the history
    recent_messages = []
    recent_message_ids = [message.id]

    message_number = 0
    async for message in message.channel.history(limit=limit):
        if message_number == 0:
            message_number += 1
            continue
        message_number += 1

        # for timestamp, we want to strip it back to a useful format
        timestamp = str(message.created_at)[:-16]

        message_content = message.clean_content
        for word in message_content.split():
            if len(word) > 28:
                message_content = message_content.replace(word, word[:28] + "...")

        formatted_message = f"[{timestamp}] {message.author.name}: {message_content} "

        current_message_tokens = len(formatted_message)
        if token_length - current_message_tokens < 0:
            break

        # append formatted message to history
        recent_messages.append(formatted_message)
        recent_message_ids.append(message.id)

        token_length -= current_message_tokens

    # reverse the history list so that the messages are in chronological order.
    recent_messages.reverse()
    # returns the recent messages from the channel upto the length requested.
    return recent_messages, recent_message_ids

# in this function we are doing our initial populating of the database for the channel. this involves iterating through all prior messages,
async def populate_database(message):
    # get the channel that the message was sent in
    async for message in message.channel.history(limit=500):
        store_message(message)
    await message.channel.send(
        f"Database populated. There are {count_channel_database(message)} messages stored from this channel."
    )
    return

async def clear_database(message):
    message_bank.delete(where={"channel": str(message.channel.id)})
    await message.channel.send(
        f"Database cleared. There are {count_channel_database(message)} messages stored from this channel."
    )
    return

# defines a helper function for counting the messages in a channel.
def count_channel_database(message):
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
        # if the user is the bot then we don't need to check for roles.
        if message.author == client.user:
            return True
        return False

# Defines a helper function that checks if the message is a command, if it is, it runs the relevant function.
async def handle_command(message):
    if message.author == client.user:
        return
    else:
        command = message.content[len(bot_name) + 2 :]
        if command.startswith("populate database"):
            await populate_database(message)
        elif command.startswith("count database"):
            await message.channel.send(
                f"There are {count_channel_database(message)} messages stored from this channel."
            )
        elif command.startswith("clear database"):
            await clear_database(message)
        elif command.startswith("set chat mode"):
            await create_chat_mode_form(message)
        elif command.startswith("show configuration"):
            config, chat_mode = await get_channel_configuration(message)
            formatted_config = format_channel_config(config, chat_mode)
            await message.channel.send(formatted_config)
        else:
            await message.channel.send("Command not found")

async def get_channel_configuration(message):
    try:
        if os.path.isfile(f"./config/{message.channel.id}.json"):
            with open(f"./config/{message.channel.id}.json", "r") as f:
                channel_config = json.load(f)
            chat_mode = channel_config.get("chat_mode", "Default")
        else:
            chat_mode = "Default"

        config = CHAT_MODE_PRESETS[chat_mode]
        return config, chat_mode

    except (IOError, ValueError, KeyError) as e:
        print(f"Error handling channel configuration: {e}")
        os.makedirs("./config/", exist_ok=True)
        with open(f"./config/{message.channel.id}.json", "w") as f:
            json.dump({"chat_mode": "Default"}, f)
        return CHAT_MODE_PRESETS["Default"], "Default"

async def update_channel_configuration(message, new_chat_mode):
    if new_chat_mode not in CHAT_MODE_PRESETS:
        await message.channel.send(f"Invalid chat mode. Available modes: {', '.join(CHAT_MODE_PRESETS.keys())}")
        return

    with open(f"./config/{message.channel.id}.json", "w") as f:
        json.dump({"chat_mode": new_chat_mode}, f)

    await message.channel.send(f"Updated chat mode to {new_chat_mode}.")

async def create_chat_mode_form(message):
    view = View()
    for mode in CHAT_MODE_PRESETS.keys():
        button = Button(label=mode, style=discord.ButtonStyle.primary)
        button.callback = lambda interaction, m=mode: handle_button_click(interaction, m)
        view.add_item(button)

    await message.channel.send("Choose a chat mode:", view=view)

async def handle_button_click(interaction, mode):
    await update_channel_configuration(interaction.message, mode)
    await interaction.response.send_message(f"Chat mode updated to {mode}", ephemeral=True)

async def reset_channel_configuration(message):
    # delete channel configuration file
    os.remove(f"./config/{message.channel.id}.json")
    # set channel back to default configuration
    await get_channel_configuration(message)
    # send confirmation message
    await message.channel.send("Channel configuration reset.")

async def show_channel_configuration(message):
    config = await get_channel_configuration(message)
    formatted_config = format_channel_config(dict(zip(
        ['system_message', 'max_response_tokens', 'temperature', 'recent_messages_length', 'relevant_messages_length', 'chat_mode'],
        config
    )))
    await message.channel.send(formatted_config)

def format_channel_config(config, chat_mode):
    formatted = f"**Current Chat Mode: {chat_mode}**\n\n"
    for key, value in config.items():
        if key == 'system_message':
            formatted += f"**{key}:**\n```\n{value}\n```\n"
        else:
            formatted += f"**{key}:** {value}\n"
    return formatted


async def get_query_terms(message):
    config, chat_mode = await get_channel_configuration(message)
    
    if chat_mode != "Memory":
        return []

    recent_messages, _ = await retrieve_recent_messages(message, 1000, 6)
    recent_messages_string = "".join(recent_messages)
    timestamp = str(message.created_at)[:-16]
    user_message = f"[{timestamp}] {message.author.name}: {message.clean_content}"

    system_content = f"""You are an AI assistant that extracts key terms for querying a vector database. 
    Your task is to analyze the recent conversation and output up to 5 key terms or phrases, each on a new line. 
    Only include terms that are directly relevant to the main topics of the conversation.
    If there are fewer than 5 relevant topics, output fewer terms.
    These terms should be concise and directly usable for database queries.
    Do not include any explanations or additional text."""

    user_content = f"Based on the following recent conversation, provide up to 5 key terms or phrases for querying a vector database:\n\n{recent_messages_string}{user_message}"

    response = anthropic_client.messages.create(
        model=model,
        max_tokens=100,
        temperature=0.7,
        system=system_content,
        messages=[
            {"role": "user", "content": user_content}
        ]
    )

    # Extract terms from the response
    terms = [term.strip() for term in response.content[0].text.strip().split('\n') if term.strip()]
    
    # Filter out any terms that are just numbers or single characters
    terms = [term for term in terms if len(term) > 1 and not term.isdigit()]

    return terms  # This will return all valid terms, up to 5

async def generate_completion_messages(
    message,
    system_message,
    query_terms,
    recent_messages_length,
    relevant_messages_length,
    sporadic_messages_length,
    chat_mode,
):
    # Map chat modes to corresponding retrieval functions
    retrieval_functions = {
        "Default": [
            ("recent messages", retrieve_recent_messages, False, False),
        ],
        "Memory": [
            ("recent messages", retrieve_recent_messages, False, False),
            ("relevant messages", retrieve_relevant_messages, True, True),
        ],
        "Day Dream": [
            ("recent messages", retrieve_recent_messages, False, False),
            ("sporadic messages", retrieve_sporadic_messages, False, False),
        ],
    }

    # Generate message tags
    message_tags = {}

    # Initialize recent_message_ids
    recent_message_ids = []

    print(f"Chat mode: {chat_mode}")  # Print the chat mode
    print(f"Query terms: {query_terms}")  # Print the query terms

    # Get messages based on chat mode
    functions_for_mode = retrieval_functions.get(
        chat_mode, [("default", do_nothing, False, False)]
    )
    for tag, function, requires_ids, requires_query_terms in functions_for_mode:
        print(
            f"Function for mode: {function.__name__}"
        )  # Print the function that should be called
        if function == retrieve_recent_messages:
            recent_messages, recent_message_ids = await function(
                message, recent_messages_length
            )
            message_content = " ".join(recent_messages)
        elif function == retrieve_relevant_messages:
            message_content = await function(
                message, query_terms, relevant_messages_length, recent_message_ids
        )
        elif function == retrieve_sporadic_messages:
            message_content = await function(message, sporadic_messages_length)
        else:
            print(f"Skipping function for {tag} because query_terms is empty.")
            continue

        print(
            f"Function for {tag} returned: {message_content}"
        )  # Print the content that was returned

        if message_content:  # Check if the content is not empty
            message_tags[f"<{tag}>"] = message_content

    # Add all tags to the message
    assistant_message = f"I am talking to the user: {message.author.name}."
    for tag, content in message_tags.items():
        if content:
            assistant_message += f" {tag} {content} {tag.replace('<', '</')}"

    # Construct the context message
    context_message = "Chat context and history: "
    for tag, content in message_tags.items():
        if content:
            context_message += f"{tag} {content} {tag.replace('<', '</')} "

    assistant_message += f" The time is {str(message.created_at)[:-16]}."
    # Construct the message array
    messages = [
        {"role": "user", "content": context_message},
        {"role": "assistant", "content": f"Thank you for providing the context. I'll keep that in mind for our conversation."},
        {"role": "user", "content": message.clean_content}
    ]

    # Extract the recent and relevant messages from the message tags for the return statement
    recent_messages = message_tags.get("<recent messages>", "")
    relevant_messages = message_tags.get("<relevant messages>", "")

    return messages, recent_messages, relevant_messages, system_message

async def do_nothing(*args, **kwargs):
    return ""

# defines a function for handling messages.
async def on_message(message):
    if check_permissions(message):
        store_message(message)
        try:
            if await is_command(message):
                await handle_command(message)
            elif await should_respond(message):
                await respond_to_message(message)
        except Exception as e:
            handle_exception(e)

# defines a helper function for checking if a message is a command.
async def is_command(message):
    return message.content.lower().startswith(f"!{bot_name.lower()}")

# defines a helper function for checking whether a message should be responded to.
async def should_respond(message):
    return (client.user in message.mentions and message.author != client.user) or (
        is_dm(message) and message.author != client.user
    )

# defines a helper function for handling responses.
async def respond_to_message(message):
    async with message.channel.typing():
        config, chat_mode = await get_channel_configuration(message)
        
        query_terms = (
            await get_query_terms(message) if chat_mode in {"Memory"} else []
        )

        completion_messages, _, _, system_message = await generate_completion_messages(
            message,
            config["system_message"],
            query_terms,
            config["recent_messages_length"],
            config["relevant_messages_length"],
            config["sporadic_messages_length"],
            chat_mode,
        )
        response = await get_response(
            completion_messages,
            config["system_message"],
            config["max_response_tokens"],
            config["temperature"],
        )
        await send_long_discord_message(message, response)

async def get_response(
    messages,
    system_message,
    max_response_tokens,
    temperature,
):
    try:
        response = anthropic_client.messages.create(
            model=model,
            max_tokens=max_response_tokens,
            temperature=temperature,
            system=system_message,
            messages=messages
        )
        return response.content[0].text

    except anthropic.APIError as e:
        print(f"Anthropic API error: {e}")
        return "Sorry, there was an issue with the request. Please try again later."

    except Exception as e:  # This will catch any other exceptions
        print(f"Non-API error occurred: {e}")
        return "Sorry, an unexpected error occurred. Please try again later."

def handle_exception(e):
    print(f"Error occurred: {e} \n")
    traceback.print_exc()

# defines a helper function that handles messages bigger than discord handles by default (nitro makes this redundant)
async def send_long_discord_message(message, response):

    # Replace escaped newlines with actual newlines
    response = response.replace('\\n', '\n')

    if len(response) <= max_discord_message_length:
        await message.channel.send(response)
    else:
        parts = textwrap.wrap(
            response,
            max_discord_message_length,
            break_long_words=False,
            replace_whitespace=False,
        )

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