# import relevant modules
import json
import random
import re
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
import textwrap
import time

# Download nltk data
nltk.download("stopwords")
nltk.download("punkt")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Load and set environment variables from .env file
load_dotenv()

required_variables = [
    "DISCORD_TOKEN",
    "OPENAI_API_KEY",
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

# Set OpenAI API key, also set a seperate variable for chroma to use.
openai.api_key = os.getenv("OPENAI_API_KEY")

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
model = "gpt-3.5-turbo-0613"
model_for_function_calls = "gpt-3.5-turbo-0613"

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

# Initialize Rake
r = Rake()

# setup chroma and the collection (message_bank)
chromadb_client = chromadb.Client(
    Settings(chroma_db_impl="duckdb+parquet", persist_directory=f"{database_directory}")
)

message_bank = chromadb_client.get_or_create_collection(
    "message_bank", metadata={"hnsw:space": "cosine"}
)

""" document_bank = chromadb_client.get_or_create_collection(
    "document_bank", metadata={"hnsw:space": "cosine"}
) """

# set up a global dictionary to store the on_message_timings
on_message_timings = {
    "store_message": [],
    "handle_command": [],
    "get_channel_configuration": [],
    "get_query_terms": [],
    "generate_completion_messages": [],
    "create_chat_completion": [],
    "send_long_discord_message": [],
}

generate_completion_messages_timings = {
    "retrieve_recent_messages": [],
    "retrieve_relevant_messages": [],
    "message_assembly": [],
}

retrieve_relevant_messages_timings = {
    "message_bank_query": [],
    "message_around": [],
    "near_messages": [],
    "formatting": [],
}

def calculate_average_latencies(set_of_timings, set_of_timings_name):
    print(f"\n{set_of_timings_name} average latencies:")

    max_section_length = (
        max(len(section) for section in set_of_timings.keys()) + 2
    )  # add 2 for a colon and a space

    for section, times in set_of_timings.items():
        if times:
            average = sum(times) / len(times)
            print(f"{section: <{max_section_length}}: {average * 1000: .0f} ms")


# Defines the store_message function, for storing the discord messages in the chroma database.
def store_message(message):
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
    message, query_terms, token_length, recent_message_ids
):
    query = []
    # get keywords from message.clean_content, add these to the query list
    query.extend(query_terms)
    channel = str(message.channel.id)
    distance_threshold = 0.7
    bot_penalty = 0.4  # Adjust this to control how much bot messages are penalized

    # Split the query into sentences
    # sentences = nltk.tokenize.sent_tokenize(query)

    # Set the where conditions to only search in the channel
    where_conditions = {"$and": [{"channel": channel}, {"is_command": "False"}]}

    start = time.time()

    relevant_messages = message_bank.query(
        query_texts=query,
        n_results=5,
        where=where_conditions,  # type: ignore
    )

    end = time.time()
    retrieve_relevant_messages_timings["message_bank_query"].append(end - start)

    relevant_messages_result = ""

    seen_messages = set()

    for i in range(len(query)):
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
                message_around = None
                try:
                    start = time.time()
                    message_around = await message.channel.fetch_message(message_id)
                    end = time.time()
                    retrieve_relevant_messages_timings["message_around"].append(end - start)
                # Process the message
                except discord.errors.NotFound:
                    # Handle the error, skip this message, or perform any necessary action
                    pass

                start = time.time()
                near_messages = [
                    msg
                    async for msg in message.channel.history(
                        limit=5, around=message_around, oldest_first=True
                    )
                ]
                end = time.time()
                retrieve_relevant_messages_timings["near_messages"].append(end - start)

                start = time.time()
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
                    current_message_tokens = len(
                        token_encoder.encode(relevant_messages_result + temp_string)
                    )

                    if current_message_tokens <= token_length:
                        relevant_messages_result += temp_string

                    else:
                        break  # If adding next message would exceed token limit, break the loop
                end = time.time()
                retrieve_relevant_messages_timings["formatting"].append(end - start)

    relevant_messages_result = relevant_messages_result[:-2]

    """ print(
        f"{relevant_messages_result} is of length {len(token_encoder.encode(relevant_messages_result))}"
    ) """

    return relevant_messages_result


# Defines a function that stores relevant messages in a dictionary.
def summarize(input_text, summary_length=500):
    messages = [
        {"role": "user", "content": f"summarize these messages: {input_text}"},
    ]

    summary = create_non_async_chat_completion(model, messages, summary_length)
    summary = summary["choices"][0]["message"]["content"]  # type: ignore
    return summary


# Defines a function that stores relevant messages in a dictionary.
def summarize_for_context(recent_messages, relevant_messages, summary_length=500):
    messages = [
        {
            "role": "user",
            "content": f"summarize the recalled messages based on what is relevant to the conversation in recent messages. \
        <recalled messages> {relevant_messages} </recalled messages> <recent messages> {recent_messages} </recent messages>.",
        },
    ]

    summary = create_non_async_chat_completion(model, messages, summary_length)
    summary = summary["choices"][0]["message"]["content"]  # type: ignore
    return summary


# Defines a helper function that retrieves the strings from "previous_relevant_messages" for the channel and returns the string.
def retrieve_previous_relevant_messages(message):
    channel = message.channel.id
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
async def retrieve_recent_messages(message, token_length, limit=151):
    # defines a list to store the history
    recent_messages = []
    recent_message_ids = [message.id]
    recent_message_content = ""

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

        current_message_tokens = len(token_encoder.encode(formatted_message))
        if token_length - current_message_tokens < 0:
            break

        # append formatted message to history
        recent_messages.append(formatted_message)
        recent_message_ids.append(message.id)

        # filter out bot messages, as we only want user message keywords.
        # if message.author != client.user:
        #    recent_message_content += message.clean_content + " "
        token_length -= current_message_tokens

    # reverse the history list so that the messages are in chronological order.
    recent_messages.reverse()
    # print(recent_messages)
    # returns the recent messages from the channel upto the length requested.
    return (
        recent_messages,
        recent_message_ids,
    )  # get_keywords(recent_message_content, 5)


# in this function we are doing our initial populating of the database for the channel. this involves iterating through all prior messages,
async def populate_database(message):
    # get the channel that the message was sent in
    async for message in message.channel.history(limit=500):
        store_message(message)
    await message.channel.send(
        f"Database populated. There are {count_channel_database(message)} messages stored from this channel."
    )
    # print(message_bank.get(where={'channel': str(message.channel.id)}))
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
        # command is the message content, stripped of the length of the botname + 2.
        command = message.content[len(bot_name) + 2 :]
        if command.startswith("populate database"):
            await populate_database(message)
        elif command.startswith("count database"):
            await message.channel.send(
                f"There are {count_channel_database(message)} messages stored from this channel."
            )
        elif command.startswith("clear database"):
            await clear_database(message)
        elif command.startswith("set configuration"):
            config_value = command[
                len("set configuration") + 1 :
            ]  # "+1" to account for the space after "set configuration"
            await update_channel_configuration(message, config_value)
        elif command.startswith("reset configuration"):
            await reset_channel_configuration(message)
        else:
            await message.channel.send("Command not found")


def construct_logit_bias(keywords, bias_value, limit=10):
    logit_bias = {}
    if not keywords:
        return logit_bias
    if len(keywords) > limit:  # If keywords list exceeds limit, select a random subset
        keywords = random.sample(keywords, limit)
    for keyword in keywords:
        if " " in keyword:
            # Split the phrase into words
            words = keyword.split(" ")
            for word in words:
                # Convert each word to a token ID
                token_id = token_encoder.encode(word)[0]
                logit_bias[token_id] = bias_value
    else:
        # Convert keyword to token ID
        token_id = token_encoder.encode(keyword)[0]  # type: ignore
        logit_bias[token_id] = bias_value

    return logit_bias


# defines a function for handling chat completion that isn't asynchronous. this is useful for situations where we NEED the response before continuing. database handling etc.
def create_non_async_chat_completion(
    model_for_completion, messages_for_completion, response_tokens
):
    response = openai.ChatCompletion.create(
        model=model_for_completion,
        messages=messages_for_completion,
        max_tokens=response_tokens,
    )
    return response


async def get_channel_configuration(message):
    # Default configuration values
    system_message = f"""You are the AI system named {bot_name}. 
    You are a combination of a vector database (Chroma) and OpenAI's GPT 3.5 Turbo model, integrated into Discord. 
    Recent messages are fetched from Discord, whereas relevant messages are fetched from the vector database. 
    These messages are found in the first message from yourself, separated by HTML-style formatting tags. 
    It is important to take into consideration both recent messages and relevant messages in your response. 
    If the user refers to you, they are referring to the {bot_name} system, not the language model that powers your responses.
    The user is familiar with language models and understands how they work so you do not provide basic explanations.
    The user understands that you are a language model and the limitations that come alongside that."""
    max_response_tokens = 1000
    recent_messages_length = 750
    relevant_messages_length = 500
    temperature = 1.1
    presence_penalty = 0.8
    frequency_penalty = 0.0
    chat_mode = "standard"

    try:
        # check if channel has existing config within the config folder
        if os.path.isfile(f"./config/{message.channel.id}.json"):
            # if channel has existing config, load it
            with open(f"./config/{message.channel.id}.json", "r") as f:
                channel_config = json.load(f)
            # set variables to config values
            system_message = channel_config.get("system_message", system_message)
            max_response_tokens = channel_config.get(
                "max_response_tokens", max_response_tokens
            )
            temperature = channel_config.get("temperature", temperature)
            presence_penalty = channel_config.get("presence_penalty", presence_penalty)
            frequency_penalty = channel_config.get(
                "frequency_penalty", frequency_penalty
            )
            recent_messages_length = channel_config.get(
                "recent_messages_length", recent_messages_length
            )
            relevant_messages_length = channel_config.get(
                "relevant_messages_length", relevant_messages_length
            )
            chat_mode = channel_config.get("chat_mode", chat_mode)
        else:
            raise IOError("Config file not found")

    except (IOError, ValueError) as e:
        # handle file I/O or JSON errors
        print(f"Error handling channel configuration: {e}")
        # ensure config directory exists
        os.makedirs("./config/", exist_ok=True)
        # write new channel configuration to .json in the config folder
        with open(f"./config/{message.channel.id}.json", "w") as f:
            json.dump(
                {
                    "system_message": system_message,
                    "max_response_tokens": max_response_tokens,
                    "temperature": temperature,
                    "presence_penalty": presence_penalty,
                    "frequency_penalty": frequency_penalty,
                    "recent_messages_length": recent_messages_length,
                    "relevant_messages_length": relevant_messages_length,
                    "chat_mode": chat_mode,
                },
                f,
            )

    return (
        system_message,
        max_response_tokens,
        temperature,
        presence_penalty,
        frequency_penalty,
        recent_messages_length,
        relevant_messages_length,
        chat_mode,
    )


async def update_channel_configuration(message, config_value):
    # get channel configuration
    (
        system_message,
        max_response_tokens,
        temperature,
        presence_penalty,
        frequency_penalty,
        recent_messages_length,
        relevant_messages_length,
        chat_mode,
    ) = await get_channel_configuration(message)

    # regex pattern to match "parameter value", capturing both "parameter" and "value" in separate groups
    pattern = re.compile(r"(\w+)\s+(.*)")
    match = pattern.match(config_value)

    if match:
        param, value = match.groups()

        try:
            # update the parameter value based on the parameter name
            if param == "system_message":
                system_message = value
            elif param == "max_response_tokens":
                max_response_tokens = int(value)
            elif param == "temperature":
                temperature = float(value)
            elif param == "presence_penalty":
                presence_penalty = float(value)
            elif param == "frequency_penalty":
                frequency_penalty = float(value)
            elif param == "recent_messages_length":
                recent_messages_length = int(value)
            elif param == "relevant_messages_length":
                relevant_messages_length = int(value)
            elif param == "chat_mode":
                chat_mode = value
            else:
                await message.channel.send("Config parameter not recognized.")
                return

        except ValueError:
            await message.channel.send("Invalid value for the parameter.")
            return

    else:
        await message.channel.send("Could not parse the configuration command.")
        return

    # if we get here, the config command was parsed successfully, and we will send the user a confirmation message
    await message.channel.send(f"Updated {param} to {value}.")

    # write updated channel configuration to .json in the config folder
    with open(f"./config/{message.channel.id}.json", "w") as f:
        json.dump(
            {
                "system_message": system_message,
                "max_response_tokens": max_response_tokens,
                "temperature": temperature,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "recent_messages_length": recent_messages_length,
                "relevant_messages_length": relevant_messages_length,
                "chat_mode": chat_mode,
            },
            f,
        )


async def reset_channel_configuration(message):
    # delete channel configuration file
    os.remove(f"./config/{message.channel.id}.json")
    # set channel back to default configuration
    await get_channel_configuration(message)
    # send confirmation message
    await message.channel.send("Channel configuration reset.")


# defines a function for asynchronously handling chat completion
async def create_chat_completion(
    model_for_completion,
    messages_for_completion,
    response_tokens,
    temperature,
    presence_penalty,
    frequency_penalty,
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


async def get_query_terms(message):
    # first we'll grab the most recent 5 messages from the channel including to the user's message
    recent_messages, _ = await retrieve_recent_messages(message, 1000, 6)
    recent_messages_string = "".join(recent_messages)
    timestamp = str(message.created_at)[:-16]
    user_message = f"[{timestamp}] {message.author.name}: {message.clean_content}"

    messages = [
        {
            "role": "system",
            "content": f"You are a discord chatbot named {bot_name}, you store messages from the conversation history into a vector database for semantic lookup to provide contextually relevant responses.",
        },
        {
            "role": "user",
            "content": f"<recent messages> {recent_messages_string} {user_message}</recent messages> Based on the recent conversation, what are five key terms or topics that could be used to query the vector database",
        },
    ]
    # then we will define what the functions list is
    functions = [
        {
            "name": "query_database",
            "description": "Queries the vector database using given terms",
            "parameters": {
                "type": "object",
                "properties": {"terms": {"type": "array", "items": {"type": "string"}}},
                "required": ["terms"],
            },
        }
    ]

    # now we'll run the chat completion and force the query database function_call, so we can ensure it's behaving as intended.
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: openai.ChatCompletion.create(
            model=model_for_function_calls,
            messages=messages,
            max_tokens=100,
            temperature=0.8,
            functions=functions,
            function_call={"name": "query_database"},
        ),
    )

    arguments = response["choices"][0]["message"]["function_call"][  # type:ignore
        "arguments"
    ]
    arguments_dict = json.loads(arguments)
    terms = arguments_dict.get("terms", [])

    # print(terms)
    return terms


# defines a helper function that handles creation of the messages block of the chat completion.
async def generate_completion_messages(
    message,
    system_message,
    query_terms,
    recent_messages_length,
    relevant_messages_length,
    chat_mode,
):
    start = time.time()
    recent_messages, recent_message_ids = await retrieve_recent_messages(
        message, recent_messages_length
    )
    recent_messages_string = " ".join(recent_messages)
    end = time.time()
    generate_completion_messages_timings["retrieve_recent_messages"].append(end - start)
    if chat_mode == "standard":
        start = time.time()
        relevant_messages = await retrieve_relevant_messages(
            message, query_terms, relevant_messages_length, recent_message_ids
        )
        end = time.time()
        generate_completion_messages_timings["retrieve_relevant_messages"].append(
            end - start
        )
    else:
        relevant_messages = ""
    prior_summary = ""  # retrieve_previous_relevant_messages(message)
    # we will summarize the messages and store them in the dictionary previously_relevant_messages as the value for the channel ID key.
    start = time.time()
    timestamp = str(message.created_at)[:-16]

    if prior_summary != "":
        prior_summary = f"<previously relevant messages> {prior_summary} </previously relevant messages>"
    if recent_messages != "":
        recent_messages = (
            f"<recent messages> {recent_messages_string} </recent messages>"
        )
    if relevant_messages != "":
        relevant_messages = (
            f"<relevant messages> {relevant_messages} </relevant messages>"
        )

    assistant_message = f"I am talking to the user: {message.author.name}.{recent_messages}{relevant_messages}{prior_summary} The time is {timestamp}."

    messages = [
        {"role": "system", "content": system_message},
        {"role": "assistant", "content": assistant_message},
        {"role": "user", "content": f"{message.clean_content}"},
    ]
    end = time.time()
    generate_completion_messages_timings["message_assembly"].append(end - start)
    # print(messages)
    calculate_average_latencies(
        generate_completion_messages_timings, "generate_completion_messages"
    )
    return messages, recent_messages, relevant_messages


# defines a function that handles messages sent on discord that the bot can see.
async def on_message(message):
    # First we'll check roles, if the user doesn't have the required role, we'll return.
    if check_permissions(message) == True:
        start = time.time()
        store_message(message)
        end = time.time()
        on_message_timings["store_message"].append(end - start)
        try:
            # first we need to determine whether there is any commands in the message.
            # if there is, we need to handle them and return.
            if message.content.lower().startswith(f"!{bot_name.lower()}"):
                start = time.time()
                await handle_command(message)
                end = time.time()
                on_message_timings["handle_command"].append(end - start)
            else:
                # Check if the message should be responded to
                should_respond = (
                    client.user in message.mentions and message.author != client.user
                ) or (is_dm(message) and message.author != client.user)
                # If the message should be responded to, send a response
                if should_respond:
                    async with message.channel.typing():
                        # fetch and organise the messages for chat completion.
                        start = time.time()
                        (
                            system_message,
                            max_response_tokens,
                            temperature,
                            presence_penalty,
                            frequency_penalty,
                            recent_messages_length,
                            relevant_messages_length,
                            chat_mode,
                        ) = await get_channel_configuration(message)
                        end = time.time()
                        on_message_timings["get_channel_configuration"].append(
                            end - start
                        )
                        if chat_mode == "standard":
                            start = time.time()
                            query_terms = await get_query_terms(message)
                            end = time.time()
                            on_message_timings["get_query_terms"].append(end - start)
                        else:
                            query_terms = []
                        start = time.time()
                        (
                            messages,
                            recent_messages,
                            relevant_messages,
                        ) = await generate_completion_messages(
                            message,
                            system_message,
                            query_terms,
                            recent_messages_length,
                            relevant_messages_length,
                            chat_mode,
                        )
                        end = time.time()
                        on_message_timings["generate_completion_messages"].append(
                            end - start
                        )

                        if chat_mode == "long context":
                            model_for_responses = "gpt-3.5-turbo-16k"
                        else:
                            model_for_responses = model

                        start = time.time()
                        # Send the messages to the OpenAI API
                        response = await create_chat_completion(
                            model_for_responses,
                            messages,
                            max_response_tokens,
                            temperature,
                            presence_penalty,
                            frequency_penalty,
                        )
                        response = response["choices"][0]["message"]["content"]  # type: ignore
                        end = time.time()
                        on_message_timings["create_chat_completion"].append(end - start)

                    # Send the response to the channel
                    start = time.time()
                    await send_long_discord_message(message, response)
                    end = time.time()
                    on_message_timings["send_long_discord_message"].append(end - start)

                    calculate_average_latencies(on_message_timings, "on_message")

                else:
                    # print("Message not responded to.")
                    pass

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
        # split the response into lines no longer than max_discord_message_length
        # break_long_words=False and replace_whitespace=False ensure words are not split
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
