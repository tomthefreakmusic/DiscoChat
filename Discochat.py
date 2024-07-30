import json
import random
import re
import traceback
import anthropic
from anthropic import AsyncAnthropic
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
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
from collections import Counter

# Set up logger
logger = logging.getLogger('discochat')
logger.setLevel(logging.DEBUG)

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Generate a unique filename for this run
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f'logs/discochat_{current_time}.log'

# Create file handler which logs even debug messages
file_handler = RotatingFileHandler(log_filename, maxBytes=5*1024*1024, backupCount=5)
file_handler.setLevel(logging.DEBUG)

# Create console handler with a higher log level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Log the start of the script
logger.info(f"Script started. Logging to {log_filename}")

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
async_anthropic_client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

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
        "max_response_tokens": 1000,
        "system_message": "You are a helpful AI assistant.",
        "temperature": 0.8
    },
    "Memory": {
        "recent_messages_length": 2000,
        "relevant_messages_length": 2000,
        "sporadic_messages_length": 0,
        "max_response_tokens": 1000,
        "system_message": "You are a helpful AI assistant with access to conversation history.",
        "temperature": 0.8
    },
    "Day Dream": {
        "recent_messages_length": 1000,
        "relevant_messages_length": 0,
        "sporadic_messages_length": 3000,
        "max_response_tokens": 1000,
        "system_message": "You are a creative AI assistant, feel free to be imaginative in your responses.",
        "temperature": 1.0
    },
    "Extended Memory": {
        "recent_messages_length": 5000,  # Increased to capture more context
        "summary_recent_messages_length": 10000,
        "relevant_messages_length": 10000,  # Increased to capture more relevant messages
        "sporadic_messages_length": 0,
        "max_response_tokens": 1000,
        "system_message": "You are a helpful AI assistant with access to an extended conversation history.",
        "temperature": 0.8,
        "summary_model": "claude-3-haiku-20240307",  # Use a smaller model for summarization
        "summary_max_tokens": 800  # Limit the summary length
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

def cleanup_database():
    logger.info("Starting database cleanup process")
    all_messages = message_bank.get()
    low_quality_ids = []
    total_messages = len(all_messages['ids'])
    failed_deletions = []

    logger.info(f"Total messages in database: {total_messages}")

    for i, (document, metadata) in enumerate(zip(all_messages['documents'], all_messages['metadatas'])):
        if not is_quality_content(document, metadata):
            low_quality_ids.append(all_messages['ids'][i])
        
        if i % 1000 == 0:
            logger.info(f"Processed {i+1}/{total_messages} messages")

    if low_quality_ids:
        logger.info(f"Attempting to remove {len(low_quality_ids)} low-quality entries from the database")
        for id_to_delete in low_quality_ids:
            try:
                message_bank.delete(ids=[id_to_delete])
            except KeyError:
                failed_deletions.append(id_to_delete)
                logger.warning(f"Failed to delete message with ID: {id_to_delete}")
            except Exception as e:
                failed_deletions.append(id_to_delete)
                logger.error(f"Error deleting message with ID {id_to_delete}: {str(e)}")

        successfully_deleted = len(low_quality_ids) - len(failed_deletions)
        logger.info(f"Successfully removed {successfully_deleted} low-quality entries from the database")
        if failed_deletions:
            logger.warning(f"Failed to delete {len(failed_deletions)} entries. IDs: {failed_deletions}")
    else:
        logger.info("No low-quality entries found in the database")

    logger.info("Database cleanup process completed")

def is_quality_content(text, metadata):
    # Remove any non-alphanumeric characters and split into words
    words = re.findall(r'\w+', text.lower())
    
    # Count total words and unique words
    total_words = len(words)
    unique_words = len(set(words))
    
    # Calculate word diversity ratio
    diversity_ratio = unique_words / total_words if total_words > 0 else 0

    # Check for repetitive patterns
    word_counts = Counter(words)
    max_repetition = max(word_counts.values()) if word_counts else 0
    repetition_ratio = max_repetition / total_words if total_words > 0 else 1

    # Check metadata for spam indicators
    is_spam = 'spam' in metadata.get('keywords', '').lower()

    # Log the quality metrics for debugging
    logger.debug(f"Message quality metrics - Total words: {total_words}, Unique words: {unique_words}, "
                 f"Diversity ratio: {diversity_ratio:.2f}, Repetition ratio: {repetition_ratio:.2f}, "
                 f"Is spam: {is_spam}")

    # Determine if the content is low quality
    is_low_quality = (
        total_words < 5 or
        diversity_ratio < 0.4 or
        repetition_ratio > 0.5 or
        is_spam
    )

    if is_low_quality:
        logger.debug(f"Low quality message detected: {text[:100]}...")

    return not is_low_quality

def get_keywords(text, num_keywords=5):
    r.extract_keywords_from_text(text)
    return r.get_ranked_phrases()[0:num_keywords]

def estimate_tokens(text):
    # Rough estimation: 1 token ~= 4 characters
    return len(text) // 4

async def retrieve_relevant_messages(message, query_terms, token_length, recent_message_ids=None):
    logger.info(f"Starting retrieve_relevant_messages for channel {message.channel.id}")
    logger.debug(f"Query terms: {query_terms}")
    logger.debug(f"Token length: {token_length}")
    logger.debug(f"Recent message IDs: {recent_message_ids}")

    if not query_terms or recent_message_ids is None:
        logger.warning("No query terms or recent message IDs provided")
        return ""

    channel = str(message.channel.id)
    distance_threshold = 0.85
    bot_penalty = 0.4
    
    where_conditions = {"$and": [{"channel": channel}, {"is_command": "False"}]}

    try:
        relevant_messages = message_bank.query(
            query_texts=query_terms,
            n_results=10,
            where=where_conditions,
        )
        logger.debug(f"Raw query results: {relevant_messages}")
    except Exception as e:
        logger.error(f"Error querying message bank: {e}", exc_info=True)
        return ""

    if relevant_messages is None or not relevant_messages:
        logger.warning("Query returned None or empty result")
        return ""

    logger.debug(f"Type of relevant_messages: {type(relevant_messages)}")
    logger.debug(f"Keys in relevant_messages: {relevant_messages.keys()}")
    
    required_keys = ['ids', 'documents', 'metadatas', 'distances']
    for key in required_keys:
        if key not in relevant_messages or not relevant_messages[key]:
            logger.warning(f"Missing or empty required key in query results: {key}")
            return ""

    list_lengths = [len(relevant_messages[key]) for key in required_keys]
    if len(set(list_lengths)) != 1:
        logger.warning(f"Inconsistent list lengths in query results: {list_lengths}")
        return ""

    relevant_messages_result = []
    seen_messages = set(recent_message_ids)
    quality_messages_count = 0

    try:
        for i in range(len(relevant_messages['ids'])):
            message_id = relevant_messages['ids'][i][0]
            document = relevant_messages['documents'][i][0]
            metadata = relevant_messages['metadatas'][i][0]
            distance = relevant_messages['distances'][i][0]

            logger.debug(f"Processing query result: message_id={message_id}, distance={distance}")

            if message_id in seen_messages:
                logger.debug(f"Skipping already seen message {message_id}")
                continue

            seen_messages.add(message_id)

            if metadata["author"] == bot_name:
                distance += bot_penalty
                logger.debug(f"Applied bot penalty to message {message_id}, new distance: {distance}")

            if distance > distance_threshold:
                logger.debug(f"Skipping message {message_id} due to distance {distance} > threshold {distance_threshold}")
                continue

            if not is_quality_content(document, metadata):
                logger.debug(f"Skipping low-quality message {message_id}")
                continue

            try:
                message_around = await message.channel.fetch_message(message_id)
                logger.debug(f"Fetched message {message_id}")
            except discord.errors.NotFound:
                logger.warning(f"Message with id {message_id} not found")
                continue
            except Exception as e:
                logger.error(f"Error fetching message {message_id}: {e}", exc_info=True)
                continue

            try:
                near_messages = [msg async for msg in message.channel.history(limit=5, around=message_around, oldest_first=True)]
                logger.debug(f"Fetched {len(near_messages)} near messages for message {message_id}")
            except Exception as e:
                logger.error(f"Error fetching message history: {e}", exc_info=True)
                continue

            block_messages = []
            for msg in near_messages:
                if msg.id in seen_messages:
                    logger.debug(f"Skipping already seen near message {msg.id}")
                    continue

                seen_messages.add(msg.id)

                message_content = re.sub(r'\S{29,}', lambda m: m.group(0)[:28] + '...', msg.clean_content)
                temp_string = f"[{str(msg.created_at)[:-16]}] {msg.author.name}: {message_content}, "
                
                estimated_tokens = estimate_tokens(temp_string)
                if token_length - estimated_tokens < 0:
                    logger.debug(f"Token limit reached, breaking loop")
                    break

                block_messages.append(temp_string)
                token_length -= estimated_tokens
                logger.debug(f"Added message {msg.id} to block, remaining tokens: {token_length}")

            if block_messages:
                relevant_messages_result.append("".join(block_messages)[:-2])
                quality_messages_count += 1

            if quality_messages_count >= 5:
                logger.debug(f"Reached 5 quality message blocks, stopping retrieval")
                break

    except Exception as e:
        logger.error(f"Error processing query results: {e}", exc_info=True)
        return ""

    result = "\n\n".join(relevant_messages_result)
    logger.info(f"retrieve_relevant_messages completed, returned {len(result)} characters in {quality_messages_count} blocks")
    return result

def get_relative_time(timestamp_str, current_time):
    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M")
    delta = current_time - timestamp
    
    if delta.days == 0:
        if delta.seconds < 60:
            return "just now"
        elif delta.seconds < 3600:
            return f"{delta.seconds // 60} minutes ago"
        else:
            return f"{delta.seconds // 3600} hours ago"
    elif delta.days == 1:
        return "yesterday"
    elif delta.days < 7:
        return f"{delta.days} days ago"
    elif delta.days < 30:
        return f"{delta.days // 7} weeks ago"
    elif delta.days < 365:
        return f"{delta.days // 30} months ago"
    else:
        return f"{delta.days // 365} years ago"

def process_messages(messages, message_type, current_time):
    processed = []
    last_timestamp = None
    group = []
    significant_time_gap = timedelta(hours=1)

    for line in messages.split('\n'):
        if line.startswith('['):
            parts = line.split(']', 1)
            if len(parts) == 2:
                timestamp_str = parts[0][1:]
                content = parts[1].strip()
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M")
                
                if last_timestamp and (timestamp - last_timestamp > significant_time_gap):
                    if group:
                        processed.append((group, get_relative_time(last_timestamp.strftime("%Y-%m-%d %H:%M"), current_time)))
                        group = []
                    processed.append(f"[{get_relative_time(timestamp_str, current_time)}]")
                
                username, message = content.split(':', 1)
                group.append((username.strip(), message.strip(), timestamp_str))
                last_timestamp = timestamp
        else:
            if group:
                group[-1] = (group[-1][0], group[-1][1] + " " + line.strip(), group[-1][2])

    if group:
        processed.append((group, get_relative_time(last_timestamp.strftime("%Y-%m-%d %H:%M"), current_time)))

    return processed

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

async def summarize_extended_context(all_recent_messages, relevant_messages, summary_model, summary_max_tokens, full_recent_messages_count):
    current_time = datetime.now()
    logger.info(f"Starting extended context summarization. Current time: {current_time}")

    # Create a unique filename for this summary input
    timestamp = current_time.strftime("%Y%m%d_%H%M%S_%f")
    filename = f"summary_input_{timestamp}.txt"
    log_dir = "logs/summary_inputs"
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, filename)

    async def summarize_block(block, block_type, max_tokens):
        system_content = f"""You are assisting the chatbot "{bot_name}" by summarizing a block of {block_type} messages.
        Create a concise summary of the provided content without adding new information or interpretations.
        Focus on key points, topics, and information present in the given text.
        Use bullet points to structure your summary."""

        user_content = f"""Summarize this block of {block_type} messages:

        {block}

        Provide a concise summary focusing on main topics and key information.
        Label notes with user names for clarity.
        The current date and time is {current_time.strftime("%Y-%m-%d %H:%M")}.
        Use relative time information if provided to give context on when these conversations occurred.
        """

        response = await async_anthropic_client.messages.create(
            model=summary_model,
            max_tokens=max_tokens,
            temperature=0.1,
            system=system_content,
            messages=[{"role": "user", "content": user_content}]
        )

        return response.content[0].text

    # Split recent messages
    recent_messages_full = all_recent_messages[-full_recent_messages_count:]
    older_recent_messages = all_recent_messages[:-full_recent_messages_count]
    
    # Split relevant messages into blocks
    relevant_blocks = [block.strip() for block in relevant_messages.split('],') if block.strip()]
    
    # Log the number of relevant blocks
    logger.info(f"Number of relevant blocks: {len(relevant_blocks)}")
    
    # Calculate token allocations
    older_recent_tokens = summary_max_tokens // 2
    if relevant_blocks:
        relevant_tokens_per_block = (summary_max_tokens - older_recent_tokens) // len(relevant_blocks)
    else:
        relevant_tokens_per_block = 0

    # Create tasks for parallel summarization
    tasks = [
        summarize_block("".join(older_recent_messages), "older recent", older_recent_tokens),
        *[summarize_block(block + ']', "semantically relevant", relevant_tokens_per_block) for block in relevant_blocks]
    ]

    # Log the summary input
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Summary input for {current_time}\n\n")
        f.write(f"Older recent messages:\n{''.join(older_recent_messages)}\n\n")
        for i, block in enumerate(relevant_blocks):
            f.write(f"Relevant block {i+1}:\n{block}\n\n")

    logger.info(f"Saved summary input to {file_path}")
    logger.info(f"Generating {len(tasks)} summaries: 1 for older recent messages and {len(tasks) - 1} for relevant blocks")

    # Run summaries in parallel
    summaries = await asyncio.gather(*tasks)

    # Log the summaries
    with open(file_path, "a", encoding="utf-8") as f:
        f.write("Summaries:\n\n")
        f.write(f"Older recent messages summary:\n{summaries[0]}\n\n")
        for i, summary in enumerate(summaries[1:]):
            f.write(f"Relevant block {i+1} summary:\n{summary}\n\n")

    # Combine summaries
    older_recent_summary = summaries[0]
    relevant_summaries = summaries[1:]

    # Format summaries
    formatted_relevant_summaries = "\n".join(relevant_summaries)

    combined_summary = f"""<older recent messages summary>
{older_recent_summary}
</older recent messages summary>

<semantically relevant messages summary>
{formatted_relevant_summaries}
</semantically relevant messages summary>"""

    logger.info(f"Generated {len(summaries)} parallel extended context summaries:")
    logger.info(f"  - 1 summary for older recent messages")
    logger.info(f"  - {len(relevant_summaries)} summaries for semantically relevant blocks")
    logger.debug(f"Combined summary:\n{combined_summary}")
    
    return post_process_summary(combined_summary), recent_messages_full

def post_process_summary(summary):
    # Remove any lines that appear to be continuing the conversation
    lines = summary.split('\n')
    processed_lines = [line for line in lines if not line.strip().startswith(('I ', 'You ', 'We ', 'Please ', 'Let me '))]
    return '\n'.join(processed_lines)

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
    recent_messages = []
    recent_message_ids = [message.id]
    
    async for msg in message.channel.history(limit=limit):
        # Skip the first message (the command message)
        if msg.id == message.id:
            continue
        
        timestamp = str(msg.created_at)[:-16]
        message_content = re.sub(r'\S{29,}', lambda m: m.group(0)[:28] + '...', msg.clean_content)
        formatted_message = f"[{timestamp}] {msg.author.name}: {message_content} "

        estimated_tokens = estimate_tokens(formatted_message)
        if token_length - estimated_tokens < 0:
            break

        recent_messages.append(formatted_message)
        recent_message_ids.append(msg.id)
        token_length -= estimated_tokens

    return list(reversed(recent_messages)), recent_message_ids

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

async def get_query_terms(message, chat_mode):
    config, _ = await get_channel_configuration(message)
    
    if chat_mode not in {"Memory", "Extended Memory"}:
        return []

    if chat_mode == "Extended Memory":
        return await get_fast_query_terms(message)
    else:
        return await get_detailed_query_terms(message)

async def get_fast_query_terms(message, num_terms=5):
    # Get recent messages
    recent_messages, _ = await retrieve_recent_messages(message, 2000, 10)
    recent_messages_string = "".join(recent_messages)
    
    # Add the current message
    timestamp = str(message.created_at)[:-16]
    user_message = f"[{timestamp}] {message.author.name}: {message.clean_content}"
    full_text = recent_messages_string + user_message

    # Tokenize and lowercase the text
    tokens = word_tokenize(full_text.lower())

    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]

    # Get the most common words
    word_freq = Counter(filtered_tokens)
    common_words = word_freq.most_common(num_terms)

    # Return just the words, not their frequencies
    return [word for word, _ in common_words]

async def get_detailed_query_terms(message):
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

    response = async_anthropic_client.messages.create(
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
    config, _ = await get_channel_configuration(message)

    # Retrieve recent messages
    all_recent_messages, recent_message_ids = await retrieve_recent_messages(
        message, config["summary_recent_messages_length"]
    )
    
    # Retrieve semantically relevant messages
    relevant_messages = await retrieve_relevant_messages(
        message, query_terms, relevant_messages_length, recent_message_ids
    )

    # Define the number of full recent messages to include
    full_recent_messages_count = 25  # Adjust this value as needed

    # Retrieve sporadic messages if in Day Dream mode
    sporadic_messages = ""
    if chat_mode == "Day Dream":
        sporadic_messages = await retrieve_sporadic_messages(message, sporadic_messages_length)

    summary, recent_messages_full = await summarize_extended_context(
        all_recent_messages,
        relevant_messages,
        config["summary_model"],
        config["summary_max_tokens"],
        full_recent_messages_count
    )

    # Construct the context message
    context_message = f"I am an AI assistant talking to the user: {message.author.name}. The current time is {str(message.created_at)[:-16]}. "
    context_message += f"Extended conversation summary: {summary}\n\n"
    context_message += "Recent messages:\n" + "".join(recent_messages_full)

    if sporadic_messages:
        context_message += f"\n\n<sporadic messages> {sporadic_messages} </sporadic messages>"

    # Construct the message array
    messages = [
        {"role": "user", "content": context_message},
        {"role": "assistant", "content": "Thank you for providing the context. I'll keep that in mind for our conversation."},
        {"role": "user", "content": message.clean_content}
    ]

    return messages, recent_messages_full, relevant_messages, system_message

# Helper function to format messages (if needed)
def format_messages(messages):
    formatted = []
    for msg in messages:
        timestamp = str(msg.created_at)[:-16]
        content = re.sub(r'\S{29,}', lambda m: m.group(0)[:28] + '...', msg.clean_content)
        formatted.append(f"[{timestamp}] {msg.author.name}: {content}")
    return formatted

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
        
        query_terms = await get_query_terms(message, chat_mode)

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
    # Create a unique filename based on the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"response_input_{timestamp}.txt"
    
    # Ensure the logs directory exists
    log_dir = "logs/response_inputs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Full path for the log file
    file_path = os.path.join(log_dir, filename)
    
    # Write the system message and messages to the file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("SYSTEM MESSAGE:\n")
        f.write(f"{system_message}\n\n")
        f.write("MESSAGES:\n")
        json.dump(messages, f, indent=2, ensure_ascii=False)
        f.write("\n")
    
    logger.info(f"Saved response input to {file_path}")

    try:
        response = await async_anthropic_client.messages.create(
            model=model,
            max_tokens=max_response_tokens,
            temperature=temperature,
            system=system_message,
            messages=messages
        )
        return response.content[0].text

    except anthropic.APIError as e:
        logger.error(f"Anthropic API error: {e}")
        return "Sorry, there was an issue with the request. Please try again later."

    except Exception as e:
        logger.error(f"Non-API error occurred: {e}", exc_info=True)
        return "Sorry, an unexpected error occurred. Please try again later."

# Modify the error handling to use the logger
def handle_exception(e):
    logger.error(f"Error occurred: {e}", exc_info=True)

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
    logger.info(f"{client.user} has connected to Discord!")
    logger.info(f"Connected servers: {', '.join([guild.name for guild in client.guilds])}")

# Add the event handlers to the client
client.event(on_ready)
client.event(on_message)

# Run the bot
try:
    cleanup_database()
    if TOKEN is not None:
        client.run(TOKEN)
    else:
        raise ValueError("TOKEN is not set.")
except ValueError as e:
    logger.critical(str(e))

# defines a function that saves the chroma database to disk.
def save_database():
    chromadb_client.persist()
    logger.info("Database saved. Script is ending.")

# saves the database on exit (workaround for https://github.com/chroma-core/chroma/issues/622)
atexit.register(save_database)