# Standard library imports
import atexit
import asyncio
import codecs
import io
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import random
import re
import sys
import time
import textwrap
import traceback
from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal

# Third-party imports
import aiohttp
from anthropic import AsyncAnthropic
import chromadb
from chromadb.errors import IDAlreadyExistsError
from chromadb.utils import embedding_functions
import discord
from discord import app_commands
from discord.ui import View, Button
from dotenv import load_dotenv
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from openai import AsyncOpenAI
from rake_nltk import Rake

# Local imports
from chat_mode_presets import CHAT_MODE_PRESETS

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

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
file_handler = RotatingFileHandler(log_filename, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)

# Create console handler with a higher log level
console_handler = logging.StreamHandler(codecs.getwriter('utf-8')(sys.stdout.buffer))
console_handler.setLevel(logging.INFO)

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
    "FAL_KEY",
    "DEEPSEEK_API_KEY",
]

for variable in required_variables:
    if os.getenv(variable) is None:
        print(f"{variable} environment variable not set.")
        exit(1)

TOKEN = os.getenv("DISCORD_TOKEN")

FAL_KEY = os.getenv("FAL_KEY")

BFL_API_KEY = os.getenv("BFL_API_KEY")

# Set Anthropic API key
async_anthropic_client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Set OpenAI client for DeepSeek
async_openai_client = AsyncOpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

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
    database_directory = "./database/"
else:
    database_directory = os.getenv("DATABASE_DIRECTORY")
    # Normalize the path for Windows and convert to absolute path if needed
    database_directory = os.path.abspath(os.path.normpath(database_directory))
    # Ensure the database directory exists
    os.makedirs(database_directory, exist_ok=True)

# sets the model.
model = "claude-3-5-sonnet-latest"



# sets the minimum messages required to be stored before relevant messages can be retrieved.
min_messages_threshold = 5

# creates a ephemeral dictionary for storing the previous relevant messages, these are currently summarized before storage.
previously_relevant_messages = {}

# sets maximum message length (if you have nitro this could be increased)
max_discord_message_length = 2000

class CustomClient(discord.Client):
    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        await self.tree.sync()


intents = discord.Intents.default()
intents.messages = True
intents.guild_messages = True
intents.message_content = True

client = CustomClient(intents=intents)

# Initialize Rake
logger.info("Initializing RAKE...")
r = Rake()
logger.info("RAKE initialized successfully")

nltk.download('punkt_tab')

sample_text = "The quick brown fox jumps over the lazy dog"
r.extract_keywords_from_text(sample_text)
keywords = r.get_ranked_phrases()
logger.info(f"Extracted keywords: {keywords}")

user_configs = {}
dm_channels = {}

DEFAULT_AUTO_FOLLOW_UP = False
last_bot_message = {}
follow_up_tasks = {}

# setup chroma and the collection (message_bank)
try:
    logger.info(f"Initializing ChromaDB with database directory: {database_directory}")
    
    # Initialize ChromaDB with the new format
    chromadb_client = chromadb.PersistentClient(path=database_directory)
    
    # Create the embedding function using SentenceTransformer
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Try to get existing collection or create new one with proper settings
    message_bank = chromadb_client.get_or_create_collection(
        name="message_bank",
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function
    )
    logger.info(f"Successfully initialized ChromaDB collection 'message_bank'")
except Exception as e:
    logger.error(f"Error initializing ChromaDB: {str(e)}")
    raise

# Defines the store_message function, for storing the discord messages in the chroma database.
def store_message(message):
    if message and message.content:
        try:
            message_id, content, metadata = extract_message_data(message)
            logger.info(f"Attempting to store message {message_id} in database")
            logger.debug(f"Message content: {content[:100]}...")  # Log first 100 chars
            logger.debug(f"Message metadata: {metadata}")

            message_bank.add(
                documents=[content],
                metadatas=[metadata],
                ids=[message_id],
            )
            logger.info(f"Successfully stored message {message_id}")
        except IDAlreadyExistsError:
            # If the document with the same id already exists in the database, skip it
            logger.debug(f"Message {message_id} already exists in database, skipping")
            pass
        except Exception as e:
            # Handle other types of exceptions
            logger.error(f"Error adding message {message_id} to database: {str(e)}")
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

async def cleanup_database():
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
            # Allow other tasks to run
            await asyncio.sleep(0)

    if low_quality_ids:
        logger.info(f"Attempting to remove {len(low_quality_ids)} low-quality entries from the database")
        for id_to_delete in low_quality_ids:
            try:
                message_bank.delete(ids=[id_to_delete])
                # Allow other tasks to run
                await asyncio.sleep(0)
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

    if not query_terms:
        logger.info("No specific query terms provided. Skipping relevant message retrieval.")
        return ""

    if recent_message_ids is None:
        logger.warning("No recent message IDs provided. Using an empty list.")
        recent_message_ids = []

    channel = str(message.channel.id)
    distance_threshold = 0.85
    bot_penalty = 0.4
    
    where_conditions = {"$and": [{"channel": channel}, {"is_command": "False"}]}

    try:
        relevant_messages = message_bank.query(
            query_texts=query_terms,
            n_results=20,  # Increased from 10 to 20 to get more results
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
                temp_string = f"[{str(msg.created_at)[:-16]}] {msg.author.name}: {message_content}"
                
                estimated_tokens = estimate_tokens(temp_string)
                if token_length - estimated_tokens < 0:
                    logger.debug(f"Token limit reached, breaking loop")
                    break

                block_messages.append(temp_string)
                token_length -= estimated_tokens
                logger.debug(f"Added message {msg.id} to block, remaining tokens: {token_length}")

            if block_messages:
                relevant_messages_result.append("\n".join(block_messages))
                quality_messages_count += 1

            if quality_messages_count >= 10:  # Changed from 5 to 10
                logger.debug(f"Reached 10 quality message blocks, stopping retrieval")
                break

    except Exception as e:
        logger.error(f"Error processing query results: {e}", exc_info=True)
        return []

    logger.info(f"retrieve_relevant_messages completed, returned {len(relevant_messages_result)} blocks")
    return relevant_messages_result

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

async def summarize_extended_context(all_recent_messages, relevant_message_blocks, summary_model, summary_max_tokens, full_recent_messages_count):
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
        
        Summary format example:
        
        last tuesday alice and bob discussed cats.
        alice said she liked indoor cats, in particular short haired ones. 
        bob responded to alice by saying he likes cats as well and that he has a dark grey cat.
        alice revealed that her cat is a three year old tabby.
        
        """

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

        logger.info(f"Token usage for summarize_block ({block_type}): Input tokens: {response.usage.input_tokens}, Output tokens: {response.usage.output_tokens}")

        return response.content[0].text

    # Split recent messages
    recent_messages_full = all_recent_messages[-full_recent_messages_count:]
    older_recent_messages = all_recent_messages[:-full_recent_messages_count]
    
    # Log the number of relevant blocks
    logger.info(f"Number of relevant blocks: {len(relevant_message_blocks)}")
    
    # Calculate token allocations
    older_recent_tokens = summary_max_tokens // 2
    if relevant_message_blocks:
        relevant_tokens_per_block = (summary_max_tokens - older_recent_tokens) // len(relevant_message_blocks)
    else:
        relevant_tokens_per_block = 0

    # Create tasks for parallel summarization
    tasks = [
        summarize_block("".join(older_recent_messages), "older recent", older_recent_tokens),
        *[summarize_block(block, "semantically relevant", relevant_tokens_per_block) for block in relevant_message_blocks]
    ]

    # Log the summary input
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Summary input for {current_time}\n\n")
        f.write(f"Older recent messages:\n{''.join(older_recent_messages)}\n\n")
        for i, block in enumerate(relevant_message_blocks):
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
    logger.info(f"Counting messages for channel {channel_id}")
    try:
        channel_messages = message_bank.get(where={"channel": channel_id})
        num_messages = len(channel_messages["ids"])
        logger.info(f"Found {num_messages} messages in channel {channel_id}")
        return num_messages
    except Exception as e:
        logger.error(f"Error counting messages in channel {channel_id}: {str(e)}")
        return 0

def check_permissions(obj):
    try:
        if isinstance(obj, discord.Interaction):
            user = obj.user
            guild = obj.guild
            channel = obj.channel
        elif isinstance(obj, discord.Message):
            user = obj.author
            guild = obj.guild
            channel = obj.channel
        else:
            logger.error(f"Unsupported object type in check_permissions: {type(obj)}")
            return False

        user_name = user.name if user else "Unknown User"
        guild_name = guild.name if guild else "DM"
        channel_type = type(channel).__name__ if channel else "Unknown Channel"

        logger.debug(f"Checking permissions for user: {user_name}, guild: {guild_name}, channel_type: {channel_type}")

        # Check if it's a DM
        if isinstance(channel, discord.DMChannel) or guild is None:
            logger.info(f"Allowing command in DM for user: {user_name}")
            return True

        # For servers, check whitelist and roles
        if guild_name in server_whitelist:
            logger.info(f"Allowing command for user: {user_name} in whitelisted guild: {guild_name}")
            return True

        if user and user.roles:
            for role in user.roles:
                if role.name == bot_name:
                    logger.info(f"Allowing command for user: {user_name} with role: {bot_name}")
                    return True

        # Check if the user is the bot itself
        if user == client.user:
            logger.info(f"Allowing command for bot user")
            return True

        logger.info(f"Denied permission for user: {user_name} in guild: {guild_name}")
        return False

    except Exception as e:
        logger.error(f"Error in check_permissions: {str(e)}", exc_info=True)
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


async def update_auto_follow_up(channel, enabled):
    config_path = f"./config/{channel.id}.json"
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            channel_config = json.load(f)
    else:
        channel_config = {"chat_mode": "Default"}

    channel_config["auto_follow_up"] = enabled

    with open(config_path, "w") as f:
        json.dump(channel_config, f)

    return enabled

@client.tree.command()
@app_commands.describe(enabled="Enable or disable auto follow-up")
async def toggle_auto_follow_up(interaction: discord.Interaction, enabled: bool):
    """Toggle the auto follow-up feature for this channel"""
    if not check_permissions(interaction):
        await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
        return

    new_state = await update_auto_follow_up(interaction.channel, enabled)
    await interaction.response.send_message(f"Auto follow-up has been {'enabled' if new_state else 'disabled'} for this channel.")

# Add this new function to handle the auto-follow-up
async def schedule_auto_follow_up(channel_id, delay):
    await asyncio.sleep(delay)
    
    channel = client.get_channel(channel_id)
    if channel is None:
        channel = dm_channels.get(channel_id)
    
    if channel is None:
        logger.warning(f"Channel {channel_id} not found for auto-follow-up")
        return

    last_message = last_bot_message.get(channel_id)
    if not last_message:
        logger.warning(f"No last message found for channel {channel_id}")
        return

    # Check the last 10 messages in the channel
    bot_messages = []
    user_responded = False
    async for message in channel.history(limit=10):
        if message.author != client.user:
            user_responded = True
            break
        if message.author == client.user:
            bot_messages.append(message)
        if message.id == last_message['message'].id:
            break

    if user_responded:
        logger.info(f"Auto-follow-up cancelled for channel {channel_id}: A user has responded")
        return

    if not bot_messages:
        logger.info(f"Auto-follow-up cancelled for channel {channel_id}: No recent bot messages found")
        return

    if bot_messages[0].id != last_message['message'].id:
        logger.info(f"Auto-follow-up cancelled for channel {channel_id}: Bot has already sent a follow-up")
        return

    logger.info(f"Generating auto-follow-up message for channel {channel_id}")

    # Generate and send the follow-up message
    original_message = last_message['message']
    original_response = last_message['response']
    time_passed = datetime.now() - last_message['timestamp']

    config, chat_mode, auto_follow_up = await get_channel_configuration(original_message)
    
    # Reuse the context from the previous response
    completion_messages, _, _, formatted_system_message = await generate_completion_messages(
        original_message,
        config["system_message"],
        await get_query_terms(original_message, chat_mode),
        config["recent_messages_length"],
        config["relevant_messages_length"],
        config["sporadic_messages_length"],
        chat_mode,
    )

    # Reformat messages to avoid the "system" role issue
    formatted_messages = [msg for msg in completion_messages if msg['role'] != 'system']
    formatted_messages.extend([
        {"role": "assistant", "content": original_response},
        {"role": "user", "content": f"It has been approximately {time_passed.total_seconds() // 60} minutes since the user last responded. Please provide a single follow-up message to continue the conversation."}
    ])

    follow_up_response = await get_response(
        formatted_messages,
        formatted_system_message,
        config["max_response_tokens"],
        config["temperature"],
        chat_mode
    )

    await send_long_discord_message(channel, follow_up_response)

    logger.info(f"Sent auto-follow-up message in channel {channel_id}")


async def get_channel_configuration(message):
    channel_id = message.channel.id
    try:
        if os.path.isfile(f"./config/{channel_id}.json"):
            with open(f"./config/{channel_id}.json", "r") as f:
                channel_config = json.load(f)
            chat_mode = channel_config.get("chat_mode", "Default")
            auto_follow_up = channel_config.get("auto_follow_up", DEFAULT_AUTO_FOLLOW_UP)
        else:
            chat_mode = "Default"
            auto_follow_up = DEFAULT_AUTO_FOLLOW_UP

        config = CHAT_MODE_PRESETS[chat_mode].copy()  # Make a copy of the preset
        # Add default value for summary_recent_messages_length if not present
        if "summary_recent_messages_length" not in config:
            config["summary_recent_messages_length"] = 4000  # Default value

        return config, chat_mode, auto_follow_up

    except (IOError, ValueError, KeyError) as e:
        logger.error(f"Error handling channel configuration for channel {channel_id}: {e}")
        os.makedirs("./config/", exist_ok=True)
        with open(f"./config/{channel_id}.json", "w") as f:
            json.dump({"chat_mode": "Default", "auto_follow_up": DEFAULT_AUTO_FOLLOW_UP}, f)
        
        # Return default configuration with the missing key
        default_config = CHAT_MODE_PRESETS["Default"].copy()
        default_config["summary_recent_messages_length"] = 4000
        return default_config, "Default", DEFAULT_AUTO_FOLLOW_UP

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

def sanitize_filename(prompt):
    # Remove invalid characters (keep only alphanumerics, spaces, underscores, and hyphens)
    sanitized = re.sub(r'[^A-Za-z0-9 _-]', '', prompt)
    # Replace spaces with underscores
    sanitized = sanitized.replace(' ', '_')
    # Truncate the filename to a maximum length (e.g., 50 characters)
    return sanitized[:40]

@client.tree.command()
@app_commands.describe(
    prompt="The prompt for image generation",
    variant="Model variant to use",
    width="Image width (multiple of 32, between 256 and 1440)",
    height="Image height (multiple of 32, between 256 and 1440)",
    steps="Number of inference steps (between 1 and 50)",
    prompt_upscaling="Enable prompt upscaling",
    guidance="Guidance scale (between 1.5 and 5)"
)
@app_commands.choices(variant=[
    app_commands.Choice(name="flux-pro-1.1", value="flux-pro-1.1"),
    app_commands.Choice(name="flux-pro", value="flux-pro"),
    app_commands.Choice(name="flux-dev", value="flux-dev")
])
async def generate_image(
    interaction: discord.Interaction, 
    prompt: str,
    variant: str = "flux-pro-1.1",
    width: int = 1024,
    height: int = 1024,
    steps: int = 30,
    prompt_upscaling: bool = False,
    guidance: float = 2.5
):
    """Generate an image based on the provided prompt and parameters"""
    if not check_permissions(interaction):
        logger.warning(f"Permission denied for user {interaction.user.name} in {interaction.guild.name if interaction.guild else 'DM'}")
        await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
        return

    await interaction.response.defer()

    try:
        # Start the image generation process
        image_url = await process_image_generation(prompt, variant, width, height, steps, prompt_upscaling, guidance)
        
        if image_url:
            # Download the image
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as resp:
                    if resp.status != 200:
                        await interaction.followup.send("An error occurred while downloading the generated image.")
                        return
                    image_data = await resp.read()

            # Sanitize the prompt to create a safe filename
            sanitized_prompt = sanitize_filename(prompt)

            # Get a short string of numbers (timestamp or random number)
            timestamp = int(time.time() * 1000) % 100000  # 5-digit timestamp

            # Alternatively, use a random number
            # random_number = random.randint(10000, 99999)  # 5-digit random number

            # Append the number to the filename
            filename = f"{sanitized_prompt}_{timestamp}.png"
            # If using random number:
            # filename = f"{sanitized_prompt}_{random_number}.png"

            # Create a file object from the image data with the new filename
            file = discord.File(io.BytesIO(image_data), filename=filename)

            # Send the generated image
            await interaction.followup.send(f"Generated image for prompt: '{prompt}'", file=file)
        else:
            await interaction.followup.send("Failed to generate the image. Please try again.")

    except Exception as e:
        logger.error(f"Error generating image: {str(e)}", exc_info=True)
        await interaction.followup.send("An error occurred while generating the image. Please try again.")

async def process_image_generation(prompt, variant, width, height, steps, prompt_upscaling, guidance):
    try:
        # Construct the endpoint URL based on the variant
        endpoint_url = f'https://api.bfl.ml/v1/{variant}'

        # Create the initial request without the 'variant' parameter
        async with aiohttp.ClientSession() as session:
            async with session.post(
                endpoint_url,
                headers={
                    'accept': 'application/json',
                    'x-key': BFL_API_KEY,
                    'Content-Type': 'application/json',
                },
                json={
                    'prompt': prompt,
                    'width': width,
                    'height': height,
                    'steps': steps,
                    'prompt_upscaling': prompt_upscaling,
                    'guidance': guidance
                }
            ) as response:
                request = await response.json()

        if 'id' not in request:
            logger.error(f"Error in image generation request: {request}")
            return None

        request_id = request['id']

        # Poll for the result
        while True:
            await asyncio.sleep(0.5)
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    'https://api.bfl.ml/v1/get_result',
                    headers={
                        'accept': 'application/json',
                        'x-key': BFL_API_KEY,
                    },
                    params={
                        'id': request_id,
                    }
                ) as response:
                    result = await response.json()

            if result["status"] == "Ready":
                return result['result']['sample']
            elif result["status"] in ["Error", "Request Moderated", "Content Moderated"]:
                logger.error(f"Error in image generation: {result}")
                return None

    except Exception as e:
        logger.error(f"Error in process_image_generation: {str(e)}", exc_info=True)
        return None

async def get_query_terms(message, chat_mode):
    config, _, auto_follow_up = await get_channel_configuration(message)
    
    if chat_mode in {"Extended Memory", "Extended Memory Mistral"}:
        return await get_extended_memory_query_terms(message)
    elif chat_mode == "Memory":
        return await get_detailed_query_terms(message)
    elif chat_mode == "Day Dream":
        return await get_fast_query_terms(message)
    else:
        logger.warning(f"Unknown chat mode: {chat_mode}. Using fast query terms.")
        return await get_fast_query_terms(message)

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

async def get_extended_memory_query_terms(message):
    # Get recent messages up to 4000 tokens
    recent_messages, _ = await retrieve_recent_messages(message, 4000)
    recent_messages_string = "".join(recent_messages)
    
    # Add the current message
    timestamp = str(message.created_at)[:-16]
    user_message = f"[{timestamp}] {message.author.name}: {message.clean_content}"
    full_text = recent_messages_string + user_message

    system_content = """You are an AI assistant that extracts key terms for querying a vector database. 
    Your task is to analyze the recent conversation and output exactly 5 key terms or phrases, each on a new line. 
    Only include phrases that are directly relevant to the most recent topics of the conversation.
    These phrases should be concise, diverse, and directly usable for database queries.
    Prioritize noun phrases and specific concepts over general words.
    Do not include any explanations or additional text."""

    user_content = f"{full_text}\n\n Based on the recent conversation, provide 5 key terms or phrases from throughout the text for querying a vector database."

    try:
        response = await async_anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            temperature=0.4,
            system=system_content,
            messages=[
                {"role": "user", "content": user_content}
            ]
        )

        

        # Extract terms from the response
        terms = [term.strip() for term in response.content[0].text.strip().split('\n') if term.strip()]
        
        # Ensure we have exactly 5 terms
        terms = terms[:5]
        while len(terms) < 5:
            terms.append("")  # Add empty strings if we have fewer than 5 terms

        logger.debug(f"Extended memory query terms: {terms}")
        return terms
    except Exception as e:
        logger.error(f"Error in get_extended_memory_query_terms: {e}", exc_info=True)
        return []  # Return an empty list if there's an error

async def generate_completion_messages(
    message,
    system_message,
    query_terms,
    recent_messages_length,
    relevant_messages_length,
    sporadic_messages_length,
    chat_mode,
):
    config, _, auto_follow_up = await get_channel_configuration(message)

    # Retrieve recent messages
    all_recent_messages, recent_message_ids = await retrieve_recent_messages(
        message, config["summary_recent_messages_length"]
    )
    
    # Retrieve semantically relevant messages
    relevant_message_blocks = await retrieve_relevant_messages(
        message, query_terms, relevant_messages_length, recent_message_ids
    )

    # Define the number of full recent messages to include
    full_recent_messages_count = 50  # Adjust this value as needed

    # Retrieve sporadic messages if in Day Dream mode
    sporadic_messages = ""
    if chat_mode == "Day Dream":
        sporadic_messages = await retrieve_sporadic_messages(message, sporadic_messages_length)

    # Generate summary for Extended Memory mode
    if chat_mode == "Extended Memory":
        summary, recent_messages_full = await summarize_extended_context(
            all_recent_messages,
            relevant_message_blocks,
            config["summary_model"],
            config["summary_max_tokens"],
            full_recent_messages_count
        )
    else:
        summary = ""
        recent_messages_full = all_recent_messages[-full_recent_messages_count:]

    # Construct the context message
    context_message = f"I am an AI assistant named {bot_name} talking to the user: {message.author.name}. The current time is {str(message.created_at)[:-16]}. "
    
    if chat_mode == "Extended Memory":
        context_message += f"Extended conversation summary: {summary}\n\n"
    
    context_message += "Recent messages:\n" + "".join(recent_messages_full)

    if chat_mode == "Memory":
        # Join the relevant message blocks into a single string for the Memory mode
        relevant_messages_str = "\n\n".join(relevant_message_blocks)
        context_message += f"\n\nRelevant past messages:\n{relevant_messages_str}"
    elif chat_mode == "Day Dream":
        context_message += f"\n\n<sporadic messages>\n{sporadic_messages}\n</sporadic messages>"

    # Format the system message with the bot's name
    formatted_system_message = system_message.format(bot_name=bot_name)

    # Construct the message array
    messages = [
        {"role": "user", "content": context_message},
        {"role": "assistant", "content": "Thank you for providing the context. I understand that this is an incomplete picture of our whole conversation, but I will do my best to respond to your message with regards to the information provided. When I respond, I will follow my primary objectives."},
        {"role": "user", "content": message.clean_content}
    ]

    return messages, recent_messages_full, relevant_message_blocks, formatted_system_message

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
        config, chat_mode, auto_follow_up = await get_channel_configuration(message)
        
        query_terms = await get_query_terms(message, chat_mode)

        completion_messages, recent_messages_full, relevant_messages, formatted_system_message = await generate_completion_messages(
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
            formatted_system_message,
            config["max_response_tokens"],
            config["temperature"],
            chat_mode
        )
        await send_long_discord_message(message.channel, response)
        
        # Set up auto-follow-up timer if enabled
        if auto_follow_up:
            channel_id = message.channel.id
            last_bot_message[channel_id] = {
                'message': message,
                'response': response,
                'timestamp': datetime.now()
            }
            
            # Cancel any existing follow-up task for this channel
            if channel_id in follow_up_tasks:
                follow_up_tasks[channel_id].cancel()
            
            # Schedule a new follow-up task with the original delay
            delay = random.randint(60, 5 * 60)  # Random delay between 5 minutes and 6 hours
            follow_up_tasks[channel_id] = asyncio.create_task(schedule_auto_follow_up(channel_id, delay))
            
            logger.info(f"Scheduled auto-follow-up for channel {channel_id} in {delay} seconds")
        
        return response

async def generate_mistral_response(system_message: str, messages: List[Dict[str, Any]], max_tokens: int, temperature: float) -> str:
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('MISTRAL_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        # Prepare the messages including the system message
        api_messages = [{"role": "system", "content": system_message}] + messages
        
        payload = {
            "model": "mistral-large-latest",
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 1,  # Default value as per documentation
            "stream": False,  # We don't want to stream the response
            "safe_prompt": False,  # Default value
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers,
                json=payload
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data['choices'][0]['message']['content']
                else:
                    error_data = await resp.text()
                    raise Exception(f"Mistral API error: Status {resp.status}, Response: {error_data}")

    except aiohttp.ClientError as e:
        logger.error(f"Network error in Mistral API call: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error in Mistral API call: {e}", exc_info=True)
        raise

async def generate_deepseek_response(system_message: str, messages: List[Dict[str, Any]], max_tokens: int, temperature: float) -> str:
    try:
        # Format messages for DeepSeek API
        formatted_messages = [{"role": "system", "content": system_message}]
        for msg in messages:
            if msg["role"] != "system":  # Skip any system messages in the input
                formatted_messages.append(msg)

        response = await async_openai_client.chat.completions.create(
            model="deepseek-chat",
            messages=formatted_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False
        )
        
        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Error in DeepSeek API call: {e}", exc_info=True)
        raise

async def get_response(
    messages: List[Dict[str, Any]],
    system_message: str,
    max_response_tokens: int,
    temperature: float,
    chat_mode: str
) -> str:
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
        # Remove any system messages from the input
        filtered_messages = [msg for msg in messages if msg['role'] != 'system']

        if chat_mode == "Extended Memory Mistral":
            try:
                response = await generate_mistral_response(
                    system_message,
                    filtered_messages,
                    max_response_tokens,
                    temperature
                )
            except Exception as e:
                logger.error(f"Error using Mistral API, falling back to Claude: {e}")
                # Fallback to Claude if Mistral fails
                response = await async_anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    max_tokens=max_response_tokens,
                    temperature=temperature,
                    system=system_message,
                    messages=filtered_messages
                )
                logger.info(f"Token usage for get_response (Claude fallback): Input tokens: {response.usage.input_tokens}, Output tokens: {response.usage.output_tokens}")
                response = response.content[0].text
        elif chat_mode == "DeepSeek":
            try:
                response = await generate_deepseek_response(
                    system_message,
                    filtered_messages,
                    max_response_tokens,
                    temperature
                )
            except Exception as e:
                logger.error(f"Error using DeepSeek API, falling back to Claude: {e}")
                # Fallback to Claude if DeepSeek fails
                response = await async_anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    max_tokens=max_response_tokens,
                    temperature=temperature,
                    system=system_message,
                    messages=filtered_messages
                )
                logger.info(f"Token usage for get_response (Claude fallback): Input tokens: {response.usage.input_tokens}, Output tokens: {response.usage.output_tokens}")
                response = response.content[0].text
        else:
            # Use Claude API for other chat modes
            response = await async_anthropic_client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=max_response_tokens,
                temperature=temperature,
                system=system_message,
                messages=filtered_messages
            )
            logger.info(f"Token usage for get_response (Claude): Input tokens: {response.usage.input_tokens}, Output tokens: {response.usage.output_tokens}")
            response = response.content[0].text

        return response

    except Exception as e:
        error_message = f"An error occurred while generating a response: {str(e)}"
        logger.error(error_message)
        return error_message

# Modify the error handling to use the logger
def handle_exception(e):
    logger.error(f"Error occurred: {e}", exc_info=True)

# defines a helper function that handles messages bigger than discord handles by default (nitro makes this redundant)
async def send_long_discord_message(channel, response):
    try:
        # Replace escaped newlines with actual newlines
        response = response.replace('\\n', '\n')

        if len(response) <= max_discord_message_length:
            await channel.send(response)
        else:
            parts = textwrap.wrap(
                response,
                max_discord_message_length,
                break_long_words=False,
                replace_whitespace=False,
            )

            for part in parts:
                await channel.send(part)
                await asyncio.sleep(1)
        
        logger.info(f"Sent message in channel {channel.id}")
    except discord.errors.Forbidden:
        logger.error(f"Forbidden to send message in channel {channel.id}")
    except Exception as e:
        logger.error(f"Error sending message in channel {channel.id}: {str(e)}")

# defines a function that prints a message to the console when the discord bot is ready
@client.event
async def on_ready():
    logger.info(f"{client.user} has connected to Discord!")
    logger.info(f"Connected servers: {', '.join([guild.name for guild in client.guilds])}")
    
    # Log the synced commands
    synced_commands = await client.tree.sync()
    logger.info(f"Synced {len(synced_commands)} command(s): {', '.join([cmd.name for cmd in synced_commands])}")

# Add the event handlers to the client
@client.event
async def on_message(message):

    if isinstance(message.channel, discord.DMChannel):
        dm_channels[message.channel.id] = message.channel

    if message.author != client.user:
        # Cancel any existing follow-up task for this channel
        channel_id = message.channel.id
        if channel_id in follow_up_tasks:
            follow_up_tasks[channel_id].cancel()
        
        # Process the message as before
        if check_permissions(message):
            store_message(message)
            try:
                if await is_command(message):
                    await handle_command(message)
                elif await should_respond(message):
                    await respond_to_message(message)
            except Exception as e:
                handle_exception(e)
    else:
        # If it's a bot message, don't process it further
        return

# Run the bot
async def main():
    try:
        await cleanup_database()
        if TOKEN is not None:
            await client.start(TOKEN)
        else:
            raise ValueError("TOKEN is not set.")
    except ValueError as e:
        logger.critical(str(e))
    finally:
        # Clean up any running tasks
        for task in follow_up_tasks.values():
            task.cancel()
        await client.close()

# defines a function that saves the chroma database to disk.
def save_database():
    chromadb_client.persist()
    logger.info("Database saved. Script is ending.")

# saves the database on exit (workaround for https://github.com/chroma-core/chroma/issues/622)
atexit.register(save_database)

if __name__ == "__main__":
    asyncio.run(main())