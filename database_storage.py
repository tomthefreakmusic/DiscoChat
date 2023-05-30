import chromadb
from chromadb.config import Settings
from chromadb.errors import IDAlreadyExistsError
from config import bot_name, database_directory
from rake_nltk import Rake
import traceback
import discord
from bot_logic import client


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
        "channel": str(message.channel.id),
        "server": server,
        "author": author,
        "created_at": created_at,
        "keywords": keywords_str,
        "bot_mentioned": bot_mentioned,
        "is_command": is_command,
    }

    return message_id, content, metadata


# Defines a helper function that checks if the message is a DM.
def is_dm(message):
    is_dm = isinstance(message.channel, discord.DMChannel)
    return is_dm


# defines a helper function for counting the messages in a channel.
def channel_database_count(message):
    # Query the database for all documents where the channel matches the current channel.
    # As we are not interested in the documents themselves, we only retrieve the metadata.
    channel_id = str(message.channel.id)
    channel_messages = message_bank.get(where={"channel": channel_id})

    # Count the number of messages
    num_messages = len(channel_messages["ids"])

    return num_messages


# defines a function that saves the chroma database to disk.
def save_database():
    chromadb_client.persist()
    pass
