import os
import openai
import discord
from dotenv import load_dotenv
import tiktoken
import nltk


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
]

for variable in required_variables:
    if os.getenv(variable) is None:
        print(f"{variable} environment variable not set.")
        exit(1)

TOKEN = os.getenv("DISCORD_TOKEN")

# Set discord intents
intents = discord.Intents.default()
intents.messages = True
intents.guild_messages = True
intents.message_content = True

# Sets the client variable
client = discord.Client(intents=intents)


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
system_message = f"You are a helpful AI system named {bot_name}. You are system that combines a vector database (Chroma) and OpenAI's GPT 3.5 Turbo model, integrated into Discord. You respond when mentioned by a user. Users message you within Discord. Recent messages within the channel are fetched from Discord, whereas relevant messages from the channel are fetched from the vector database. Avoid repeating what the user says, be concise and conversational."

# sets the model.
model = "gpt-3.5-turbo"

# sets the max response tokens.
max_response_tokens = 1000

# sets the minimum messages required to be stored before relevant messages can be retrieved.
min_messages_threshold = 5

# creates a ephemeral dictionary for storing the previous relevant messages, these are currently summarized before storage.
previous_relevant_messages = {}

# sets maximum message length (if you have nitro this could be increased)
max_discord_message_length = 2000

# Sets the token encoder to match the model we are using.
token_encoder = tiktoken.encoding_for_model(model)
