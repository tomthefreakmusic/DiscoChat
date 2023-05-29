# DiscoChat 
## An GPT-3.5-Turbo based Discord Bot with Vector Database message handling.

### Introduction
DiscoChat is a Discord bot that leverages OpenAI's API to interact with users, and integrates a vector database, ChromaDB, to maintain and retrieve relevant historical context for enhanced chat completions. It fetches recent and relevant messages from Discord and the ChromaDB respectively, and uses them to generate context-aware responses.

The bot can be interacted with by mentioning it. In Direct Messages (DMs), the bot responds to all messages. I have setup a test server on Discord, feel free to join and try out the latest version of the bot (codenamed Athena) (https://discord.gg/u2Qw4QfG)

### Features

- Uses the GPT-3.5-Turbo model for chat completions, this can easily be reconfigured for GPT 4 if you have API access.
- Uses ChromaDB for vector database message storage and retrival.
- Treats all channels and direct messages as seperate histories. Messages from one channel can not access messages from others, although this can be configured with minimal alterations.
- Can handle long responses that exceed Discord's max message length.
- Extracts and stores metadata (such as server, channel, author, timestamp, keywords, etc.) along with messages for relevance-based message retrieval.

### Setup

Install the required Python packages: 

    openai, discord.py, chromadb, dotenv, rake-nltk, tiktoken, asyncio, atexit, nltk

Setup a discord bot via https://discord.com/developers/applications. The bot must have permissions to: read messages/view channels, send messages and read message history.

Setup a .env file in your project root with your Discord bot token and OpenAI API key:

    DISCORD_TOKEN=your_discord_token
    OPENAI_API_KEY=your_openai_key
    BOT_NAME=your_bot_name
    DATABASE_DIRECTORY=where_you_want_the_database_stored

Then run the script using Python 3.8 or later.

    python discochat.py

### Commands

    !botname populate database: Populates the database with historical messages from the Discord server.

