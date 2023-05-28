# DiscoChat 
## A GPT-3.5-Turbo based Discord Bot with Vector Database message handling.

### Introduction
DiscoChat is a Discord bot that leverages OpenAI's GPT-3.5-Turbo model to interact with users, and integrates a vector database, ChromaDB, to maintain and retrieve relevant historical context for enhanced chat completions. It fetches recent and relevant messages from Discord and the ChromaDB respectively, and uses them to generate context-aware responses.

The bot can be interacted with by mentioning it, and it can also handle commands that start with !DiscoChat. In Direct Messages (DMs), the bot responds to all messages.

### Features

- Uses the GPT-3.5-Turbo model for chat completions.
- Uses ChromaDB for vector database message storage and retrival.
- Treats all channels and direct messages as seperate histories. Messages from one channel can not access messages from others, although this can be configured with minimal alterations.
- Responds to messages where the bot is mentioned or commands that start with !DiscoChat.
- Can handle Direct Messages (DMs).
- Can handle long responses that exceed Discord's max message length.
- Extracts and stores metadata (such as server, channel, author, timestamp, keywords, etc.) along with messages for advanced querying and relevance-based message retrieval.

### Setup

Install the required Python packages: 

    openai, discord.py, chromadb, dotenv, rake-nltk, tiktoken, asyncio, atexit

Setup a .env file in your project root with your Discord token and OpenAI API key:

    DISCORD_TOKEN=your_discord_token
    OPENAI_API_KEY=your_openai_key

Then run the script using Python 3.8 or later.

    python discochat.py


### Commands

    populate database: Populates the ChromaDB with historical messages from the Discord server.

### Disclaimer
This bot is a demonstration of how to integrate the GPT-3.5-Turbo model and a vector database into a Discord bot. It should be used responsibly, adhering to Discord's guidelines and OpenAI's use case policy. It might require additional error handling and optimizations for production use.