# DiscoChat

A Discord bot powered by Claude 3.5 Sonnet with advanced memory and conversation capabilities.

## Features

- Multiple chat modes including Extended Memory, Memory, and Day Dream
- Image generation capabilities
- Semantic message search and retrieval
- Auto follow-up conversations
- Server whitelist and permission management

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Create a `.env` file with the following required variables:
```env
DISCORD_TOKEN=your_discord_bot_token
ANTHROPIC_API_KEY=your_anthropic_api_key
BOT_NAME=your_bot_name
DATABASE_DIRECTORY=path_to_database_directory
DEV_NAME=developer_name
SERVER_WHITELIST=comma_separated_server_list
FAL_KEY=your_fal_api_key
```

Optional environment variables:
```env
MISTRAL_API_KEY=your_mistral_api_key  # Required for Extended Memory Mistral mode
BFL_API_KEY=your_bfl_api_key  # Required for image generation
```

4. Download required NLTK data:
```python
import nltk
nltk.download("stopwords")
nltk.download("punkt")
```

## Running the Bot

```bash
python Discochat.py
```

## Available Commands

- `!botname populate database` - Populate the message database
- `!botname clear database` - Clear the message database
- `!botname count database` - Show database message count
- `!botname set chat mode` - Change the chat mode
- `!botname show configuration` - Display current configuration

## Chat Modes

- **Extended Memory** - Uses Claude to summarize conversation history
- **Memory** - Uses semantic search to find relevant past messages
- **Day Dream** - Includes random past messages for creative responses
- **Extended Memory Mistral** - Uses Mistral model for memory processing

## Permissions

The bot can be restricted to specific servers and users with the bot's role.

