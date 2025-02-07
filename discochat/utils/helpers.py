import re
import asyncio
import discord
from typing import List, Tuple
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from datetime import datetime, timedelta
from .logger import logger
from ..config.settings import settings

def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text."""
    return len(text) // 4

async def send_long_discord_message(channel, response: str):
    """Send a long message to Discord, splitting it if necessary."""
    try:
        # Replace escaped newlines with actual newlines
        response = response.replace('\\n', '\n')

        if len(response) <= settings["max_discord_message_length"]:
            await channel.send(response)
        else:
            parts = textwrap.wrap(
                response,
                settings["max_discord_message_length"],
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

def is_dm(message) -> bool:
    """Check if a message is a DM."""
    return isinstance(message.channel, discord.DMChannel)

def check_permissions(obj) -> bool:
    """Check if the user has permission to use the bot."""
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
        if guild_name in settings["server_whitelist"]:
            logger.info(f"Allowing command for user: {user_name} in whitelisted guild: {guild_name}")
            return True

        if user and user.roles:
            for role in user.roles:
                if role.name == settings["bot_name"]:
                    logger.info(f"Allowing command for user: {user_name} with role: {settings['bot_name']}")
                    return True

        logger.info(f"Denied permission for user: {user_name} in guild: {guild_name}")
        return False

    except Exception as e:
        logger.error(f"Error in check_permissions: {str(e)}", exc_info=True)
        return False

def get_relative_time(timestamp_str: str, current_time: datetime) -> str:
    """Convert a timestamp to a relative time string."""
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

def process_messages(messages: str, message_type: str, current_time: datetime) -> List:
    """Process and format messages for display."""
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

def format_messages(messages: List[discord.Message]) -> List[str]:
    """Format a list of Discord messages."""
    formatted = []
    for msg in messages:
        timestamp = str(msg.created_at)[:-16]
        content = re.sub(r'\S{29,}', lambda m: m.group(0)[:28] + '...', msg.clean_content)
        formatted.append(f"[{timestamp}] {msg.author.name}: {content}")
    return formatted

def extract_message_data(message: discord.Message) -> Tuple[str, str, dict]:
    """Extract relevant data from a Discord message."""
    message_id = str(message.id)
    author = str(message.author)
    created_at = str(message.created_at)
    content = message.clean_content

    bot_mentioned = "True" if message.guild.me in message.mentions else "False"
    server = "DM" if is_dm(message) else str(message.guild)
    is_command = "True" if message.content.lower().startswith(f"!{settings['bot_name'].lower()}") else "False"

    # Extract keywords from the message content
    keywords = get_keywords(message.clean_content)
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

def get_keywords(text: str, num_keywords: int = 5) -> List[str]:
    """Extract keywords from text using NLTK."""
    # Tokenize and lowercase the text
    tokens = word_tokenize(text.lower())

    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]

    # Get the most common words
    word_freq = Counter(filtered_tokens)
    common_words = word_freq.most_common(num_keywords)

    # Return just the words, not their frequencies
    return [word for word, _ in common_words]
