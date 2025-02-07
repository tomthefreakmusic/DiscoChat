import asyncio
import json
import os
from datetime import datetime
import discord
from discord.ui import View, Button
from ..utils.logger import logger
from ..utils.helpers import (
    is_dm, check_permissions, send_long_discord_message,
    extract_message_data, format_messages
)
from ..database.chroma_manager import db_manager
from ..api_clients.llm_client import llm_client
from ..config.settings import settings

class MessageProcessor:
    def __init__(self):
        self.last_bot_message = {}
        self.follow_up_tasks = {}
        self.dm_channels = {}

    async def process_message(self, message: discord.Message):
        """Process an incoming Discord message."""
        if isinstance(message.channel, discord.DMChannel):
            self.dm_channels[message.channel.id] = message.channel

        if message.author != message.guild.me:
            # Cancel any existing follow-up task for this channel
            channel_id = message.channel.id
            if channel_id in self.follow_up_tasks:
                self.follow_up_tasks[channel_id].cancel()
            
            # Process the message
            if check_permissions(message):
                db_manager.store_message(message)
                try:
                    if await self.is_command(message):
                        await self.handle_command(message)
                    elif await self.should_respond(message):
                        await self.respond_to_message(message)
                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)

    async def is_command(self, message: discord.Message) -> bool:
        """Check if a message is a command."""
        return message.content.lower().startswith(f"!{settings['bot_name'].lower()}")

    async def should_respond(self, message: discord.Message) -> bool:
        """Check if the bot should respond to a message."""
        return (message.guild.me in message.mentions and message.author != message.guild.me) or (
            is_dm(message) and message.author != message.guild.me
        )

    async def handle_command(self, message: discord.Message):
        """Handle bot commands."""
        if message.author == message.guild.me:
            return

        command = message.content[len(settings['bot_name']) + 2:]
        if command.startswith("populate database"):
            await self.populate_database(message)
        elif command.startswith("count database"):
            await message.channel.send(
                f"There are {db_manager.count_messages({'channel': str(message.channel.id)})} messages stored from this channel."
            )
        elif command.startswith("clear database"):
            await self.clear_database(message)
        elif command.startswith("set chat mode"):
            await self.create_chat_mode_form(message)
        elif command.startswith("show configuration"):
            config, chat_mode = await self.get_channel_configuration(message)
            formatted_config = self.format_channel_config(config, chat_mode)
            await message.channel.send(formatted_config)
        else:
            await message.channel.send("Command not found")

    async def populate_database(self, message: discord.Message):
        """Populate the database with channel messages."""
        async for msg in message.channel.history(limit=500):
            db_manager.store_message(msg)
        await message.channel.send(
            f"Database populated. There are {db_manager.count_messages({'channel': str(message.channel.id)})} messages stored from this channel."
        )

    async def clear_database(self, message: discord.Message):
        """Clear messages from the database for this channel."""
        db_manager.delete_messages(where_conditions={"channel": str(message.channel.id)})
        await message.channel.send(
            f"Database cleared. There are {db_manager.count_messages({'channel': str(message.channel.id)})} messages stored from this channel."
        )

    async def get_channel_configuration(self, message: discord.Message):
        """Get the configuration for a channel."""
        channel_id = message.channel.id
        try:
            if os.path.isfile(f"./config/{channel_id}.json"):
                with open(f"./config/{channel_id}.json", "r") as f:
                    channel_config = json.load(f)
                chat_mode = channel_config.get("chat_mode", "Default")
                auto_follow_up = channel_config.get("auto_follow_up", settings["DEFAULT_AUTO_FOLLOW_UP"])
            else:
                chat_mode = "Default"
                auto_follow_up = settings["DEFAULT_AUTO_FOLLOW_UP"]

            return chat_mode, auto_follow_up

        except Exception as e:
            logger.error(f"Error handling channel configuration for channel {channel_id}: {e}")
            os.makedirs("./config/", exist_ok=True)
            with open(f"./config/{channel_id}.json", "w") as f:
                json.dump({"chat_mode": "Default", "auto_follow_up": settings["DEFAULT_AUTO_FOLLOW_UP"]}, f)
            
            return "Default", settings["DEFAULT_AUTO_FOLLOW_UP"]

    async def update_channel_configuration(self, message: discord.Message, new_chat_mode: str):
        """Update the configuration for a channel."""
        with open(f"./config/{message.channel.id}.json", "w") as f:
            json.dump({"chat_mode": new_chat_mode}, f)

        await message.channel.send(f"Updated chat mode to {new_chat_mode}.")

    async def create_chat_mode_form(self, message: discord.Message):
        """Create a form for selecting chat mode."""
        view = View()
        for mode in ["Default", "Memory", "Extended Memory", "Day Dream"]:  # Add your chat modes here
            button = Button(label=mode, style=discord.ButtonStyle.primary)
            button.callback = lambda interaction, m=mode: self.handle_button_click(interaction, m)
            view.add_item(button)

        await message.channel.send("Choose a chat mode:", view=view)

    async def handle_button_click(self, interaction: discord.Interaction, mode: str):
        """Handle chat mode button clicks."""
        await self.update_channel_configuration(interaction.message, mode)
        await interaction.response.send_message(f"Chat mode updated to {mode}", ephemeral=True)

    async def respond_to_message(self, message: discord.Message):
        """Generate and send a response to a message."""
        async with message.channel.typing():
            chat_mode, auto_follow_up = await self.get_channel_configuration(message)
            
            # Generate response using the LLM client
            response = await llm_client.get_response(
                messages=[{"role": "user", "content": message.clean_content}],
                system_message=f"You are {settings['bot_name']}, a helpful AI assistant.",
                max_tokens=1000,
                temperature=0.7,
                chat_mode=chat_mode
            )

            # Send the response
            await send_long_discord_message(message.channel, response)
            
            # Set up auto-follow-up if enabled
            if auto_follow_up:
                channel_id = message.channel.id
                self.last_bot_message[channel_id] = {
                    'message': message,
                    'response': response,
                    'timestamp': datetime.now()
                }
                
                # Schedule a new follow-up task
                delay = 300  # 5 minutes
                self.follow_up_tasks[channel_id] = asyncio.create_task(
                    self.schedule_auto_follow_up(channel_id, delay)
                )
                
                logger.info(f"Scheduled auto-follow-up for channel {channel_id} in {delay} seconds")

    async def schedule_auto_follow_up(self, channel_id: int, delay: int):
        """Schedule an auto-follow-up message."""
        await asyncio.sleep(delay)
        
        channel = self.dm_channels.get(channel_id)
        if not channel:
            logger.warning(f"Channel {channel_id} not found for auto-follow-up")
            return

        last_message = self.last_bot_message.get(channel_id)
        if not last_message:
            logger.warning(f"No last message found for channel {channel_id}")
            return

        # Check if user has responded
        async for message in channel.history(limit=10):
            if message.author != message.guild.me:
                logger.info(f"Auto-follow-up cancelled for channel {channel_id}: A user has responded")
                return

        # Generate and send follow-up message
        chat_mode, _ = await self.get_channel_configuration(last_message['message'])
        follow_up_response = await llm_client.get_response(
            messages=[
                {"role": "assistant", "content": last_message['response']},
                {"role": "user", "content": "The user hasn't responded in a while. Please provide a follow-up message."}
            ],
            system_message=f"You are {settings['bot_name']}, a helpful AI assistant.",
            max_tokens=1000,
            temperature=0.7,
            chat_mode=chat_mode
        )

        await send_long_discord_message(channel, follow_up_response)
        logger.info(f"Sent auto-follow-up message in channel {channel_id}")

# Create the global message processor instance
message_processor = MessageProcessor()
