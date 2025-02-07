import asyncio
import json
import discord
from discord import app_commands
from discord.ui import View, Button
import io
from ..utils.logger import logger
from ..utils.helpers import check_permissions
from ..message_handling.processor import message_processor
from ..image_gen.generator import image_generator
from ..config.settings import settings

class CustomClient(discord.Client):
    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        await self.tree.sync()

class DiscordBot:
    def __init__(self):
        # Set up intents
        intents = discord.Intents.default()
        intents.messages = True
        intents.guild_messages = True
        intents.message_content = True

        # Create the client
        self.client = CustomClient(intents=intents)
        self.setup_event_handlers()
        self.setup_commands()

    def setup_event_handlers(self):
        @self.client.event
        async def on_ready():
            logger.info(f"{self.client.user} has connected to Discord!")
            logger.info(f"Connected servers: {', '.join([guild.name for guild in self.client.guilds])}")
            
            # Log the synced commands
            synced_commands = await self.client.tree.sync()
            logger.info(f"Synced {len(synced_commands)} command(s): {', '.join([cmd.name for cmd in synced_commands])}")

        @self.client.event
        async def on_message(message):
            await message_processor.process_message(message)

    def setup_commands(self):
        @self.client.tree.command()
        @app_commands.describe(enabled="Enable or disable auto follow-up")
        async def toggle_auto_follow_up(interaction: discord.Interaction, enabled: bool):
            """Toggle the auto follow-up feature for this channel"""
            if not check_permissions(interaction):
                await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
                return

            # Update configuration
            channel_id = interaction.channel.id
            config_path = f"./config/{channel_id}.json"
            with open(config_path, "r") as f:
                channel_config = json.load(f)
            
            channel_config["auto_follow_up"] = enabled
            
            with open(config_path, "w") as f:
                json.dump(channel_config, f)

            await interaction.response.send_message(
                f"Auto follow-up has been {'enabled' if enabled else 'disabled'} for this channel."
            )

        @self.client.tree.command()
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
                # Generate the image
                image_data, filename = await image_generator.generate_image(
                    prompt, variant, width, height, steps, prompt_upscaling, guidance
                )
                
                if image_data and filename:
                    # Create a file object from the image data
                    file = discord.File(io.BytesIO(image_data), filename=filename)
                    # Send the generated image
                    await interaction.followup.send(f"Generated image for prompt: '{prompt}'", file=file)
                else:
                    await interaction.followup.send("Failed to generate the image. Please try again.")

            except Exception as e:
                logger.error(f"Error generating image: {str(e)}", exc_info=True)
                await interaction.followup.send("An error occurred while generating the image. Please try again.")

    async def run(self):
        """Run the Discord bot."""
        if settings["TOKEN"] is None:
            raise ValueError("Discord token is not set.")
        
        try:
            await self.client.start(settings["TOKEN"])
        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            raise

# Create the main entry point
def main():
    """Main entry point for the Discord bot."""
    try:
        # Create and run the bot
        bot = DiscordBot()
        asyncio.run(bot.run())
    except Exception as e:
        logger.critical(f"Critical error in main: {e}")
        raise

if __name__ == "__main__":
    main()
