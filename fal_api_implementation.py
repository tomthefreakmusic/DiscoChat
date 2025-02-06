""" class ConfigView(discord.ui.View):
    def __init__(self, user_id):
        super().__init__()
        self.user_id = user_id
        self.config = load_user_config(user_id)
        self.add_item(ImageSizeSelect(self.config.get('image_size', DEFAULT_CONFIG['image_size'])))
        self.add_item(ModelSelect(self.config.get('model', DEFAULT_CONFIG['model'])))

    @discord.ui.button(label="Toggle Prompt Enhancement", style=discord.ButtonStyle.primary)
    async def toggle_prompt_enhancement(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.config['enhance_prompt'] = not self.config.get('enhance_prompt', False)
        status = "enabled" if self.config['enhance_prompt'] else "disabled"
        await interaction.response.send_message(f"Prompt enhancement {status}.", ephemeral=True)

    @discord.ui.button(label="Set Inference Steps", style=discord.ButtonStyle.primary)
    async def set_inference_steps(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(InferenceStepsModal(self))

    @discord.ui.button(label="Set Number of Images", style=discord.ButtonStyle.primary)
    async def set_num_images(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(NumImagesModal(self))

    @discord.ui.button(label="Set Guidance Scale", style=discord.ButtonStyle.primary)
    async def set_guidance_scale(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(GuidanceScaleModal(self))

    @discord.ui.button(label="Save Configuration", style=discord.ButtonStyle.success)
    async def save_config(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not all([self.config.get('image_size'), self.config.get('model'), 
                    self.config.get('num_inference_steps'), 
                    self.config.get('num_images'),
                    self.config.get('guidance_scale')]):
            await interaction.response.send_message("Please set all configuration options before saving.", ephemeral=True)
            return

        save_user_config(self.user_id, self.config)
        await interaction.response.send_message("Configuration saved successfully!", ephemeral=True)
        self.stop()

class ImageSizeSelect(discord.ui.Select):
    def __init__(self, default):
        options = [
            discord.SelectOption(label="Square HD", value="square_hd"),
            discord.SelectOption(label="Square", value="square"),
            discord.SelectOption(label="Portrait 4:3", value="portrait_4_3"),
            discord.SelectOption(label="Portrait 16:9", value="portrait_16_9"),
            discord.SelectOption(label="Landscape 4:3", value="landscape_4_3"),
            discord.SelectOption(label="Landscape 16:9", value="landscape_16_9"),
        ]
        super().__init__(placeholder="Select image size", options=options)
        self.default = default

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer()
        self.view.config['image_size'] = self.values[0]
        await interaction.followup.send(f"Image size set to {self.values[0]}", ephemeral=True)

class ModelSelect(discord.ui.Select):
    def __init__(self, default):
        options = [
            discord.SelectOption(label="Flux Pro", value="fal-ai/flux-pro"),
            discord.SelectOption(label="Flux Schnell", value="fal-ai/flux/schnell"),
            discord.SelectOption(label="Flux Dev", value="fal-ai/flux/dev"),
            discord.SelectOption(label="Flux Realism", value="fal-ai/flux-realism"),
            discord.SelectOption(label="Flux General", value="fal-ai/flux-general")
        ]
        super().__init__(placeholder="Select model", options=options)
        self.default = default

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer()
        self.view.config['model'] = self.values[0]
        await interaction.followup.send(f"Model set to {self.values[0]}", ephemeral=True)

class InferenceStepsModal(discord.ui.Modal, title='Set Inference Steps'):
    steps = discord.ui.TextInput(label='Inference Steps', default='28')

    def __init__(self, view):
        super().__init__()
        self.view = view
        self.steps.default = str(view.config.get('num_inference_steps', DEFAULT_CONFIG['num_inference_steps']))

    async def on_submit(self, interaction: discord.Interaction):
        try:
            steps = int(self.steps.value)
            if steps <= 0:
                raise ValueError
        except ValueError:
            await interaction.response.send_message("Please enter a positive integer for inference steps.", ephemeral=True)
            return

        self.view.config['num_inference_steps'] = steps
        await interaction.response.send_message(f"Inference steps set to {steps}.", ephemeral=True)


class NumImagesModal(discord.ui.Modal, title='Set Number of Images'):
    num_images = discord.ui.TextInput(label='Number of Images', default='1')

    def __init__(self, view):
        super().__init__()
        self.view = view
        self.num_images.default = str(view.config.get('num_images', DEFAULT_CONFIG['num_images']))

    async def on_submit(self, interaction: discord.Interaction):
        try:
            num = int(self.num_images.value)
            if num <= 0:
                raise ValueError
        except ValueError:
            await interaction.response.send_message("Please enter a positive integer for number of images.", ephemeral=True)
            return

        self.view.config['num_images'] = num
        await interaction.response.send_message(f"Number of images set to {num}.", ephemeral=True)

class GuidanceScaleModal(discord.ui.Modal, title='Set Guidance Scale'):
    guidance_scale = discord.ui.TextInput(label='Guidance Scale', default='3.5')

    def __init__(self, view):
        super().__init__()
        self.view = view
        self.guidance_scale.default = str(view.config.get('guidance_scale', DEFAULT_CONFIG['guidance_scale']))

    async def on_submit(self, interaction: discord.Interaction):
        try:
            scale = float(self.guidance_scale.value)
            if scale <= 0:
                raise ValueError
        except ValueError:
            await interaction.response.send_message("Please enter a positive number for guidance scale.", ephemeral=True)
            return

        self.view.config['guidance_scale'] = scale
        await interaction.response.send_message(f"Guidance scale set to {scale}.", ephemeral=True)

@client.tree.command()
async def configure_image_gen(interaction: discord.Interaction):

    if not check_permissions(interaction):
        logger.warning(f"Permission denied for user {interaction.user.name} in {interaction.guild.name if interaction.guild else 'DM'}")
        await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
        return

    view = ConfigView(interaction.user.id)
    await interaction.response.send_message("Please configure your image generation settings:", view=view, ephemeral=True)

@client.tree.command()
@app_commands.describe(prompt="The prompt for image generation")
async def generate_image(interaction: discord.Interaction, prompt: str):

    if not check_permissions(interaction):
        logger.warning(f"Permission denied for user {interaction.user.name} in {interaction.guild.name if interaction.guild else 'DM'}")
        await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
        return

    await interaction.response.defer()

    # Load user configuration
    config = load_user_config(interaction.user.id)
    
    # Use default values for any missing configuration items
    for key, default_value in DEFAULT_CONFIG.items():
        if key not in config:
            config[key] = default_value

    # Start the image generation process in a separate task
    client.loop.create_task(process_image_generation(interaction, prompt, config))

    await interaction.followup.send("Image generation started. Please wait...")

async def process_image_generation(interaction: discord.Interaction, prompt: str, config: dict):
    try:
        # Enhance the prompt if the option is enabled
        if config['enhance_prompt']:
            enhanced_prompt = await enhance_prompt(prompt)
        else:
            enhanced_prompt = prompt

        if config["model"] == "fal-ai/flux-general":
            loras = [{
                "path": "https://storage.googleapis.com/fal-flux-lora/d119f2d4b0f24ac1a59b28950371d624_lora.safetensors",
                "weight": 1
            }]
        else:
            loras = []

        handler = fal_client.submit(
            config['model'],
            arguments={
                "prompt": enhanced_prompt,
                "image_size": config['image_size'],
                "num_inference_steps": config['num_inference_steps'],
                "num_images": config['num_images'],
                "loras": loras,
                "guidance_scale": config['guidance_scale'],
            },
        )

        result = await asyncio.to_thread(handler.get)

        for i, image_info in enumerate(result['images']):
            image_url = image_info['url']
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as resp:
                    if resp.status != 200:
                        logger.error(f"Failed to download image {i+1}. Status: {resp.status}")
                        await interaction.followup.send(f"An error occurred while downloading image {i+1}.")
                        continue
                    image_data = await resp.read()

            # Open the image with PIL
            image = Image.open(io.BytesIO(image_data))

            # Create PngInfo object and add metadata
            metadata = PngInfo()
            metadata.add_text("Enhanced Prompt", enhanced_prompt)

            # Save the image with metadata to a new bytes buffer
            buffer = io.BytesIO()
            image.save(buffer, format="PNG", pnginfo=metadata)
            buffer.seek(0)

            # Create Discord file object
            file = discord.File(buffer, filename=f"generated_image_{i+1}.png")

            # Send the image with a simple message
            await interaction.followup.send(
                content=f"Generated image {i+1} for prompt: '{prompt}'",
                file=file
            )

        logger.info(f"Successfully generated and sent {config['num_images']} image(s) for user {interaction.user.name}")

    except Exception as e:
        logger.error(f"Error generating images: {str(e)}", exc_info=True)
        await interaction.followup.send("An error occurred while generating the images. Please try again.")

@client.tree.command()
@app_commands.describe(
    image="The image to upscale (attach an image)",
    image_url="URL of the image to upscale (if not attaching)",
    upscaling_factor="Upscaling factor (currently only 4x is supported)",
    overlapping_tiles="Use overlapping tiles to reduce seams (slower)",
    checkpoint="Checkpoint to use for upscaling"
)
async def upscale(
    interaction: discord.Interaction,
    image: discord.Attachment = None,
    image_url: str = None,
    upscaling_factor: Literal["4"] = "4",
    overlapping_tiles: bool = False,
    checkpoint: Literal["v1", "v2"] = "v2"
):

    if not check_permissions(interaction):
        logger.warning(f"Permission denied for user {interaction.user.name} in {interaction.guild.name if interaction.guild else 'DM'}")
        await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
        return

    await interaction.response.defer()

    if not image and not image_url:
        await interaction.followup.send("Please provide an image or an image URL.")
        return

    if image:
        image_url = image.url

    try:
        handler = fal_client.submit(
            "fal-ai/aura-sr",
            arguments={
                "image_url": image_url,
                "upscaling_factor": int(upscaling_factor),
                "overlapping_tiles": overlapping_tiles,
                "checkpoint": checkpoint
            },
        )

        result = await asyncio.to_thread(handler.get)

        upscaled_image_url = result['image']['url']

        # Download the upscaled image
        async with aiohttp.ClientSession() as session:
            async with session.get(upscaled_image_url) as resp:
                if resp.status != 200:
                    await interaction.followup.send("An error occurred while downloading the upscaled image.")
                    return
                image_data = await resp.read()

        # Create a file object from the image data
        file = discord.File(BytesIO(image_data), filename="upscaled_image.png")

        # Send the upscaled image
        await interaction.followup.send(f"Here's your upscaled image:", file=file)

    except Exception as e:
        logger.error(f"Error upscaling image: {str(e)}", exc_info=True)
        await interaction.followup.send("An error occurred while upscaling the image. Please try again.")

def save_user_config(user_id, config):
    if not os.path.exists('user_configs'):
        os.makedirs('user_configs')
    
    with open(f'user_configs/{user_id}.json', 'w') as f:
        json.dump(config, f)

def load_user_config(user_id):
    if os.path.exists(f'user_configs/{user_id}.json'):
        with open(f'user_configs/{user_id}.json', 'r') as f:
            config = json.load(f)
            # Ensure the new option is present
            if 'enhance_prompt' not in config:
                config['enhance_prompt'] = DEFAULT_CONFIG['enhance_prompt']
            return config
    return DEFAULT_CONFIG.copy()

async def enhance_prompt(prompt):
    system_message = "You are an AI assistant specializing in enhancing image generation prompts. Your task is to take a user's brief prompt and expand it into a more detailed and vivid description. Focus on adding specific details about the scene, lighting, mood, and style. Keep the enhanced prompt concise and directly usable for image generation. Provide only the enhanced prompt without any introductory text or explanations."

    user_message = f"Please enhance the following image generation prompt: {prompt}"

    try:
        response = await async_anthropic_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=150,
            temperature=0.7,
            system=system_message,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        enhanced_prompt = response.content[0].text.strip()
        return enhanced_prompt
    except Exception as e:
        logger.error(f"Error enhancing prompt: {str(e)}", exc_info=True)
        return prompt  # Return the original prompt if enhancement fails
            """