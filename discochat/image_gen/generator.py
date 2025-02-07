import asyncio
import io
import time
import re
import aiohttp
from ..utils.logger import logger
from ..config.settings import settings

class ImageGenerator:
    def __init__(self):
        self.bfl_api_key = settings["BFL_API_KEY"]

    def sanitize_filename(self, prompt):
        """Sanitize the prompt to create a safe filename."""
        # Remove invalid characters (keep only alphanumerics, spaces, underscores, and hyphens)
        sanitized = re.sub(r'[^A-Za-z0-9 _-]', '', prompt)
        # Replace spaces with underscores
        sanitized = sanitized.replace(' ', '_')
        # Truncate the filename to a maximum length
        return sanitized[:40]

    async def generate_image(self, prompt: str, variant: str = "flux-pro-1.1",
                           width: int = 1024, height: int = 1024,
                           steps: int = 30, prompt_upscaling: bool = False,
                           guidance: float = 2.5):
        """Generate an image based on the provided parameters."""
        try:
            # Get the image URL from the API
            image_url = await self.process_image_generation(
                prompt, variant, width, height, steps, prompt_upscaling, guidance
            )
            
            if not image_url:
                return None, None

            # Download the image
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as resp:
                    if resp.status != 200:
                        logger.error("Error downloading generated image")
                        return None, None
                    image_data = await resp.read()

            # Create a filename
            sanitized_prompt = self.sanitize_filename(prompt)
            timestamp = int(time.time() * 1000) % 100000  # 5-digit timestamp
            filename = f"{sanitized_prompt}_{timestamp}.png"

            return image_data, filename

        except Exception as e:
            logger.error(f"Error generating image: {str(e)}", exc_info=True)
            return None, None

    async def process_image_generation(self, prompt, variant, width, height, steps, prompt_upscaling, guidance):
        """Process the image generation request with the API."""
        try:
            # Construct the endpoint URL based on the variant
            endpoint_url = f'https://api.bfl.ml/v1/{variant}'

            # Create the initial request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint_url,
                    headers={
                        'accept': 'application/json',
                        'x-key': self.bfl_api_key,
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
                            'x-key': self.bfl_api_key,
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

# Create the global image generator instance
image_generator = ImageGenerator()
