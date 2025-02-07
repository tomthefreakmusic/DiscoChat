import os
import aiohttp
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from ..utils.logger import logger
from ..config.settings import settings

class LLMClient:
    def __init__(self):
        # Initialize Anthropic client
        self.anthropic_client = AsyncAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )

        # Initialize OpenAI client for DeepSeek
        self.deepseek_client = AsyncOpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )

    async def generate_anthropic_response(self, system_message: str, messages: list, max_tokens: int, temperature: float):
        """Generate a response using the Anthropic Claude model."""
        try:
            response = await self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_message,
                messages=messages
            )
            logger.info(f"Token usage for Claude response: Input tokens: {response.usage.input_tokens}, Output tokens: {response.usage.output_tokens}")
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error in Anthropic API call: {e}")
            raise

    async def generate_mistral_response(self, system_message: str, messages: list, max_tokens: int, temperature: float):
        """Generate a response using the Mistral API."""
        try:
            headers = {
                "Authorization": f"Bearer {os.getenv('MISTRAL_API_KEY')}",
                "Content-Type": "application/json"
            }
            
            api_messages = [{"role": "system", "content": system_message}] + messages
            
            payload = {
                "model": "mistral-large-latest",
                "messages": api_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 1,
                "stream": False,
                "safe_prompt": False,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    headers=headers,
                    json=payload
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data['choices'][0]['message']['content']
                    else:
                        error_data = await resp.text()
                        raise Exception(f"Mistral API error: Status {resp.status}, Response: {error_data}")

        except Exception as e:
            logger.error(f"Error in Mistral API call: {e}")
            raise

    async def generate_deepseek_response(self, system_message: str, messages: list, max_tokens: int, temperature: float):
        """Generate a response using the DeepSeek API."""
        try:
            formatted_messages = [{"role": "system", "content": system_message}]
            for msg in messages:
                if msg["role"] != "system":
                    formatted_messages.append(msg)

            response = await self.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=formatted_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )
            
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error in DeepSeek API call: {e}")
            raise

    async def get_response(self, messages: list, system_message: str, max_tokens: int, temperature: float, chat_mode: str):
        """Get a response based on the specified chat mode."""
        try:
            if chat_mode == "Extended Memory Mistral":
                try:
                    return await self.generate_mistral_response(
                        system_message,
                        messages,
                        max_tokens,
                        temperature
                    )
                except Exception as e:
                    logger.error(f"Error using Mistral API, falling back to Claude: {e}")
                    return await self.generate_anthropic_response(
                        system_message,
                        messages,
                        max_tokens,
                        temperature
                    )
            elif chat_mode == "DeepSeek":
                try:
                    return await self.generate_deepseek_response(
                        system_message,
                        messages,
                        max_tokens,
                        temperature
                    )
                except Exception as e:
                    logger.error(f"Error using DeepSeek API, falling back to Claude: {e}")
                    return await self.generate_anthropic_response(
                        system_message,
                        messages,
                        max_tokens,
                        temperature
                    )
            else:
                return await self.generate_anthropic_response(
                    system_message,
                    messages,
                    max_tokens,
                    temperature
                )

        except Exception as e:
            error_message = f"An error occurred while generating a response: {str(e)}"
            logger.error(error_message)
            return error_message

# Create the global LLM client instance
llm_client = LLMClient()
