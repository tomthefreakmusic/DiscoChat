from config import (
    bot_name,
    token_encoder,
    model,
    max_response_tokens,
    previous_relevant_messages,
)
from database_storage import message_bank
from chat_completion import chat_completion_create


# Defines a function that queries the database based on the query contents and bounds.
def retrieve_relevant_messages(message, token_length):
    query = message.clean_content
    channel = str(message.channel.id)
    distance_threshold = 0.6

    # Set the where conditions to only search in the channel

    where_conditions = {"$and": [{"channel": channel}, {"is_command": "False"}]}

    if f"@{bot_name}" in query.lower():
        query = query.replace(f"@{bot_name}", "")

    relevant_messages = message_bank.query(
        query_texts=query,
        n_results=25,
        where=where_conditions,  # type: ignore
    )

    # now we should grab the relevant_messages dictionary, which is formatted as follows. the keys of this dictionary are ids, metadatas, documents and distances.
    # the values of these keys are lists of size len(query) where query is a list.
    # so if there is only one query, for each of the keys there is a list which features n_results within a list.
    # so first we need to extract the lists and grab the first element of the list
    # (as this will be the results, there won't be other elements as we have only passed one query text)
    # then, we need to extract the individual elements of each list, and use this information to generate the final relevant messages output.

    ids = relevant_messages["ids"][0]
    documents = relevant_messages["documents"][0]  # type: ignore
    metadatas = relevant_messages["metadatas"][0]  # type: ignore
    distances = relevant_messages["distances"][0]  # type: ignore

    result_string = ""
    for i in range(len(ids)):
        distance = distances[i]
        if distance <= distance_threshold and distance >= 0.05:
            author = metadatas[i]["author"]
            document = documents[i]
            # rounded_distance = round(distance, 2)

            temp_string = f"{author}: {document}, "
            current_message_tokens = len(
                token_encoder.encode(result_string + temp_string)
            )

            if current_message_tokens <= token_length:
                result_string += temp_string
            else:
                break  # If adding next message would exceed token limit, break the loop

    result_string = result_string[:-2]

    # print(f"{result_string} is of length {len(token_encoder.encode(result_string))}")

    store_relevant_messages(message, result_string)
    return result_string


# Defines a alternative function that queries the database based on the query contents and bounds.
async def alt_retrieve_relevant_messages(message, token_length):
    query = message.clean_content
    channel = str(message.channel.id)
    distance_threshold = 0.7

    # Set the where conditions to only search in the channel
    where_conditions = {"$and": [{"channel": channel}, {"is_command": "False"}]}

    relevant_messages = message_bank.query(
        query_texts=query,
        n_results=5,
        where=where_conditions,  # type: ignore
    )

    ids = relevant_messages["ids"][0]
    documents = relevant_messages["documents"][0]  # type: ignore
    metadatas = relevant_messages["metadatas"][0]  # type: ignore
    distances = relevant_messages["distances"][0]  # type: ignore

    result_string = ""
    for i in range(len(ids)):
        distance = distances[i]
        if distance <= distance_threshold and distance >= 0.00:
            author = metadatas[i]["author"]
            document = documents[i]
            message_id = ids[i]  # Discord message ID is stored in ids

            # Fetch message object by id
            message_around = await message.channel.fetch_message(message_id)
            near_messages = [
                msg
                async for msg in message.channel.history(
                    limit=5, around=message_around, oldest_first=True
                )
            ]

            for msg in near_messages:
                temp_string = f"{str(msg.created_at)[:-16]} {msg.author.name}: {msg.clean_content}, "
                current_message_tokens = len(
                    token_encoder.encode(result_string + temp_string)
                )

                if current_message_tokens <= token_length:
                    result_string += temp_string
                else:
                    break  # If adding next message would exceed token limit, break the loop

    result_string = result_string[:-2]

    # print(f"{result_string} is of length {len(token_encoder.encode(result_string))}")

    store_relevant_messages(message, result_string)
    return result_string


# Defines a function that stores relevant messages in a dictionary.
def store_relevant_messages(message, result_string):
    channel = str(message.channel.id)

    messages = [
        {"role": "assistant", "content": f"summarize these messages: {result_string}"},
    ]

    summary = chat_completion_create(model, messages, max_response_tokens)
    summary = summary["choices"][0]["message"]["content"]  # type: ignore
    previous_relevant_messages[channel] = summary


# Defines a helper function that retrieves the strings from "previous_relevant_messages" for the channel and returns the string.
def retrieve_previous_relevant_messages(message):
    channel = str(message.channel.id)
    if channel in previous_relevant_messages:
        result_string = previous_relevant_messages[channel]
        return result_string
    else:
        return ""


# Defines a function that fetches most recent messages from discord based on the token length bounds.
async def retrieve_recent_messages(message, token_length):
    # defines a list to store the history
    recent_messages = []

    message_number = 0
    async for message in message.channel.history(limit=11):
        if message_number == 0:
            message_number += 1
            continue
        message_number += 1

        # format messages with username and message content, and a timestamp for the earliest message.
        # !!! i need to make sure that timestamp is applied to earliest message regardless of whether we get to 21 messages.
        # for timestamp, we want to strip it back to a useful format
        timestamp = str(message.created_at)[:-16]

        formatted_message = (
            f"[{timestamp}] {message.author.name}: {message.clean_content} "
        )

        current_message_tokens = len(token_encoder.encode(formatted_message))
        if token_length - current_message_tokens < 0:
            break

        # append formatted message to history
        recent_messages.append(formatted_message)

        token_length -= current_message_tokens

    # reverse the history list so that the messages are in chronological order.
    recent_messages.reverse()
    # print(recent_messages)
    # returns the recent messages from the channel upto the length requested.
    return recent_messages
