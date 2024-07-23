import openai

client = openai.OpenAI(
    base_url = "https://api.fireworks.ai/inference/v1",
    api_key="IyTsfmzfeGTsaayzX7prTJEyS6zZGDR4Aim47AtGR7BfBi5A",
)

def create_vector(input_text):
    """
    Creates a vector representation of the input text using the OpenAI embeddings model.

    Args:
        input_text (str): The text to be embedded.

    Returns:
        list: A list representing the vector embedding of the input text.
    """
    response = client.embeddings.create(
        model="nomic-ai/nomic-embed-text-v1.5",
        input=input_text,
        dimensions=512,
    )
    vector = response.data[0].embedding
    return vector







