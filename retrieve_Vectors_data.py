import pinecone_interaction

def fetch_context(index_query_response, match_score=0.55):
    """
    Fetches the context chunks from the index query response based on the match score.

    Args:
        index_query_response (dict): The response from the index query.
        match_score (float, optional): The minimum match score required to include a context chunk. Defaults to 0.55.

    Returns:
        str: The concatenated context chunks.
    """
    context_chunks = [match["metadata"]["text"] for match in index_query_response["matches"] if match["score"] > match_score]
    context = " ".join(context_chunks)
    return context

def formatted_input(context, query):
    """
    Formats the context and query into a single string AI prompt.

    Args:
        context (str): The context to be included in the formatted input.
        query (str): The query to be included in the formatted input.

    Returns:
        str: The formatted input string.
    """
    return f"Context: {context}\nQuery: {query}"

def query_index(vector_data, top_k_value=5):
    """
    Queries the index with the given vector data and retrieves the top k matches.

    Args:
        vector_data (list): The vector data to be used for querying the index.
        top_k_value (int, optional): The number of top matches to retrieve. Defaults to 5.

    Returns:
        dict: The response from the index query.
    """
    query_response = pinecone_interaction.query_index(vector_data, top_k_value)
    return query_response

'''debugging 
if __name__ == "__main__":
    pass
'''