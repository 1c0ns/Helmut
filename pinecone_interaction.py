from pinecone import Pinecone
import uuid

pc = Pinecone(api_key="0091548f-d131-45be-b986-e62da549d4d3")
index = pc.Index(name="vector", host="https://vector-fn3jdmm.svc.aped-4627-b74a.pinecone.io")

def generateId():
    """
    Generate a unique identifier using UUID.

    Returns:
        str: A unique identifier.
    """
    return str(uuid.uuid4())

def generateIds(n:int):
    """
    Generate a list of unique identifiers.

    Args:
        n (int): The number of identifiers to generate.

    Returns:
        List[str]: A list of unique identifiers.
    """
    ids = []
    for i in range(n):
        ids.append(generateId())
    return list[str](ids)

def upsert_vector(vector: list, identify: list[str]=None, metadata: str=None, namespace: str=""):
    """
    Upsert vectors to the Pinecone index.

    Args:
        vector (list): The list of vectors to upsert.
        identify (list[str], optional): The list of identifiers for the vectors. If not provided, new identifiers will be generated. Defaults to None.
        metadata (str, optional): The metadata associated with the vectors. Defaults to None.
        namespace (str, optional): The namespace for the vectors. Defaults to "".
    """
    if identify is None:
        identify = generateIds(len(vector))

    if identify is not None and len(identify) != len(vector):
        print("Length of identify list must be equal to the length of the vector list")
        return

    # Prepare the vector data for upsert
    vector_data = [{"id": identify[i], "values": vector[i], "metadata": {"text": metadata[i]}} for i in range(len(vector))]
    
    # Upsert the vector to the Pinecone index
    # Upsert create or if the vector already exists it will be updated
    index.upsert(
        vectors=vector_data,
        namespace=namespace
        )
    
    # Confirmation message
    print("Upserted vectors to Pinecone index")

def delete_vectors(vector_ids):
    """
    Delete vectors from the Pinecone index.

    Args:
        vector_ids (list): The list of identifiers for the vectors to delete.
    """
    index.delete(ids=vector_ids)

def query_index(vector_data, top_k_value=5):
    """
    Query the Pinecone index.

    Args:
        vector_data (list): The vector data for the query.
        top_k_value (int, optional): The number of nearest neighbors to retrieve. Defaults to 5.

    Returns:
        dict: The query response containing the nearest neighbors and their metadata.
    """
    query_response = index.query(vector=vector_data, filter={"text": {"$exists": True}}, top_k=top_k_value, include_metadata=True)

    return query_response
