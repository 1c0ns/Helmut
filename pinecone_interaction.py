from pinecone import Pinecone
import uuid

pc = Pinecone(api_key="0091548f-d131-45be-b986-e62da549d4d3")
index = pc.Index(name="vector", host="https://vector-fn3jdmm.svc.aped-4627-b74a.pinecone.io")

def generateId():
    return str(uuid.uuid4())

def generateIds(n:int):
    ids = []
    for i in range(n):
        ids.append(generateId())
    return list[str](ids)

def upsert_vector(vector: list, identify: list[str]=None, metadata: str=None, namespace: str=""):
    """
    Upsert vectors to the Pinecone index
    :param vector: list of vectors to upsert
    :param identify: list of identifiers for the vectors
    :param metadata: metadata for the vectors optional
        data will be used for all vectors upserted in one go
    :param namespace: namespace for the vectors optional
        data will be used for all vectors upserted in one go
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
    index.delete(ids=vector_ids)

def query_index(vector_data, top_k_value=5):

    query_response = index.query(vector=vector_data, filter={"text": {"$exists": True}}, top_k=top_k_value, include_metadata=True)

    return query_response
