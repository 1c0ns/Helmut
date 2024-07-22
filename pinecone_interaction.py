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

    query_response = []
    for vec in vector:
        query_response.append(query_embeddings(vec, top_k = 1))

    query_response_array = [bool]

    ids = []
    scores = []
    ids, scores = separate_ids_and_scores(query_response)
    for i in range(len(ids)):
        query_response_array.append(vector_already_in_index(ids[i], scores[i]))

    '''
    try:
        # Attempt to convert all elements to float
        vector = [float(item) for item in vector]
    except ValueError as e:
        print(f"Error converting vector elements to float: {e}")
        return  # Exit the function if conversion fails
    '''
    #print("vector len before: " + str(len(vector))) #debug

    # Prepare the vector data for upsert
    vector_data = [{"id": identify[i], "values": vector[i]} for i in range(len(vector)) if not query_response_array[i]]

    # print("vector len after: " + str(len(vector_data))) #debug
    # return #debug

    # Upsert the vector to the Pinecone index
    # Upsert create or if the vector already exists it will be updated
    index.upsert(
        vectors=vector_data,
        namespace=namespace
        )
    
    # Confirmation message
    print("Upserted vector to Pinecone index")

def separate_ids_and_scores(data):
    ids =[]
    scores =[]

    for item in data:
        matches = item['matches']
        for match in matches:
            ids.append(match['id'])
            scores.append(match['score'])
    return ids, scores

def vector_already_in_index(id, score):

    if float(score) > 0.95:
        return True
    return False

def query_embeddings(vector, top_k=5):
    results = index.query(vector=vector, top_k=top_k)
    print("Queried Pinecone index")
    return results

def delete_vectors(vector_ids, score=None):
    index.delete(ids=vector_ids)

def fetch_vector(vector_ids):
    '''
    ids = [str]
    for id in vector_ids:
        ids.append(id)
    vector_data = index.fetch(ids=ids)
    return vector_data
    '''
    vectors = []

    vectors.append(index.fetch(ids=vector_ids))

    return vectors

