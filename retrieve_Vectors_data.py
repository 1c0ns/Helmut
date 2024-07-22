import pinecone_interaction
import embeddings_openAI

def query_vector(query, top_k):
    vector = embeddings_openAI.create_vector(query)
    response = pinecone_interaction.query_embeddings(vector, top_k)
    return response
'''
    match_id = []
    for match in response['matches']:
        match_id.append(match['id'])
        print(match['id'])

    return match_id
'''

def fetch_content(index_query_response):
    # print(index_query_response) #debug

    context_chuncks = [match["metadata"]["text"] for match in index_query_response["matches"] if match["score"] > 0.55]
    
    context = " ".join(context_chuncks)

    # print("Context: " + str(context))  #debug

    return context

'''
def get_context(query):
    query_response = query_vector(query, top_k=5)
    index_query_response = index_query(query_response)
    context = fetch_content(index_query_response)

    return context
'''

def query_vector(query, top_k):
    vector = embeddings_openAI.create_vector(query)
    response = pinecone_interaction.query_embeddings(vector, top_k)

    match_ids = []
    for match in response['matches']:
        match_ids.append(match['id'])
        print(match['id'])

    return match_ids

def fetch_vector(vector_ids):
    response = pinecone_interaction.fetch_vector(vector_ids)
    return response

def formatted_input(context, query):
    return f"Context: {context}\nQuery: {query}"

def index_query(vector_data, top_k_value=5):
    query_response = pinecone_interaction.index_query(vector_data, top_k_value)
    return query_response

'''debugging 
if __name__ == "__main__":
    vector_ids = query_vector("Julius")
    print(vector_ids)

    print(fetch_vector(vector_ids))
'''
