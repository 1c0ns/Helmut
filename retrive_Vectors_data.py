import pinecone_interaction
import embeddings_openAI

def querry_vector(querry):
    vector = embeddings_openAI.create_vector(querry)
    response = pinecone_interaction.query_embeddings(vector)

    match_id = []
    for match in response['matches']:
        match_id.append(match['id'])
        print(match['id'])

    return match_id

def fetch_vector(vector_ids):
    response = pinecone_interaction.fetch_vector(vector_ids)
    
    vector_data = response

    return vector_data

if __name__ == "__main__":
    vector_ids = querry_vector("Julius")
    print(vector_ids)

    print(fetch_vector(vector_ids))
