import embeddings_openAI 
import pinecone_interaction


def create_vector_embeding(file_path):
    with open(file_path, "r") as file:
        input_text = file.read()
        vector = []
        for line in input_text:
            vector.append(embeddings_openAI.create_vector(line))
        pinecone_interaction.upsert_vector(vector)

# from retrive_vectors_data import querry_vector
# retruning the ids of the top 5 matches in the pinecone index (vector database)
#----- do we need a score comarison to generate good responses or does it not matter if we dont compare?------
def querry_vector(querry):
    vector = embeddings_openAI.create_vector(querry)
    response = pinecone_interaction.query_embeddings(vector)

    match_id = []
    for match in response['matches']:
        match_id.append(match['id'])
        print(match['id'])

    return match_id

def remove_vectors(file_path):
    with open(file_path, "r") as file:
        input_text = file.read()
        for line in input_text:
            try:
                pinecone_interaction.delete_vectors(querry_vector(line))
                print("Deleted")
            except:
                print("Failed to delete")


if __name__ == "__main__": 
    create_vector_embeding("Julius_info.txt")
