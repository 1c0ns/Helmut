import embeddings_openAI 
import pinecone_interaction



def create_vector_embeding(chunck_data):
    vector = []
    for line in chunck_data:
        vector.append(embeddings_openAI.create_vector(line))
    pinecone_interaction.upsert_vector(vector, metadata=chunck_data)


def chunck_text(file_path, chunck_size=500):
    with open(file_path, "r") as file:
        input_text = file.read()
        chuncked_text = []
        for i in range(0, len(input_text), chunck_size):
            chuncked_text.append(input_text[i:i+chunck_size])
        return chuncked_text



# from retrive_vectors_data import query_vector
# retruning the ids of the top 5 matches in the pinecone index (vector database)
#----- do we need a score comarison to generate good responses or does it not matter if we dont compare?------
def query_vector(query):
    vector = embeddings_openAI.create_vector(query)
    response = pinecone_interaction.query_embeddings(vector)

    match_ids = []
    for match in response['matches']:
        match_ids.append(match['id'])
        print(match['id'])

    return match_ids

def remove_vectors(file_path):
    with open(file_path, "r") as file:
        input_text = file.read()
        for line in input_text:
            try:
                pinecone_interaction.delete_vectors(query_vector(line))
                print("Deleted")
            except:
                print("Failed to delete")


if __name__ == "__main__": 
    # remove_vectors("Julius_info.txt")
    create_vector_embeding(chunck_text("none.txt", chunck_size=500))

