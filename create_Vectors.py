import embeddings_openAI 
import pinecone_interaction

def create_vector_embeding(chunck_data):
    """
    Creates vector embeddings for each line in the given chunk data.

    Parameters:
    chunck_data (list): A list of strings representing the chunked data.

    Returns:
    None
    """
    vector = []
    for line in chunck_data:
        vector.append(embeddings_openAI.create_vector(line))
    pinecone_interaction.upsert_vector(vector, metadata=chunck_data)

def chunck_text(file_path, chunck_size=500):
    """
    Chunks the text from the given file path into smaller chunks of the specified size.

    Parameters:
    file_path (str): The path to the file containing the text.
    chunck_size (int): The size of each chunk. Default is 500.

    Returns:
    list: A list of strings representing the chunked text.
    """
    with open(file_path, "r") as file:
        input_text = file.read()
        chuncked_text = []
        for i in range(0, len(input_text), chunck_size):
            chuncked_text.append(input_text[i:i+chunck_size])
        return chuncked_text

if __name__ == "__main__": 
    create_vector_embeding(chunck_text("none.txt", chunck_size=500))    #filename "none.txt" is just a placeholder

