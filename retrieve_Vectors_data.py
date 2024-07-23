import pinecone_interaction

def fetch_context(index_query_response, match_score=0.55):

    context_chuncks = [match["metadata"]["text"] for match in index_query_response["matches"] if match["score"] > match_score]
    
    context = " ".join(context_chuncks)

    # print("Context: " + str(context))  #debug

    return context

def formatted_input(context, query):
    return f"Context: {context}\nQuery: {query}"


def query_index(vector_data, top_k_value=5):
    query_response = pinecone_interaction.query_index(vector_data, top_k_value)
    return query_response

'''debugging 
if __name__ == "__main__":
    pass
'''
