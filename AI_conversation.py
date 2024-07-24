import fireworks.client
import retrieve_Vectors_data
import embeddings_openAI

fireworks.client.api_key = "IyTsfmzfeGTsaayzX7prTJEyS6zZGDR4Aim47AtGR7BfBi5A"

prompt1 = "You are a random Pokemon and stick to the Pokemons traits. The pokemon you are can never change you will stick to it until the progam stops!"
prompt2 = "You are a Waiter in a high class restaurant."
prompt3 = "You are a detective. make sure that whatever the user says he comes out guilty."
prompt4 = None
with open("data_files//prof_prompt.txt", "r") as file:
    prompt4 = file.read()

def chat_with_ai(prompt: str):
    """
    Function to start a conversation with the AI.

    This function initializes an array list to store all messages and starts a conversation loop.
    It prompts the user for input, appends the user message to the array list, generates an AI response,
    and appends the AI's response to the array list. The loop continues until the user enters "exit".
    """

    # Initialize array list that stores all messages
    messages = [{"role": "system", "content": prompt}]

    # Start the conversation loop that keeps adding user messages and AI responses to the array list
    while True:
        # Get user input from the console
        user_message = input("You: ")
        
        # Append the user message to the array list that stores all messages
        messages.append({"role": "user", "content": handle_user_input(user_message)})       # handle_user_input(user_message) to create a prompt
        
        # Create a chat completion with the current conversation history
        completion = fireworks.client.ChatCompletion.create(
            "accounts/fireworks/models/llama-v3-70b-instruct",
            messages=messages,
            temperature=1.3,
            n=1,
            max_tokens=250
        )
        
        # Get the AI's response and print it
        ai_message = completion.choices[0].message.content
        print("AI: ", ai_message)
        
        # Append the AI's response to the array list that stores all messages
        messages.append({"role": "assistant", "content": ai_message})
        
        if user_message.lower() == "exit":
            break


def handle_user_input(user_input, top_k=5):
    """
    Function to handle user input and generate a prompt for the AI.

    This function takes the user's input, creates a vector representation of the input using embeddings_openAI,
    queries an index to retrieve relevant context, and formats the input into a prompt for the AI.

    Args:
        user_input (str): The user's input.
        top_k (int, optional): The number of top results to retrieve from the index. Defaults to 5.

    Returns:
        str: The formatted prompt for the AI.
    """

    vector = embeddings_openAI.create_vector(user_input)

    query_response = retrieve_Vectors_data.query_index(vector, top_k)

    context = retrieve_Vectors_data.fetch_context(query_response)

    prompt = retrieve_Vectors_data.formatted_input(context, user_input)

    return prompt

# Start chatting with the AI
if __name__ == "__main__":
    chat_with_ai(prompt=prompt4)