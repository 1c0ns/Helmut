import fireworks.client

# Set your API key here
fireworks.client.api_key = "IyTsfmzfeGTsaayzX7prTJEyS6zZGDR4Aim47AtGR7BfBi5A"

prompt1 = "You are a random Pokemon and stick to the Pokemons traits. The pokemon you are can never change you will stick to it until the progam stops!"
prompt2 = "You are a Waiter in a high class restaurant."
prompt3 = "You are a detective. make sure that whatever the user says he comes out guilty."

def chat_with_ai():
    # Initialize array list that stores all messages
    messages = [{"role": "system", "content": prompt3}]

    # Start the conversation loop that keeps adding user messages and AI responses to the array list
    while True:
        # Get user input from the console
        user_message = input("You: ")
        
        # Append the user message to the array list that stores all messages
        messages.append({"role": "user", "content": user_message})
        
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
        
        if user_message == "exit":
            break

# Start chatting with the AI
chat_with_ai()