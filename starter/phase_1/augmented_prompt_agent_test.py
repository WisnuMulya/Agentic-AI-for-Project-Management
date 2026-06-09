from workflow_agents.base_agents import AugmentedPromptAgent
import os

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"
persona = (
    "You are a college professor; your answers always start with: 'Dear students,'"
)

# Instantiate an object of AugmentedPromptAgent with the required parameters
augmented_agent = AugmentedPromptAgent(openai_api_key=openai_api_key, persona=persona)

# Send the 'prompt' to the agent and store the response in a variable named 'augmented_agent_response'
augmented_agent_response = augmented_agent.respond(prompt)

# Print the agent's response
print(augmented_agent_response)

# - What knowledge the agent likely used to answer the prompt.
# The agent's response is based on the selected LLM's pre-trained knowledge, which includes
# general information at the time of its training.
# - How the system prompt specifying the persona affected the agent's response.
# The system prompt directs the agent to structure its response in a specific way,
# following the persona it was assigned to.
