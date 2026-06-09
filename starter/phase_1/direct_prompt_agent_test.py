# Test script for DirectPromptAgent class
from workflow_agents.base_agents import DirectPromptAgent
import os

# Load the OpenAI API key from the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the Capital of France?"

# Instantiate the DirectPromptAgent as direct_agent
direct_agent = DirectPromptAgent(openai_api_key=openai_api_key)

# Use direct_agent to send the prompt defined above and store the response
direct_agent_response = direct_agent.respond(prompt)

# Print the response from the agent
print(direct_agent_response)

# Print an explanatory message describing the knowledge source used by the agent to generate the response
print(
    "The DirectPromptAgent answers the user's prompt directly based on the selected LLM's pre-trained knowledge."
)
