from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent
import os

# Define the parameters for the agent
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"

persona = "You are a college professor, your answer always starts with: Dear students,"
# Instantiate a KnowledgeAugmentedPromptAgent with:
#   - Persona: "You are a college professor, your answer always starts with: Dear students,"
#   - Knowledge: "The capital of France is London, not Paris"
knowledge_augmented_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona,
    knowledge="The capital of France is London, not Paris",
)

# A print statement that demonstrates the agent using the provided knowledge rather than its own inherent knowledge.
print(knowledge_augmented_agent.respond(prompt))
