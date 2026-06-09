# agentic_workflow.py
import os
from workflow_agents.base_agents import (
    ActionPlanningAgent,
    KnowledgeAugmentedPromptAgent,
    EvaluationAgent,
    RoutingAgent,
)

# Load the OpenAI key into a variable called openai_api_key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load the product spec
with open("Product-Spec-Email-Router.txt", "r") as file:
    product_spec = file.read()
    file.close()

# Instantiate all the agents

# Action Planning Agent
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components"
)
action_planning_agent = ActionPlanningAgent(
    openai_api_key=openai_api_key,
    knowledge=knowledge_action_planning,
)

# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = f"""
Stories are defined by writing sentences with a persona, an action, and a desired outcome.
The sentences always start with: As a [type of user], I want [an action or feature] so that [benefit/value].
Write several stories for the product spec below, where the personas are the different users of the product.

# CONTEXT
{product_spec}
"""
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_product_manager,
    knowledge=knowledge_product_manager,
)

# Product Manager - Evaluation Agent
persona_product_manager_eval = (
    "You are an evaluation agent that checks the answers of other worker agents."
)
product_manager_eval_criteria = (
    "The answer should be stories that follow the following structure: As a [type of user], I want [an action or feature] so that [benefit/value]."
    "The answer must not be the how-to, but the actual stories that need to be built."
)
product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_product_manager_eval,
    evaluation_criteria=product_manager_eval_criteria,
    worker_agent=product_manager_knowledge_agent,
    max_interactions=10,
)

# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = f"""
# OUTPUT FORMAT
Feature Name: <A clear, concise title that identifies the capability>
Description: <A brief explanation of what the feature does and its purpose>
Key Functionality: <The specific capabilities or actions the feature provides>
User Benefit: <How this feature creates value for the user>

# CONTEXT
Features of a product are defined by organizing similar user stories into cohesive groups.
"""
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_program_manager,
    knowledge=knowledge_program_manager,
)

# Program Manager - Evaluation Agent
persona_program_manager_eval = (
    "You are an evaluation agent that checks the answers of other worker agents."
)
program_manager_eval_criteria = (
    "The answer should be product features that follow the following structure: "
    "Feature Name: <A clear, concise title that identifies the capability>\n"
    "Description: <A brief explanation of what the feature does and its purpose>\n"
    "Key Functionality: <The specific capabilities or actions the feature provides>\n"
    "User Benefit: <How this feature creates value for the user>\n"
    "The answer must not be the how-to, but the actual features that need to be built."
)
program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_program_manager_eval,
    evaluation_criteria=program_manager_eval_criteria,
    worker_agent=program_manager_knowledge_agent,
    max_interactions=10,
)

# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = f"""
# OUTPUT FORMAT
Task ID: <A unique identifier for tracking purposes>
Task Title: <Brief description of the specific development work>
Related User Story: <Reference to the parent user story>
Description: <Detailed explanation of the technical work required>
Acceptance Criteria: <Specific requirements that must be met for completion>
Estimated Effort: <Time or complexity estimation>
Dependencies: <Any tasks that must be completed first>

# CONTEXT
Development tasks are defined by identifying what needs to be built to implement each user story.
"""
dev_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer,
    knowledge=knowledge_dev_engineer,
)

# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = (
    "You are an evaluation agent that checks the answers of other worker agents."
)
dev_engineer_eval_criteria = (
    "The answer should be tasks following this exact structure: "
    "Task ID: <A unique identifier for tracking purposes>\n"
    "Task Title: <Brief description of the specific development work>\n"
    "Related User Story: <Reference to the parent user story>\n"
    "Description: <Detailed explanation of the technical work required>\n"
    "Acceptance Criteria: <Specific requirements that must be met for completion>\n"
    "Estimated Effort: <Time or complexity estimation>\n"
    "Dependencies: <Any tasks that must be completed first>\n"
    "The answer must not be the how-to, but the actual tasks that need to be done."
)
dev_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer_eval,
    evaluation_criteria=dev_engineer_eval_criteria,
    worker_agent=dev_engineer_knowledge_agent,
    max_interactions=10,
)

# Routing Agent
routing_agent = RoutingAgent(openai_api_key=openai_api_key, agents=[])
agents = [
    {
        "name": "Product Manager",
        "description": f"Responsible for defining product personas and user stories only. Does not define features or tasks. Does not group stories.\n\n{persona_product_manager}",
        "func": lambda x: product_manager_support_function(x),
    },
    {
        "name": "Program Manager",
        "description": f"Responsible for defining product features by grouping related user stories. Does not define personas, user stories, or tasks.\n\n{persona_program_manager}",
        "func": lambda x: program_manager_support_function(x),
    },
    {
        "name": "Development Engineer",
        "description": f"Responsible for defining development tasks based on the user stories. Does not define personas, user stories, or features.\n\n{persona_dev_engineer}",
        "func": lambda x: dev_engineer_support_function(x),
    },
]
routing_agent.agents = agents


# Job function persona support functions
def product_manager_support_function(query):
    # Get response from the evaluation agent
    # which will in turn get responses from the knowledge agent as needed
    evaluation = product_manager_evaluation_agent.evaluate(query)
    final_response = evaluation["final_response"]

    return final_response


def program_manager_support_function(query):
    # Get response from the evaluation agent
    # which will in turn get responses from the knowledge agent as needed
    evaluation = program_manager_evaluation_agent.evaluate(query)
    final_response = evaluation["final_response"]

    return final_response


def dev_engineer_support_function(query):
    # Get response from the evaluation agent
    # which will in turn get responses from the knowledge agent as needed
    evaluation = dev_engineer_evaluation_agent.evaluate(query)
    final_response = evaluation["final_response"]

    return final_response


# Run the workflow

print("\n*** Workflow execution started ***\n")
# Workflow Prompt
# ****
workflow_prompt = "What would the development tasks for this product be?"
# ****
print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

print("\nDefining workflow steps from the workflow prompt")
# Implement the workflow.
# 1. Use the 'action_planning_agent' to extract steps from the 'workflow_prompt'.
action_plan = action_planning_agent.extract_steps_from_prompt(workflow_prompt)

# 2. Initialize an empty list to store 'completed_steps'.
completed_steps = []
last_step_result = ""

# 3. Loop through the extracted workflow steps:
for step in action_plan:
    print(f"\n=== Executing step: {step} ===\n")
    print("Routing the step to the appropriate agent...")
    # a. For each step, use the 'routing_agent' to route the step to the appropriate support function.
    step_prompt = f"""
    {step}

    # Last step result:
    {last_step_result}
    """
    step_result = routing_agent.route(step_prompt)
    # b. Append the result to 'completed_steps'.
    completed_steps.append({"step": step, "result": step_result})
    last_step_result = step_result
    # c. Print information about the step being executed and its result.
    print(f"Step result: {step_result}")

# 4. After the loop, print the final output of the workflow (the last completed step).
if completed_steps:
    final_output = completed_steps[-1]["result"]
    print("\n*** Workflow execution completed ***\n")
    print(f"Final output of the workflow:\n{final_output}")
else:
    print("\n*** Workflow execution completed ***\n")
    print("No steps were completed in the workflow.")
