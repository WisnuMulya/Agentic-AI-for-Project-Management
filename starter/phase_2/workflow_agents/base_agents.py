from email.mime import base

from openai import OpenAI
import numpy as np
import pandas as pd
import re
import csv
import uuid
from datetime import datetime


# DirectPromptAgent class definition
class DirectPromptAgent:
    """
    A simple agent answering user prompts directly without incorporating additional context,
    memory, or specialised tools.
    """

    def __init__(self, openai_api_key):
        """
        Initializes the DirectPromptAgent with API credentials.
        """
        self.openai_api_key = openai_api_key

    def respond(self, prompt):
        """
        Generates a response to the given prompt using the OpenAI API.

        Parameters:
        prompt (str): The user input prompt to which the agent should respond.

        Returns:
        str: The generated response from the agent.
        """
        client = OpenAI(
            base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        return response.choices[0].message.content


# AugmentedPromptAgent class definition
class AugmentedPromptAgent:
    """
    An agent that responds according to its specified persona.
    """

    def __init__(self, openai_api_key, persona):
        """
        Initializes the AugmentedPromptAgent with API credentials and a defined persona.

        Parameters:
        openai_api_key (str): API key for accessing OpenAI.
        persona (str): A description of the agent's persona that will influence its responses.
        """
        self.openai_api_key = openai_api_key
        self.persona = persona

    def respond(self, input_text):
        """
        Generate a response using OpenAI API.

        Parameters:
        input_text (str): The input text or prompt for the agent.

        Returns:
        str: The generated response from the agent.
        """
        client = OpenAI(
            base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key
        )

        # A system prompt assuming the defined persona and forgetting any previous context
        system_prompt = "Forget all previous context.\n" + self.persona
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text},
            ],
            temperature=0.7,
        )

        return response.choices[0].message.content


# KnowledgeAugmentedPromptAgent class definition
class KnowledgeAugmentedPromptAgent:
    """
    An agent that responds to user prompts solely based on its persona and
    provided knowledge.
    """

    def __init__(self, openai_api_key, persona, knowledge):
        """
        Initializes the KnowledgeAugmentedPromptAgent with API credentials, persona, and knowledge.

        Parameters:
        openai_api_key (str): API key for accessing OpenAI.
        persona (str): A description of the agent's persona that will influence its responses.
        knowledge (str): The knowledge base that the agent will use to generate responses.
        """
        self.persona = persona
        self.knowledge = knowledge
        self.openai_api_key = openai_api_key

    def respond(self, input_text):
        """
        Generate a response using the OpenAI API.

        Parameters:
        input_text (str): The input text or prompt for the agent.

        Returns:
        str: The generated response from the agent.
        """
        client = OpenAI(
            base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key
        )
        system_prompt = f"""
        You are a knowledge-based assistant. Forget all previous context.

        # Persona
        {self.persona}

        # Knowledge
        Use only the following knowledge to answer, do not use your own knowledge:
        {self.knowledge}

        Answer the prompt based on this knowledge, not your own.
        """
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text},
            ],
            temperature=0,
        )
        return response.choices[0].message.content


# RAGKnowledgePromptAgent class definition
class RAGKnowledgePromptAgent:
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge from a large corpus
    and leverages embeddings to respond to prompts based solely on retrieved information.
    """

    def __init__(self, openai_api_key, persona, chunk_size=2000, chunk_overlap=100):
        """
        Initializes the RAGKnowledgePromptAgent with API credentials and configuration settings.

        Parameters:
        openai_api_key (str): API key for accessing OpenAI.
        persona (str): Persona description for the agent.
        chunk_size (int): The size of text chunks for embedding. Defaults to 2000.
        chunk_overlap (int): Overlap between consecutive chunks. Defaults to 100.
        """
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.unique_filename = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"
        )

    def get_embedding(self, text):
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.

        Parameters:
        text (str): Text to embed.

        Returns:
        list: The embedding vector.
        """
        client = OpenAI(
            base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key
        )
        response = client.embeddings.create(
            model="text-embedding-3-large", input=text, encoding_format="float"
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two):
        """
        Calculates cosine similarity between two vectors.

        Parameters:
        vector_one (list): First embedding vector.
        vector_two (list): Second embedding vector.

        Returns:
        float: Cosine similarity between vectors.
        """
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def chunk_text(self, text):
        """
        Splits text into manageable chunks, attempting natural breaks.

        Parameters:
        text (str): Text to split into chunks.

        Returns:
        list: List of dictionaries containing chunk metadata.
        """
        separator = "\n"
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) <= self.chunk_size:
            return [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]

        chunks, start, chunk_id = [], 0, 0

        while start < len(text) - self.chunk_overlap:
            end = min(start + self.chunk_size, len(text))
            if separator in text[start:end]:
                end = start + text[start:end].rindex(separator) + len(separator)

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": text[start:end],
                    "chunk_size": end - start,
                    "start_char": start,
                    "end_char": end,
                }
            )

            start = end - self.chunk_overlap
            chunk_id += 1

        with open(
            f"./chunks-{self.unique_filename}", "w", newline="", encoding="utf-8"
        ) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
            writer.writeheader()
            for chunk in chunks:
                writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})

        return chunks

    def calculate_embeddings(self):
        """
        Calculates embeddings for each chunk and stores them in a CSV file.

        Returns:
        DataFrame: DataFrame containing text chunks and their embeddings.
        """
        df = pd.read_csv(f"./chunks-{self.unique_filename}", encoding="utf-8")
        df["embeddings"] = df["text"].apply(self.get_embedding)
        df.to_csv(f"embeddings-{self.unique_filename}", encoding="utf-8", index=False)
        return df

    def find_prompt_in_knowledge(self, prompt):
        """
        Finds and responds to a prompt based on similarity with embedded knowledge.

        Parameters:
        prompt (str): User input prompt.

        Returns:
        str: Response derived from the most similar chunk in knowledge.
        """
        prompt_embedding = self.get_embedding(prompt)
        df = pd.read_csv(f"./embeddings-{self.unique_filename}", encoding="utf-8")
        df["embeddings"] = df["embeddings"].apply(lambda x: np.array(eval(x)))
        df["similarity"] = df["embeddings"].apply(
            lambda emb: self.calculate_similarity(prompt_embedding, emb)
        )

        best_chunk = df.loc[df["similarity"].idxmax(), "text"]

        client = OpenAI(
            base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"You are {self.persona}, a knowledge-based assistant. Forget previous context.",
                },
                {
                    "role": "user",
                    "content": f"Answer based only on this information: {best_chunk}. Prompt: {prompt}",
                },
            ],
            temperature=0,
        )

        return response.choices[0].message.content


class EvaluationAgent:
    """
    An agent that assesses responses from another agent against a given set of
    evaluation criteria.
    """

    def __init__(
        self,
        openai_api_key,
        persona,
        evaluation_criteria,
        worker_agent,
        max_interactions,
    ):
        """
        Initializes the EvaluationAgent with the given attributes.

        Parameters:
        openai_api_key (str): API key for OpenAI.
        persona (str): Persona description for the agent.
        evaluation_criteria (str): Criteria for evaluating responses.
        worker_agent (object): The agent whose responses are being evaluated.
        max_interactions (int): Maximum number of interactions allowed.
        """
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions

    def evaluate(self, initial_prompt):
        """
        Evaluates the response from the worker agent against the evaluation criteria.

        Parameters:
        initial_prompt (str): The initial prompt to be evaluated.

        Returns:
        dict: A dictionary containing the final response, evaluation, and number of iterations.
        """
        client = OpenAI(
            base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key
        )
        prompt_to_evaluate = initial_prompt

        for i in range(self.max_interactions):
            print(f"\n--- Interaction {i+1} ---")

            print(" Step 1: Worker agent generates a response to the prompt")
            print(f"Prompt:\n{prompt_to_evaluate}")
            response_from_worker = self.worker_agent.respond(prompt_to_evaluate)
            print(f"Worker Agent Response:\n{response_from_worker}")

            print(" Step 2: Evaluator agent judges the response")
            eval_prompt = (
                f"Does the following answer: {response_from_worker}\n"
                f"Meet this criteria: {self.evaluation_criteria}\n"
                f"Respond Yes or No, and the reason why it does or doesn't meet the criteria."
            )
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.persona},
                    {"role": "user", "content": eval_prompt},
                ],
                temperature=0,
            )
            evaluation = response.choices[0].message.content.strip()
            print(f"Evaluator Agent Evaluation:\n{evaluation}")

            print(" Step 3: Check if evaluation is positive")
            if evaluation.lower().startswith("yes"):
                print("✅ Final solution accepted.")
                break
            else:
                print(" Step 4: Generate instructions to correct the response")
                instruction_prompt = f"Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation}"
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": self.persona},
                        {"role": "user", "content": instruction_prompt},
                    ],
                    temperature=0,
                )
                instructions = response.choices[0].message.content.strip()
                print(f"Instructions to fix:\n{instructions}")

                print(" Step 5: Send feedback to worker agent for refinement")
                prompt_to_evaluate = (
                    f"The original prompt was: {initial_prompt}\n"
                    f"The response to that prompt was: {response_from_worker}\n"
                    f"It has been evaluated as incorrect.\n"
                    f"Make only these corrections, do not alter content validity: {instructions}"
                )
        return {
            "final_response": response_from_worker,
            "evaluation": evaluation,
            "iterations": i + 1,
        }


class RoutingAgent:
    """
    An agent that routes user prompts to the most appropriate agent based on the similarity.
    """

    def __init__(self, openai_api_key, agents):
        """
        Initializes the RoutingAgent with API credentials and a list of agents to route to.

        Parameters:
        openai_api_key (str): API key for OpenAI.
        agents (list): A list of dictionaries, each containing 'name', 'description', and 'func' for an agent.
        """
        self.openai_api_key = openai_api_key
        self.agents = agents

    def get_embedding(self, text):
        """
        Calculates the embedding for the given text.

        Parameters:
        text (str): The text to be embedded

        Returns:
        list: The embedding vector
        """
        client = OpenAI(
            base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key
        )

        # Extract and return the embedding vector from the response
        response = client.embeddings.create(
            model="text-embedding-3-large", input=text, encoding_format="float"
        )
        embedding = response.data[0].embedding
        return embedding

    def route(self, prompt):
        """
        Routes the user prompt to the most appropriate agent based on the similarity.

        Parameters:
        prompt (str): The user input prompt to be routed.

        Returns:
        str: The response from the selected agent.
        """
        # Compute the embedding of the user input prompt
        input_emb = self.get_embedding(prompt)
        best_agent = None
        best_score = -1

        for agent in self.agents:
            # Compute the embedding of the agent description
            agent_emb = self.get_embedding(agent["description"])
            if agent_emb is None:
                continue

            similarity = np.dot(input_emb, agent_emb) / (
                np.linalg.norm(input_emb) * np.linalg.norm(agent_emb)
            )
            print(similarity)

            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")
        return best_agent["func"](prompt)


class ActionPlanningAgent:
    """
    An agent that extract and list the steps required to execute a task based on user prompts.
    """

    def __init__(self, openai_api_key, knowledge):
        """
        Initializes the ActionPlanningAgent with API credentials and a knowledge base.

        Parameters:
        openai_api_key (str): API key for OpenAI
        knowledge (str): The knowledge base
        """
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge

    def extract_steps_from_prompt(self, prompt):
        """
        Extracts and lists the steps required to execute a task based on the user prompt.

        Parameters:
        prompt (str): The user input prompt describing the task.

        Returns:
        list: A list of steps required to execute the task.
        """
        client = OpenAI(
            base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key
        )

        system_prompt = f"""
        You are an action planning agent.
        Using your knowledge, you extract from the user prompt the steps requested to complete the action the user is asking for.
        You return the steps as a list with no words other than the steps themselves. Only return the steps in your knowledge.
        Forget any previous context.
        
        # OUTPUT FORMAT
        1. <Step 1>
        2. <Step 2>
        3. <Step 3>
        ...

        # KNOWLEDGE
        {self.knowledge}
        """
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

        response_text = response.choices[0].message.content.strip()

        # Clean and format the extracted steps by removing empty lines and unwanted text
        steps = response_text.split("\n")

        return steps
