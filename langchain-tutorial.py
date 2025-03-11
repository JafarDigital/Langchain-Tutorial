#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LangChain Basics Tutorial
=========================

This tutorial covers the basic features of LangChain, a framework for developing applications
powered by language models. It includes examples for:

1. Basic LLM Interactions
2. Prompts and Templates
3. Chains
4. Memory
5. Agents
6. Document Loaders and Text Splitting
7. Vector Stores and Retrieval
8. Output Parsing

Requirements:
- Python 3.8+
- langchain
- openai (or another LLM provider)
- chromadb (for vector storage)
"""

# First, you'll need to install langchain and related packages:
# pip install langchain openai chromadb

import os
from typing import List, Dict, Any

# Set your API keys in environment variables (best practice)
# os.environ["OPENAI_API_KEY"] = "your-api-key"

###############################################################################
# SECTION 1: BASIC LLM INTERACTIONS
###############################################################################

"""
LangChain provides a consistent interface to various Language Model providers.
This allows you to easily switch between different models.
"""

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# Example 1: Using a completion model (text-in, text-out)
def basic_llm_example():
    """
    This function demonstrates how to make a simple call to an LLM.
    """
    # Initialize the LLM with a specific model and temperature
    # Temperature controls randomness (0.0 = deterministic, 1.0 = creative)
    llm = OpenAI(model_name="text-davinci-003", temperature=0.7)
    
    # Make a simple completion request
    response = llm("Explain quantum computing in one paragraph")
    
    print("LLM Response:", response)
    
    # You can also pass parameters for each invocation
    technical_response = llm("Explain quantum computing in one paragraph", temperature=0.2)
    creative_response = llm("Explain quantum computing in one paragraph", temperature=0.9)
    
    return response

# Example 2: Using a chat model (messages-in, message-out)
def basic_chat_model_example():
    """
    This function demonstrates how to use chat models with LangChain.
    Chat models take a series of messages and return a message.
    """
    from langchain.schema import HumanMessage, SystemMessage, AIMessage
    
    # Initialize a chat model
    chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    
    # Creating messages - chat models work with sequences of messages
    messages = [
        SystemMessage(content="You are a helpful assistant that specializes in quantum physics."),
        HumanMessage(content="Explain quantum entanglement in simple terms.")
    ]
    
    # Get response from the chat model
    response = chat_model(messages)
    
    print("Chat Model Response:", response.content)
    
    # You can continue the conversation by adding more messages
    messages.append(AIMessage(content=response.content))
    messages.append(HumanMessage(content="How is this used in quantum computers?"))
    
    follow_up_response = chat_model(messages)
    print("Follow-up Response:", follow_up_response.content)
    
    return follow_up_response.content

###############################################################################
# SECTION 2: PROMPTS AND TEMPLATES
###############################################################################

"""
LangChain provides tools for creating, formatting, and reusing prompts.
This makes your prompts more maintainable and flexible.
"""

def prompt_templates_example():
    """
    This function demonstrates LangChain's prompt templates.
    """
    from langchain.prompts import PromptTemplate, ChatPromptTemplate
    from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
    
    # Example 1: Basic prompt template
    # Create a template with variables
    template = "Write a {length} summary about {topic}."
    
    # Create a PromptTemplate with the template string and input variables
    prompt_template = PromptTemplate(
        input_variables=["length", "topic"],
        template=template
    )
    
    # Format the prompt with specific values
    prompt = prompt_template.format(length="short", topic="artificial intelligence")
    print("Formatted Prompt:", prompt)
    
    # Example 2: Chat prompt template (for chat models)
    system_template = "You are an expert on {subject}."
    human_template = "Please explain {concept} in simple terms."
    
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    chat_prompt = ChatPromptTemplate.from_messages([
        system_message_prompt,
        human_message_prompt
    ])
    
    # Format the chat prompt
    messages = chat_prompt.format_messages(
        subject="quantum physics",
        concept="quantum entanglement"
    )
    
    print("Chat Messages:")
    for message in messages:
        print(f"{message.type}: {message.content}")
        
    # Example 3: Few-shot prompting (providing examples)
    few_shot_template = """
    Answer the question based on the context.
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
    
    few_shot_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=few_shot_template
    )
    
    return {
        "basic_prompt": prompt,
        "chat_messages": messages,
    }

###############################################################################
# SECTION 3: CHAINS
###############################################################################

"""
Chains allow you to combine multiple components (like models and prompts)
together into a single, coherent application.
"""

def chains_example():
    """
    This function demonstrates different types of chains in LangChain.
    """
    from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
    from langchain.prompts import PromptTemplate
    from langchain.llms import OpenAI
    
    llm = OpenAI(temperature=0.7)
    
    # Example 1: Basic LLMChain
    # Create a prompt template
    template = "Generate a catchy title for an article about {topic}."
    prompt_template = PromptTemplate(input_variables=["topic"], template=template)
    
    # Create an LLMChain
    title_chain = LLMChain(llm=llm, prompt=prompt_template)
    
    # Run the chain
    title = title_chain.run("renewable energy")
    print("Generated Title:", title)
    
    # Example 2: SimpleSequentialChain (single input/output)
    # First chain generates a title
    title_template = "Generate a catchy title for an article about {topic}."
    title_prompt = PromptTemplate(input_variables=["topic"], template=title_template)
    title_chain = LLMChain(llm=llm, prompt=title_prompt)
    
    # Second chain generates an outline based on the title
    outline_template = "Generate a 3-point outline for an article titled '{title}'."
    outline_prompt = PromptTemplate(input_variables=["title"], template=outline_template)
    outline_chain = LLMChain(llm=llm, prompt=outline_prompt)
    
    # Combine the chains
    sequential_chain = SimpleSequentialChain(chains=[title_chain, outline_chain])
    
    # Run the sequential chain
    outline = sequential_chain.run("artificial intelligence ethics")
    print("Generated Outline:", outline)
    
    # Example 3: SequentialChain (multiple inputs/outputs)
    # First chain: generates a title
    title_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            input_variables=["topic"],
            template="Generate a catchy title for an article about {topic}."
        ),
        output_key="title"  # This will be available to the next chain
    )
    
    # Second chain: generates an introduction
    intro_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            input_variables=["title", "audience"],
            template="Write an introduction for an article titled '{title}' for a {audience} audience."
        ),
        output_key="introduction"
    )
    
    # Combine the chains with multiple inputs/outputs
    multi_chain = SequentialChain(
        chains=[title_chain, intro_chain],
        input_variables=["topic", "audience"],
        output_variables=["title", "introduction"]
    )
    
    # Run the multi-input/output chain
    result = multi_chain({"topic": "machine learning", "audience": "technical"})
    print("Title:", result["title"])
    print("Introduction:", result["introduction"])
    
    return result

###############################################################################
# SECTION 4: MEMORY
###############################################################################

"""
Memory components allow chains to retain information across multiple calls,
enabling stateful conversations.
"""

def memory_example():
    """
    This function demonstrates different memory types in LangChain.
    """
    from langchain.chains import ConversationChain
    from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
    from langchain.memory import ConversationSummaryMemory, ConversationEntityMemory
    from langchain.llms import OpenAI
    
    llm = OpenAI(temperature=0.7)
    
    # Example 1: ConversationBufferMemory
    # Simplest form - stores messages in a buffer
    buffer_memory = ConversationBufferMemory()
    
    # Add messages to memory
    buffer_memory.chat_memory.add_user_message("Hi, my name is Bob.")
    buffer_memory.chat_memory.add_ai_message("Hello Bob! How can I help you today?")
    
    # Create a conversation chain with memory
    conversation = ConversationChain(
        llm=llm,
        memory=buffer_memory,
        verbose=True  # Set to True to see the prompts being passed
    )
    
    # Continue the conversation
    response = conversation.predict(input="I'm interested in learning about AI.")
    print("Response:", response)
    
    # Example 2: ConversationBufferWindowMemory
    # Only keeps the last k turns in memory
    window_memory = ConversationBufferWindowMemory(k=2)
    window_memory.chat_memory.add_user_message("Hi, my name is Alice.")
    window_memory.chat_memory.add_ai_message("Hello Alice! How can I help you today?")
    window_memory.chat_memory.add_user_message("Tell me about machine learning.")
    window_memory.chat_memory.add_ai_message("Machine learning is a branch of AI...")
    
    # With k=2, only the last 2 interactions are kept
    print("Window Memory Variables:", window_memory.load_memory_variables({}))
    
    # Example 3: ConversationSummaryMemory
    # Summarizes the conversation over time
    summary_memory = ConversationSummaryMemory(llm=llm)
    summary_memory.chat_memory.add_user_message("Hi, I'm Dave.")
    summary_memory.chat_memory.add_ai_message("Hello Dave! How can I help?")
    summary_memory.chat_memory.add_user_message("I want to learn about deep learning.")
    summary_memory.chat_memory.add_ai_message("Deep learning is a subset of machine learning...")
    
    # The memory will be summarized rather than stored verbatim
    print("Summary Memory:", summary_memory.load_memory_variables({}))
    
    # Example 4: ConversationEntityMemory
    # Keeps track of entities mentioned in the conversation
    entity_memory = ConversationEntityMemory(llm=llm)
    
    conversation_with_entities = ConversationChain(
        llm=llm,
        memory=entity_memory,
        verbose=True
    )
    
    # Entity memory tracks information about specific entities mentioned
    response = conversation_with_entities.predict(
        input="My favorite city is Paris and I love French cuisine."
    )
    print("Entities tracked:", entity_memory.entity_store.keys())
    
    return {
        "buffer_response": response,
        "window_memory": window_memory.load_memory_variables({}),
        "summary_memory": summary_memory.load_memory_variables({}),
        "entity_memory": list(entity_memory.entity_store.keys())
    }

###############################################################################
# SECTION 5: AGENTS
###############################################################################

"""
Agents use LLMs to determine which actions to take and in what order.
They can use tools/functions and make decisions about which ones to use based
on user input.
"""

def agents_example():
    """
    This function demonstrates how to use agents in LangChain.
    """
    from langchain.agents import load_tools, initialize_agent, AgentType
    from langchain.llms import OpenAI
    
    # Initialize the LLM to use for the agent
    llm = OpenAI(temperature=0)
    
    # Example 1: Load some basic tools for the agent to use
    # Available tools include: 'serpapi', 'wolfram-alpha', 'requests', 'terminal' etc.
    # For this example, we'll use simple calculator tools that don't require external API keys
    tools = load_tools(["llm-math"], llm=llm)
    
    # Initialize the agent with the tools and LLM
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # This agent uses the ReAct framework
        verbose=True  # Set to True to see the agent's thought process
    )
    
    # Let's ask the agent a question that requires calculation
    # response = agent.run("What is the square root of 1369 multiplied by 15?")
    # print("Agent Response:", response)
    
    # Example 2: Creating custom tools
    from langchain.tools import BaseTool, StructuredTool, tool
    
    # Method 1: Using the @tool decorator (simplest)
    @tool
    def get_weather(location: str) -> str:
        """Get the current weather in a given location."""
        # In a real scenario, you'd call a weather API here
        return f"The weather in {location} is currently sunny and 75 degrees."
    
    # Method 2: Creating a custom tool class
    class SearchTool(BaseTool):
        name = "search_tool"
        description = "Useful for searching information about a topic"
        
        def _run(self, query: str) -> str:
            # In a real scenario, you'd call a search API here
            return f"Here are the search results for '{query}'..."
        
        def _arun(self, query: str):
            # For async implementation
            raise NotImplementedError("This tool does not support async")
    
    # Create a new agent with these custom tools
    custom_tools = [get_weather, SearchTool()]
    custom_agent = initialize_agent(
        custom_tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    # Use the agent with custom tools
    # response = custom_agent.run("What's the weather like in New York?")
    # print("Custom Tool Response:", response)
    
    # Example 3: Using the high-level API for structured tools
    from langchain.agents import create_structured_chat_agent
    from langchain.schema import SystemMessage
    from langchain.chat_models import ChatOpenAI
    
    chat_model = ChatOpenAI(temperature=0)
    
    # Define system message for the agent
    system_message = SystemMessage(
        content="You are a helpful assistant that can use tools to answer questions."
    )
    
    # Create a structured agent with the tools and chat model
    structured_agent = create_structured_chat_agent(
        llm=chat_model,
        tools=custom_tools,
        system_message=system_message
    )
    
    # Create an agent executor with the structured agent
    # from langchain.agents import AgentExecutor
    # agent_executor = AgentExecutor(
    #     agent=structured_agent,
    #     tools=custom_tools,
    #     verbose=True
    # )
    
    # Run the agent executor
    # response = agent_executor.invoke({"input": "What's the weather in Miami?"})
    # print("Structured Agent Response:", response)
    
    return {
        "available_tools": [tool.name for tool in custom_tools],
        "agent_description": "Agent with custom weather and search tools"
    }

###############################################################################
# SECTION 6: DOCUMENT LOADERS AND TEXT SPLITTING
###############################################################################

"""
LangChain provides utilities for loading documents from various sources
and splitting them into manageable chunks for processing.
"""

def document_loaders_example():
    """
    This function demonstrates document loaders and text splitters in LangChain.
    """
    from langchain.document_loaders import TextLoader, CSVLoader
    from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
    import tempfile
    import os
    
    # Example 1: Loading and splitting a text document
    # Create a temporary text file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as temp:
        temp.write(b"""
        LangChain is a framework for developing applications powered by language models.
        It enables applications that:
        - Are context-aware: connect a language model to sources of context
        - Reason: rely on a language model to reason
        
        The main value props of LangChain are:
        1. Components: abstractions for working with language models, along with a collection of implementations
        2. Off-the-shelf chains: for accomplishing specific higher-level tasks
        
        These modules can be used independently, but they are designed to work well together.
        """)
        temp_path = temp.name
    
    # Load the document
    loader = TextLoader(temp_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s)")
    print(f"Document content preview: {documents[0].page_content[:100]}...")
    
    # Example 2: Splitting text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=200,
        chunk_overlap=50
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    
    # Example 3: Using RecursiveCharacterTextSplitter (more advanced)
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=20,
        separators=["\n\n", "\n", ". ", " ", ""]  # Try these separators in order
    )
    
    recursive_chunks = recursive_splitter.split_documents(documents)
    print(f"Recursively split into {len(recursive_chunks)} chunks")
    
    # Example 4: Loading a CSV file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_csv:
        temp_csv.write(b"""
        id,name,age,occupation
        1,Alice,28,Data Scientist
        2,Bob,34,Software Engineer
        3,Charlie,45,Product Manager
        """)
        csv_path = temp_csv.name
    
    csv_loader = CSVLoader(file_path=csv_path)
    csv_data = csv_loader.load()
    print(f"Loaded {len(csv_data)} rows from CSV")
    print(f"First row: {csv_data[0].page_content}")
    
    # Clean up temp files
    os.unlink(temp_path)
    os.unlink(csv_path)
    
    # Example 5: Other available loaders (not executed)
    """
    LangChain includes many other loaders:
    - PDFLoader, PyPDFLoader, PyMuPDFLoader for PDF files
    - WebBaseLoader for web pages
    - YoutubeLoader for YouTube transcripts
    - NotionDirectoryLoader for Notion exports
    - SlackDirectoryLoader for Slack exports
    - And many more!
    """
    
    return {
        "num_chunks": len(chunks),
        "num_recursive_chunks": len(recursive_chunks),
        "csv_rows": len(csv_data)
    }

###############################################################################
# SECTION 7: VECTOR STORES AND RETRIEVAL
###############################################################################

"""
Vector stores allow storage and retrieval of documents based on their 
embeddings (vector representations). This enables semantic search and 
retrieval of relevant documents.
"""

def vector_stores_example():
    """
    This function demonstrates vector stores and retrievers in LangChain.
    """
    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Chroma, FAISS
    import tempfile
    import os
    
    # Example 1: Create sample documents
    docs = [
        "LangChain provides many modules that can be used to build language model applications.",
        "Chains are a core component of LangChain, allowing multiple components to be combined.",
        "Agents use LLMs to determine which actions to take and in what order.",
        "Tools are functions that agents can use to interact with the world.",
        "Vector stores are used to store and retrieve documents based on their embeddings."
    ]
    
    # Create documents
    documents = [{"page_content": doc, "metadata": {"source": f"doc_{i}"}} for i, doc in enumerate(docs)]
    documents = [Document(page_content=doc["page_content"], metadata=doc["metadata"]) for doc in documents]
    
    # Example 2: Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)
    
    # Example 3: Create embeddings
    # In a real scenario, you would use:
    # embeddings = OpenAIEmbeddings()
    
    # For this example, we'll create a simple mock embeddings model
    from langchain.embeddings.base import Embeddings
    import numpy as np
    
    class MockEmbeddings(Embeddings):
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            # Return a simple mock embedding for each text
            return [self._get_mock_embedding(text) for text in texts]
        
        def embed_query(self, text: str) -> List[float]:
            # Return a simple mock embedding for the query
            return self._get_mock_embedding(text)
        
        def _get_mock_embedding(self, text: str) -> List[float]:
            # Create a deterministic but simple mock embedding
            # In a real scenario, you'd use a proper embedding model
            np.random.seed(sum(ord(c) for c in text))
            return list(np.random.rand(768))  # 768-dim embedding
    
    embeddings = MockEmbeddings()
    
    # Example 4: Create and use a Chroma vector store
    # Chroma is an open-source embedding database
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings
    )
    
    # Example 5: Similarity search
    query = "What are chains in LangChain?"
    similar_docs = vector_store.similarity_search(query, k=2)
    print("Similar documents:")
    for doc in similar_docs:
        print(f"- {doc.page_content} (Source: {doc.metadata['source']})")
    
    # Example 6: Create a retriever from the vector store
    retriever = vector_store.as_retriever(
        search_type="similarity",  # Options: similarity, mmr
        search_kwargs={"k": 2}
    )
    
    retrieved_docs = retriever.get_relevant_documents(query)
    print("\nRetrieved documents:")
    for doc in retrieved_docs:
        print(f"- {doc.page_content}")
    
    # Example 7: Using FAISS vector store (Facebook AI Similarity Search)
    # FAISS is a library for efficient similarity search
    faiss_vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )
    
    faiss_results = faiss_vector_store.similarity_search(query, k=2)
    
    # Example 8: Saving and loading a vector store
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the FAISS index to disk
        faiss_vector_store.save_local(temp_dir)
        
        # Load the FAISS index from disk
        loaded_vector_store = FAISS.load_local(
            temp_dir,
            embeddings
        )
        
        # Verify it works
        loaded_results = loaded_vector_store.similarity_search(query, k=2)
        print(f"\nLoaded vector store returns {len(loaded_results)} results")
    
    from langchain.schema import Document
    
    return {
        "num_documents": len(documents),
        "num_similar_docs": len(similar_docs),
        "vector_stores": ["Chroma", "FAISS"],
    }

###############################################################################
# SECTION 8: OUTPUT PARSING
###############################################################################

"""
Output parsers help structure the output from LLMs into usable formats
like dictionaries, lists, or custom objects.
"""

def output_parsing_example():
    """
    This function demonstrates output parsers in LangChain.
    """
    from langchain.output_parsers import CommaSeparatedListOutputParser
    from langchain.output_parsers import StructuredOutputParser, ResponseSchema
    from langchain.prompts import PromptTemplate
    from langchain.llms import OpenAI
    
    llm = OpenAI(temperature=0)
    
    # Example 1: Comma-separated list parser
    list_parser = CommaSeparatedListOutputParser()
    
    # Format instructions for the model
    format_instructions = list_parser.get_format_instructions()
    
    # Create a prompt template that includes the format instructions
    list_template = """
    Generate a list of {n} {item}.
    
    {format_instructions}
    """
    
    list_prompt = PromptTemplate(
        template=list_template,
        input_variables=["n", "item"],
        partial_variables={"format_instructions": format_instructions}
    )
    
    # Format the prompt with our inputs
    prompt = list_prompt.format(n=5, item="fruits")
    
    # Send to LLM and parse the output
    # output = llm(prompt)
    # parsed_list = list_parser.parse(output)
    # print("Parsed List:", parsed_list)
    
    # Example 2: Structured output parser with multiple fields
    # Define the schema for our structured output
    response_schemas = [
        ResponseSchema(name="name", description="The name of the book"),
        ResponseSchema(name="author", description="The author of the book"),
        ResponseSchema(name="year", description="The publication year of the book"),
        ResponseSchema(name="genres", description="A list of genres the book belongs to"),
        ResponseSchema(name="summary", description="A brief summary of the book's plot")
    ]
    
    # Create a structured parser from our schemas
    structured_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    
    # Get the format instructions
    struct_format_instructions = structured_parser.get_format_instructions()
    
    # Create a prompt template
    struct_template = """
    Provide information about a famous book titled '{title}'.
    
    {format_instructions}
    """
    
    struct_prompt = PromptTemplate(
        template=struct_template,
        input_variables=["title"],
        partial_variables={"format_instructions": struct_format_instructions}
    )
    
    # Format the prompt with our inputs
    prompt = struct_prompt.format(title="To Kill a Mockingbird")
    
    # Send to LLM and parse the output
    # output = llm(prompt)
    # parsed_structure = structured_parser.parse(output)
    # print("Parsed Structure:", parsed_structure)
    
    # Example 3: Using Pydantic models for output parsing
    from langchain.output_parsers import PydanticOutputParser
    from pydantic import BaseModel, Field
    from typing import List
    
    # Define a Pydantic model for the output
    class BookInfo(BaseModel):
        title: str = Field(description="The title of the book")
        author: str = Field(description="The author of the book")
        year: int = Field(description="The publication year of the book")
        genres: List[str] = Field(description="List of genres the book belongs to")
        rating: float = Field(description="The average rating of the book (1-5)")
    
    # Create a parser based on the Pydantic model
    pydantic_parser = PydanticOutputParser(pydantic_object=BookInfo)
    
    # Get format instructions
    pydantic_format_instructions = pydantic_parser.get_format_instructions()
    
    # Create a prompt template
    pydantic_template = """
    Provide information about the book '{book_title}'.
    
    {format_instructions}
    """
    
    pydantic_prompt = PromptTemplate(
        template=pydantic_template,
        input_variables=["book_title"],
        partial_variables={"format_instructions": pydantic_format_instructions}
    )
    
    # Format the prompt
    prompt = pydantic_prompt.format(book_title="1984")
    
    # In a real scenario, you would send this to the LLM and parse the result:
    # output = llm(prompt)
    # parsed_book = pydantic_parser.parse(output)
    # print("Parsed Book:", parsed_book)
    
    return {
        "list_parser_instructions": format_instructions,
        "structured_parser_schemas": [schema.name for schema in response_schemas],
        "pydantic_model_fields": list(BookInfo.__annotations__.keys())
    }

###############################################################################
# MAIN EXECUTION
###############################################################################

def main():
    """
    This main function demonstrates all of the above examples together.
    """
    print("\n=== LangChain Basics Tutorial ===\n")
    
    # Uncomment any of these to run the examples
    # print("\n1. Basic LLM Interactions:")
    # basic_llm_example()
    # basic_chat_model_example()
    
    # print("\n2. Prompts and Templates:")
    # prompt_templates_example()
    
    # print("\n3. Chains:")
    # chains_example()
    
    # print("\n4. Memory:")
    # memory_example()
    
    # print("\n5. Agents:")
    # agents_example()
    
    # print("\n6. Document Loaders:")
    # document_loaders_example()
    
    # print("\n7. Vector Stores:")
    # vector_stores_example()
    
    # print("\n8. Output Parsing:")
    # output_parsing_example()
    
    print("\nTutorial complete!")

if __name__ == "__main__":
    # This block executes when the script
