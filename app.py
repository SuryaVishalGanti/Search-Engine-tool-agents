import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

# Load environment variables
load_dotenv()

# Streamlit App Title
st.title("🤖 LangChain - AI Chat with Search")

# Sidebar for API Key Input
st.sidebar.title("⚙️ Settings")
api_key = st.sidebar.text_input("🔑 Enter your Groq API Key:", type="password")

# Ensure API key is entered
if not api_key:
    st.warning("⚠️ Please enter your Groq API Key to continue.")
    st.stop()

# Initialize Search Tools
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv_tool = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

search_tool = DuckDuckGoSearchRun(name="Search")

# Store Chat History in Session State
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! 🤖 I'm a chatbot with web search capabilities. How can I assist you today?"}
    ]

# Display Previous Messages
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# Input Box for User Messages
prompt = st.chat_input(placeholder="Type your message here...")

# Process User Input
if prompt:
    # Append user message to history
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Initialize Groq LLM
    llm = ChatGroq(
        groq_api_key=api_key, 
        model_name="gemma2-9b-it",
        streaming=True
    )

    # Define AI Agent with Tools
    tools = [search_tool, arxiv_tool, wiki_tool]
    search_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

    # Generate AI Response
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        
        try:
            response = search_agent.invoke({"input": prompt}, callbacks=[st_cb])
            st.session_state["messages"].append({"role": "assistant", "content": response["output"]})
            st.write(response["output"])
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
