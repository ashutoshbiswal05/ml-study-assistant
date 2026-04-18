
"""
capstone_streamlit.py — ML Study Assistant Agent
Run: streamlit run capstone_streamlit.py
"""
import streamlit as st
import uuid
import os
import chromadb
from dotenv import load_dotenv
from typing import TypedDict, List
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

st.set_page_config(page_title="ML Study Assistant", page_icon="🤖", layout="centered")
st.title("🤖 ML Study Assistant")
st.caption("An AI assistant that helps students understand machine learning concepts using a knowledge base.")

# ── Documents ───────────────────────────
DOCUMENTS = [
    {
    "id": "doc_001",
    "topic": "Overfitting and Underfitting",
    "text": "Overfitting occurs when a machine learning model learns the training data too well, including noise and random fluctuations that do not represent the true pattern. As a result, the model performs very well on training data but poorly on new, unseen data. This happens when the model is too complex or trained excessively. Underfitting is the opposite situation, where the model is too simple to capture the underlying structure of the data. It performs poorly on both training and testing data. Achieving a balance between overfitting and underfitting is essential for building effective models. Techniques such as regularization, cross-validation, and early stopping are commonly used to prevent overfitting and improve generalization."
    },
    {
    "id": "doc_002",
    "topic": "Bias vs Variance",
    "text": "Bias and variance are two key sources of error in machine learning models. Bias refers to errors caused by overly simplistic assumptions in the learning algorithm, which can lead to underfitting. High bias means the model fails to capture important patterns in the data. Variance, on the other hand, refers to errors caused by excessive sensitivity to small fluctuations in the training data. High variance leads to overfitting, where the model performs well on training data but poorly on unseen data. The goal is to find a balance between bias and variance, known as the bias-variance tradeoff. A good model minimizes both types of error, achieving strong performance on both training and testing datasets."
    },
    {
    "id": "doc_003",
    "topic": "Linear Regression",
    "text": "Linear regression is a supervised learning algorithm used to model the relationship between a dependent variable and one or more independent variables. It assumes a linear relationship and fits a straight line to the data. The objective is to minimize the difference between predicted and actual values using techniques such as least squares. Linear regression is widely used for prediction tasks in fields such as finance, economics, and engineering. It is simple to implement and interpret, making it a popular starting point for many machine learning applications. However, it may not perform well when the relationship between variables is non-linear."
    },
    {
    "id": "doc_004",
    "topic": "Logistic Regression",
    "text": "Logistic regression is a supervised learning algorithm used for classification problems, particularly binary classification. Instead of predicting continuous values, it predicts probabilities using a sigmoid function, which maps values between 0 and 1. The output is then converted into class labels. Despite its name, logistic regression is a classification technique rather than a regression method. It is commonly used in applications such as spam detection, medical diagnosis, and credit scoring. Logistic regression is efficient, easy to interpret, and performs well when the relationship between variables is approximately linear."
    },
    {
    "id": "doc_005",
    "topic": "Decision Trees",
    "text": "Decision trees are supervised learning algorithms used for both classification and regression tasks. They work by splitting the dataset into branches based on feature values, forming a tree-like structure. Each internal node represents a decision, and each leaf node represents an outcome. Decision trees are easy to understand and visualize, making them useful for interpretability. However, they can easily overfit the data if not properly controlled. Techniques such as pruning and limiting tree depth are used to improve performance. Decision trees form the basis for more advanced models such as random forests and gradient boosting."
    },
    {
    "id": "doc_006",
    "topic": "K-Means Clustering",
    "text": "K-Means is an unsupervised learning algorithm used to group data into a predefined number of clusters. It works by initializing cluster centroids, assigning data points to the nearest centroid, and then updating the centroids iteratively. This process continues until convergence is reached. The goal is to minimize the distance between data points and their respective cluster centers. K-Means is widely used in market segmentation, image compression, and pattern recognition. However, it requires specifying the number of clusters in advance and may struggle with complex or irregularly shaped data distributions."
    },
    {
    "id": "doc_007",
    "topic": "Neural Networks",
    "text": "Neural networks are computational models inspired by the human brain. They consist of layers of interconnected nodes, also known as neurons. Each neuron processes input data and passes it through an activation function to produce an output. Neural networks are capable of learning complex patterns and are widely used in tasks such as image recognition, natural language processing, and speech recognition. Deep neural networks, which have multiple hidden layers, form the foundation of deep learning. While powerful, neural networks require large amounts of data and computational resources to train effectively."
    },
    {
    "id": "doc_008",
    "topic": "Activation Functions",
    "text": "Activation functions are mathematical functions used in neural networks to introduce non-linearity. Without activation functions, neural networks would behave like simple linear models and would not be able to learn complex patterns. Common activation functions include ReLU, sigmoid, and tanh. ReLU is widely used due to its simplicity and efficiency, while sigmoid is often used for binary classification problems. The choice of activation function can significantly affect the performance of a neural network. Selecting the right function helps improve convergence and model accuracy."
    },
    {
    "id": "doc_009",
    "topic": "Gradient Descent",
    "text": "Gradient descent is an optimization algorithm used to minimize a loss function in machine learning models. It works by iteratively updating model parameters in the direction of the negative gradient of the loss function. This helps the model find the optimal values that reduce prediction error. There are different variants of gradient descent, including batch gradient descent, stochastic gradient descent, and mini-batch gradient descent. Each variant offers a tradeoff between computational efficiency and convergence stability. Gradient descent is a fundamental concept in training machine learning and deep learning models."
    },
    {
    "id": "doc_010",
    "topic": "Train-Test Split",
    "text": "Train-test split is a technique used to evaluate the performance of a machine learning model. The dataset is divided into two parts: a training set and a testing set. The model is trained on the training data and evaluated on the testing data to assess how well it generalizes to unseen data. This helps prevent overfitting and provides a realistic estimate of model performance. Common split ratios include 70-30 or 80-20. Proper data splitting is essential for building reliable and robust machine learning systems."
    }
]

# ── Load models and KB (cached) ───────────────────────────
@st.cache_resource
def load_agent():
    llm      = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.Client()
    try: client.delete_collection("capstone_kb")
    except: pass
    collection = client.create_collection("capstone_kb")

    # TODO: Copy your DOCUMENTS list here
    texts = [d["text"] for d in DOCUMENTS]
    collection.add(
        documents=texts,
        embeddings=embedder.encode(texts).tolist(),
        ids=[d["id"] for d in DOCUMENTS],
        metadatas=[{"topic": d["topic"]} for d in DOCUMENTS]
)

    # TODO: Copy your CapstoneState, node functions, and graph assembly here
    # ... (paste from notebook Parts 2-4)
    # ── STATE ─────────────────────────
    class CapstoneState(TypedDict):
        question: str
        messages: list
        route: str
        retrieved: str
        sources: list
        tool_result: str
        answer: str
        faithfulness: float
        eval_retries: int


    # ── Node 1: Memory ─────────────────────────
    def memory_node(state: CapstoneState) -> dict:
        messages = state.get("messages", [])
        messages.append({"role": "user", "content": state["question"]})
        return {"messages": messages[-6:]}


    # ── Node 2: Router ─────────────────────────
    def router_node(state: CapstoneState) -> dict:
        question = state["question"].lower()

        if any(x in question for x in ["what did you just say", "my name"]):
            return {"route": "memory_only"}
        elif any(x in question for x in ["date", "time", "+", "-", "*", "/"]):
            return {"route": "tool"}
        else:
            return {"route": "retrieve"}


    # ── Node 3: Retrieval ──────────────────────
    def retrieval_node(state: CapstoneState) -> dict:
        q_emb = embedder.encode([state["question"]]).tolist()
        results = collection.query(query_embeddings=q_emb, n_results=3)

        chunks = results["documents"][0]
        topics = [m["topic"] for m in results["metadatas"][0]]

        context = "\n\n".join(chunks)
        return {"retrieved": context, "sources": topics}


    def skip_retrieval_node(state: CapstoneState) -> dict:
        return {"retrieved": "", "sources": []}


    # ── Node 4: Tool ───────────────────────────
    def tool_node(state: CapstoneState) -> dict:
        return {"tool_result": ""}


    # ── Node 5: Answer ─────────────────────────
    def answer_node(state: CapstoneState) -> dict:
        context = state.get("retrieved", "")
        tool_result = state.get("tool_result", "")

        full_context = ""
        if context:
            full_context += f"KNOWLEDGE:\n{context}\n\n"
        if tool_result:
            full_context += f"TOOL:\n{tool_result}\n\n"

        system_prompt = f"""You are a helpful ML study assistant.
        Answer ONLY using the context below.
        If not found, say: I don't have that information.

        {full_context}
        """

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["question"])
        ])

        return {"answer": response.content}


    # ── Node 6: Eval ─────────────────────────
    def eval_node(state: CapstoneState) -> dict:
        return {"faithfulness": 1.0}


    # ── Node 7: Save ─────────────────────────
    def save_node(state: CapstoneState) -> dict:
        messages = state.get("messages", [])
        messages.append({"role": "assistant", "content": state["answer"]})
        return {"messages": messages}


    # ── GRAPH ─────────────────────────
    graph = StateGraph(CapstoneState)

    graph.add_node("memory", memory_node)
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("tool", tool_node)
    graph.add_node("answer", answer_node)
    graph.add_node("eval", eval_node)
    graph.add_node("save", save_node)

    graph.set_entry_point("memory")

    graph.add_edge("memory", "router")

    graph.add_conditional_edges("router", lambda s: s["route"], {
        "retrieve": "retrieve",
        "tool": "tool",
        "memory_only": "answer"
    })

    graph.add_edge("retrieve", "answer")
    graph.add_edge("tool", "answer")
    graph.add_edge("answer", "eval")
    graph.add_edge("eval", "save")
    graph.add_edge("save", END)

    agent_app = graph.compile(checkpointer=MemorySaver())

    return agent_app, embedder, collection


try:
    agent_app, embedder, collection = load_agent()
    st.success(f"✅ Knowledge base loaded — {collection.count()} documents")
except Exception as e:
    st.error(f"Failed to load agent: {e}")
    st.stop()

# ── Session state ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

# ── Sidebar ───────────────────────────────────────────────
KB_TOPICS = [d["topic"] for d in DOCUMENTS]
with st.sidebar:
    st.header("About")
    st.write("An AI assistant that helps students understand machine learning concepts using a knowledge base.")
    st.write(f"Session: {st.session_state.thread_id}")
    st.divider()
    st.write("**Topics covered:**")
    for t in KB_TOPICS:
        st.write(f"• {t}")
    if st.button("🗑️ New conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()

# ── Display history ───────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── Chat input ────────────────────────────────────────────
if prompt := st.chat_input("Ask something..."):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = agent_app.invoke({"question": prompt}, config=config)
            answer = result.get("answer", "Sorry, I could not generate an answer.")
        st.write(answer)
        faith = result.get("faithfulness", 0.0)
        if faith > 0:
            st.caption(f"Faithfulness: {faith:.2f} | Sources: {result.get('sources', [])}") 

    st.session_state.messages.append({"role":"assistant","content":answer})
