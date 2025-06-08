"""
Fraud Detection with the Enterprise-Ready Ground Truth Framework
================================================================
This single-file prototype demonstrates the *three‑pillar* architecture described in
"Enterprise‑Ready Ground Truth Framework for Unlocking Agentic AI Value" (Anjali Sharma, May 19 2025):

1. **Triple‑Extraction Knowledge Graph** – build a semantic graph from raw
   transaction logs.
2. **Agentic AI Orchestration** – use LangGraph to run an autonomous, multi‑step
   reasoning loop over the graph.
3. **Tool / Function Calling** – call external tools (risk‑scoring API and
   Slack alert) from within the agent.

The goal: ingest historical card‑payment events, learn behavioural patterns,
then process new transactions in real time, flagging suspicious activity and
posting an alert in Slack.

> ⚠️  This is *reference* code only.  Replace the stub API keys & endpoints with
> real ones, and harden for production (authZ, retries, monitoring, etc.).
"""

# -----------------------------
# 0. Setup & dependencies
# -----------------------------
# pip install openai langchain langgraph neo4j networkx slack_sdk python-dotenv
import os, csv, json, time, random, logging
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import networkx as nx
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda
# LangGraph (>=0.0.34)
from langgraph.graph import StateGraph, END, MessageGraph
from slack_sdk import WebClient as SlackClient
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY", "sk-demo")
slack_token = os.getenv("SLACK_BOT_TOKEN", "xoxb-demo")
slack_channel = os.getenv("SLACK_CHANNEL", "#fraud-alerts")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, openai_api_key=openai_api_key)
slack_client = SlackClient(slack_token)

# -----------------------------
# 1. Triple‑Extraction Knowledge Graph
# -----------------------------
@dataclass
class Triple:
    subject: str
    predicate: str
    obj: str

class KnowledgeGraph:
    """Simple in‑memory property graph built on NetworkX."""

    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_triple(self, t: Triple):
        self.graph.add_edge(t.subject, t.obj, predicate=t.predicate)

    def query_neighbors(self, node: str):
        return list(self.graph.neighbors(node))

KG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Extract <=5 (subject, predicate, object) triples from the text."),
        ("human", "{text}")
    ]
)
TRIPLE_PARSER = PydanticOutputParser(pydantic_object=Triple)

def extract_triples(text: str) -> List[Triple]:
    # Ask GPT to return JSON lines of triples (subject, predicate, object)
    triples: List[Triple] = []
    response = llm.invoke(KG_PROMPT.format(text=text))
    lines = response.content.strip().split("\n")
    for line in lines:
        try:
            s, p, o = json.loads(line)
            triples.append(Triple(s, p, o))
        except Exception:
            continue
    return triples

kg = KnowledgeGraph()

# Historical CSV example: date,card_id,merchant,amount,currency,location
HISTORICAL_CSV = "historical_transactions.csv"

if os.path.exists(HISTORICAL_CSV):
    with open(HISTORICAL_CSV) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            text = f"Card {row['card_id']} spent {row['amount']} {row['currency']} at {row['merchant']} in {row['location']} on {row['date']}."
            for t in extract_triples(text):
                kg.add_triple(t)

# -----------------------------
# 2. Agentic Orchestration with LangGraph
# -----------------------------

class TxnState(MessageGraph):
    """State passed between nodes."""
    transaction: Dict[str, Any]
    analysis: List[str] = []
    risk_score: float = 0.0
    decision: str = "undecided"

# -- Helper tool wrappers --

def call_risk_api(transaction: Dict[str, Any]) -> float:
    """Dummy risk engine – replace with real HTTPS call."""
    base = random.uniform(0, 1)
    # Example heuristic bumps
    if float(transaction["amount"]) > 1000:
        base += 0.5
    if transaction["location"] not in ["Toronto", "Montreal", "Vancouver"]:
        base += 0.3
    return min(base, 1.0)

def post_slack_alert(txn: Dict[str, Any], score: float):
    msg = (
        f"⚠️ *High‑risk transaction detected*\n"
        f"Card {txn['card_id']} – {txn['amount']} {txn['currency']} at {txn['merchant']} "
        f"in {txn['location']} (score={score:.2f})."
    )
    try:
        slack_client.chat_postMessage(channel=slack_channel, text=msg)
    except Exception as e:
        logging.error(f"Slack alert failed: {e}")

# -- LangGraph nodes --

def analyse_node(state: TxnState) -> TxnState:
    txn = state.transaction
    # contextualize with graph neighbors
    neighbors = kg.query_neighbors(f"Card {txn['card_id']}")
    neighbor_str = ", ".join(neighbors[:10]) if neighbors else "none"
    prompt = (
        f"The following card transaction occurred: {txn}. "
        f"Known historical connections: {neighbor_str}. "
        "Provide a short analysis: is it unusual?"
    )
    analysis = llm.invoke(prompt).content.strip()
    state.analysis.append(analysis)
    return state

def score_node(state: TxnState) -> TxnState:
    txn = state.transaction
    state.risk_score = call_risk_api(txn)
    return state

def decide_node(state: TxnState) -> TxnState:
    decision_prompt = (
        "You are a fraud analyst AI. Given analysis and risk score, decide if this "
        "transaction should be *cleared*, *escalated*, or *blocked*.\n"
        f"Analysis: {state.analysis[-1]}\nRisk: {state.risk_score:.2f}"
    )
    decision = llm.invoke(decision_prompt).content.strip().lower()
    state.decision = decision
    if decision in ["escalated", "blocked"]:
        post_slack_alert(state.transaction, state.risk_score)
    return state

# Build the graph
SG = StateGraph(TxnState)
SG.add_node("analyse", RunnableLambda(analyse_node))
SG.add_node("score", RunnableLambda(score_node))
SG.add_node("decide", RunnableLambda(decide_node))
SG.set_entry_point("analyse")
SG.add_edge("analyse", "score")
SG.add_edge("score", "decide")
SG.add_edge("decide", END)
fraud_graph = SG.compile()

# -----------------------------
# 3. Streaming new transactions
# -----------------------------

NEW_TXNS = "incoming_transactions.jsonl"  # one JSON doc per line

if os.path.exists(NEW_TXNS):
    with open(NEW_TXNS) as fh:
        for raw in fh:
            txn = json.loads(raw)
            result = fraud_graph.invoke({"transaction": txn})
            print(f"Processed txn {txn['id']} – decision: {result.decision}")
            time.sleep(0.2)
else:
    print("No new transactions file found – demo complete.")
