import os
import uuid
import json
import re
import requests
from typing import Optional, List, Annotated, TypedDict, Dict
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# =========================
# 🔐 CONFIG
# =========================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PROPERTY_API_BASE = os.getenv("PROPERTY_API_BASE")
PROPERTY_API_KEY = os.getenv("PROPERTY_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is required")

if not PROPERTY_API_KEY:
    raise ValueError("PROPERTY_API_KEY environment variable is required")

if not PROPERTY_API_BASE:
    raise ValueError("PROPERTY_API_BASE environment variable is required")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    api_key=GROQ_API_KEY
)

# =========================
# 🛠 TOOL
# =========================
@tool
def search_properties(address: str,
                     min_price: Optional[int] = None,
                     max_price: Optional[int] = None,
                     min_bedroom: Optional[int] = None,
                     min_bathroom: Optional[float] = None) -> str:
    """Search properties using city or address."""


    payload = {
        "min_price": min_price or 0,
        "max_price": max_price or 999999999,
        "min_bedroom": min_bedroom,
        "min_bathroom": min_bathroom,
        "searched_address_formatted": address,
        "size": 5
    }


    payload = {k: v for k, v in payload.items() if v is not None}

    try:
        url = f"{PROPERTY_API_BASE}/address/{address}"

        resp = requests.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {PROPERTY_API_KEY}"}
        )

        data = resp.json()
        # print(data)
        props = data.get("data", []) or data.get("properties", [])

        if not props:
            return f"No listings found in {address}"

        result = f"\n🏡 Found {len(props)} properties in {address}:\n\n"

        for p in props:
            result += f"- {p.get('address')} | ${p.get('price')} | {p.get('bedroom')}bd/{p.get('bathroom')}ba\n"

        return result

    except Exception as e:
        return f"API Error: {str(e)}"


# =========================
# 🧠 STATE
# =========================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    slots: Dict
    question_count: int
    asked_fields: List[str]


# =========================
# 🔍 FALLBACK EXTRACTORS
# =========================
def extract_price(text):
    text = text.lower()

    match = re.search(r'(\d+)\s*k', text)
    if match:
        return int(match.group(1)) * 1000

    match = re.search(r'\d+', text)
    if match:
        val = int(match.group())
        if val >= 10000:
            return val

    return None


def extract_bedroom(text):
    match = re.search(r'(\d+)\s*(bed|bedroom)', text.lower())
    if match:
        return int(match.group(1))
    return None


def extract_address(text):
    text = text.lower().strip()
    text = re.sub(r"(properties|property|houses|homes|in|at|for)", "", text).strip()

    if len(text.split()) == 1 and text.isalpha():
        return text.capitalize()

    return None


# =========================
# 🧠 LLM DECISION
# =========================
def llm_decision(state):
    history = "\n".join([m.content for m in state["messages"]])

    prompt = f"""
You are a smart real estate assistant.

Extract:
- address (city)
- min_price
- min_bedroom
- min_bathroom

Rules:
- If address exists AND any filter exists → search
- Else → ask
- Never repeat questions

Return JSON:

{{
  "action": "ask" or "search",
  "question": "...",
  "slots": {{
    "address": string or null,
    "min_price": number or null,
    "max_price": number or null,
    "min_bedroom": number or null,
    "min_bathroom": number or null
  }}
}}

Conversation:
{history}
"""

    res = llm.invoke(prompt)
    content = res.content.strip().replace("```json", "").replace("```", "")

    try:
        return json.loads(content)
    except:
        return {"action": "ask", "question": "Could you clarify?", "slots": {}}


# =========================
# 🧭 ROUTER
# =========================
def router_node(state: AgentState):

    print("\n==============================")
    print("📥 STEP: New user input")

    # 👉 LAST USER MESSAGE
    last_user_msg = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_user_msg = m.content
            break

    # 👉 FALLBACK EXTRACTION
    price = extract_price(last_user_msg)
    beds = extract_bedroom(last_user_msg)
    address = extract_address(last_user_msg)

    if price:
        print("💰 Price:", price)
        state["slots"]["min_price"] = price

    if beds:
        print("🛏 Bedrooms:", beds)
        state["slots"]["min_bedroom"] = beds

    if address:
        print("📍 Address:", address)
        state["slots"]["address"] = address

    # 👉 LLM DECISION
    decision = llm_decision(state)
    print("🧠 LLM Decision:", decision)

    for k, v in decision.get("slots", {}).items():
        if v is not None:
            state["slots"][k] = v

    print("📦 Slots:", state["slots"])

    # =========================
    # 🔍 SEARCH
    # =========================
    if decision["action"] == "search":
        if "address" not in state["slots"]:
            return {
                "messages": [AIMessage(content="Which city are you looking in?")],
                "question_count": state["question_count"] + 1,
                "asked_fields": state["asked_fields"]
            }

        return {
            "messages": [
                AIMessage(
                    content="Searching properties 🔍",
                    tool_calls=[{
                        "id": str(uuid.uuid4()),
                        "name": "search_properties",
                        "args": state["slots"]
                    }]
                )
            ],
            "question_count": state["question_count"],
            "asked_fields": state["asked_fields"]
        }

    # =========================
    # 🛑 STOP AFTER 2 QUESTIONS
    # =========================
    if state["question_count"] >= 2:
        if "address" not in state["slots"]:
            return {
                "messages": [AIMessage(content="Please tell me the city.")],
                "question_count": state["question_count"],
                "asked_fields": state["asked_fields"]
            }

        return {
            "messages": [
                AIMessage(
                    content="Searching with available info 🔍",
                    tool_calls=[{
                        "id": str(uuid.uuid4()),
                        "name": "search_properties",
                        "args": state["slots"]
                    }]
                )
            ],
            "question_count": state["question_count"],
            "asked_fields": state["asked_fields"]
        }

    # =========================
    # ❓ ASK (NO REPEAT)
    # =========================
    question = decision["question"]

    field = "unknown"
    q = question.lower()

    if "budget" in q:
        field = "min_price"
    elif "city" in q or "where" in q:
        field = "address"
    elif "bedroom" in q:
        field = "min_bedroom"
    elif "bathroom" in q:
        field = "min_bathroom"

    if field in state["asked_fields"]:

    # 👉 Only search if we have minimum required data
        if "address" in state["slots"]:
            return {
                "messages": [
                    AIMessage(
                        content="Searching properties 🔍",
                        tool_calls=[{
                            "id": str(uuid.uuid4()),
                            "name": "search_properties",
                            "args": state["slots"]
                        }]
                    )
                ],
                "question_count": state["question_count"],
                "asked_fields": state["asked_fields"]
            }

        return {
            "messages": [AIMessage(content="Please tell me the city.")],
            "question_count": state["question_count"],
            "asked_fields": state["asked_fields"]
        }

    state["asked_fields"].append(field)

    return {
        "messages": [AIMessage(content=question)],
        "question_count": state["question_count"] + 1,
        "asked_fields": state["asked_fields"]
    }


# =========================
# 🧾 FINAL
# =========================
def final_node(state: AgentState):
    last = state["messages"][-1].content
    res = llm.invoke(f"Make this user friendly:\n{last}")
    return {"messages": [AIMessage(content=res.content)]}


# =========================
# 🏗 GRAPH
# =========================
tool_node = ToolNode([search_properties])

workflow = StateGraph(AgentState)

workflow.add_node("decision", router_node)
workflow.add_node("api", tool_node)
workflow.add_node("final", final_node)

workflow.add_edge(START, "decision")

workflow.add_conditional_edges(
    "decision",
    lambda x: "api" if x["messages"][-1].tool_calls else END,
    {"api": "api", END: END}
)

workflow.add_edge("api", "final")
workflow.add_edge("final", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


# =========================
# 💬 CHAT LOOP
# =========================
def interactive_chat():
    thread_id = "session-1"

    state = {
        "messages": [],
        "slots": {},
        "question_count": 0,
        "asked_fields": []
    }

    print("\n🏠 Smart Property Agent (Stable Version)\n")

    while True:
        user_input = input("👤 You: ")

        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        state["messages"].append(HumanMessage(content=user_input))

        state = app.invoke(
            state,
            {"configurable": {"thread_id": thread_id}}
        )

        print("🤖 Agent:", state["messages"][-1].content)


# ▶️ RUN
interactive_chat()