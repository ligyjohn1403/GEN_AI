import os
import requests

from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from tavily import TavilyClient
from rich import print
from dotenv import load_dotenv

load_dotenv()

# ── Tools ─────────────────────────────────────────────────────────────────────
@tool
def weather_report(city: str) -> str:
    """This tool should provide weather related info about the city in India"""
    api_key  = os.getenv("OPENWEATHER_API_KEY")
    url      = f"https://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"
    response = requests.get(url)
    data     = response.json()

    if response.status_code != 200:
        return f"Error: {data.get('message', 'Unable to fetch weather')}"

    weather  = data["weather"][0]["description"]
    temp     = data["main"]["temp"]
    humidity = data["main"]["humidity"]

    return f"Weather report of {city}: temperature {temp}°C, humidity {humidity}%. Weather is {weather}."


@tool
def get_city_news(city: str) -> str:
    """This tool should provide latest news about the city in India"""
    client   = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = client.search(query=f"latest news in {city} India", search_depth="basic", max_results=2)
    results  = response.get("results", [])

    if not results:
        return "No news found."

    news_list = []
    for r in results:
        title   = r.get("title", "No title")
        url     = r.get("url", "")
        snippet = r.get("content", "")
        news_list.append(f"- {title}\n  🔗 {url}\n  📝 {snippet[:100]}...")

    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)


# ── Tool registry ─────────────────────────────────────────────────────────────
tools         = [weather_report, get_city_news]
tools_by_name = {t.name: t for t in tools}

# ── LLM with tools bound ──────────────────────────────────────────────────────
llm       = ChatMistralAI(model="mistral-small-2506")
llm_tools = llm.bind_tools(tools)

SYSTEM = SystemMessage(content="You are a useful city assistant AI tool.")


# ── Runnable: prepare messages for LLM ───────────────────────────────────────
prepare_messages = RunnableLambda(
    lambda state: [SYSTEM] + state["messages"]
)

# ── Runnable: call LLM ────────────────────────────────────────────────────────
call_llm = RunnableLambda(
    lambda messages: llm_tools.invoke(messages)
)

# ── Runnable: LLM chain (prepare → call) ─────────────────────────────────────
llm_chain = prepare_messages | call_llm


# ── Runnable: human approval check ───────────────────────────────────────────
def _human_approval(tool_call: dict) -> bool:
    """Ask user before running each tool."""
    tool_name    = tool_call["name"]
    tool_args    = tool_call["args"]
    confirmation = input(f"\n⚠️  Agent wants to use [{tool_name}] with args {tool_args}. Allow? (yes/no): ")
    return confirmation.strip().lower() == "yes"

human_approval = RunnableLambda(_human_approval)


# ── Runnable: execute single tool call ───────────────────────────────────────
def _execute_tool(tool_call: dict) -> ToolMessage:
    """Run approved tool or return denial message."""
    approved = human_approval.invoke(tool_call)

    if approved:
        tool   = tools_by_name[tool_call["name"]]
        result = tool.invoke(tool_call["args"])
        print(f"[green]✅ Tool '{tool_call['name']}' executed.[/green]")
    else:
        result = f"Tool '{tool_call['name']}' was denied by the user. Use available information to answer."
        print(f"[red]❌ Tool '{tool_call['name']}' denied.[/red]")

    return ToolMessage(
        content=str(result),
        tool_call_id=tool_call["id"]
    )

execute_tool = RunnableLambda(_execute_tool)


# ── Runnable: execute all tool calls in AI message ───────────────────────────
def _execute_all_tools(ai_message: AIMessage) -> list[ToolMessage]:
    """Map execute_tool runnable over all tool calls."""
    return execute_tool.map().invoke(ai_message.tool_calls)

execute_all_tools = RunnableLambda(_execute_all_tools)


# ── Runnable: check if LLM wants tools ───────────────────────────────────────
has_tool_calls = RunnableLambda(
    lambda ai_message: bool(ai_message.tool_calls)
)


# ── Agent loop ────────────────────────────────────────────────────────────────
def run_agent(messages: list) -> str:
    """
    Agentic loop using runnables:
    llm_chain | has_tool_calls → execute_all_tools → loop back
    """
    state = {"messages": messages}

    while True:
        # Runnable chain: prepare messages | call LLM
        ai_message = llm_chain.invoke(state)
        state["messages"].append(ai_message)

        # Runnable: check tool calls
        if not has_tool_calls.invoke(ai_message):
            # Final answer — extract content via runnable
            extract_content = RunnableLambda(lambda msg: msg.content)
            return extract_content.invoke(ai_message)

        # Runnable: execute all tools with human approval
        tool_messages = execute_all_tools.invoke(ai_message)
        state["messages"].extend(tool_messages)


# ── Chat loop ─────────────────────────────────────────────────────────────────
print("[bold cyan]🏙️  City Intelligence System[/bold cyan]")
print("Type [bold]0[/bold] to quit\n")

chat_history = []

while True:
    userchat = input("You: ")
    if userchat == "0":
        break

    chat_history.append(HumanMessage(content=userchat))
    final_answer = run_agent(chat_history)
    print(f"\n[green]Assistant:[/green] {final_answer}\n")