"""
Tau-Bench Airline Assessment Green Agent
Evaluates white agents on airline customer service tasks with turn efficiency tracking.
"""

import uvicorn
import tomllib
import dotenv
import json
import time
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, SendMessageSuccessResponse, Message
from a2a.utils import new_agent_text_message, get_text_parts
from src.my_util import parse_tags, my_a2a

from tau_bench.envs import get_env
from tau_bench.types import SolveResult, RESPOND_ACTION_NAME, Action

dotenv.load_dotenv()


def load_agent_card_toml(agent_name):
    """Load agent configuration from TOML file."""
    current_dir = __file__.rsplit("/", 1)[0]
    with open(f"{current_dir}/{agent_name}.toml", "rb") as f:
        return tomllib.load(f)


async def ask_agent_to_solve(white_agent_url, env, task_index, max_num_steps=30):
    """
    Run the assessment loop between green and white agent.
    Tracks turn count for efficiency metric.
    """
    total_cost = 0.0
    turn_count = 0
    env_reset_res = env.reset(task_index=task_index)
    obs = env_reset_res.observation
    info = env_reset_res.info.model_dump()
    reward = 0.0

    # Build initial task prompt with tools and instructions
    task_description = f"""
You are an airline customer service assistant. Below is your knowledge base:
{env.wiki}

Available tools (use one per turn):
{json.dumps(env.tools_info, indent=2)}

Output format: Return JSON wrapped in <json>...</json> tags containing:
- "name": tool function name, or "{RESPOND_ACTION_NAME}" to reply directly to user
- "kwargs": tool arguments, or {{"content": "message"}} for direct replies

Current customer request: {obs}
    """

    next_green_message = task_description
    context_id = None
    
    for _ in range(max_num_steps):
        turn_count += 1
        print(f"[Turn {turn_count}] Sending to white agent...")
        
        white_agent_response = await my_a2a.send_message(
            white_agent_url, next_green_message, context_id=context_id
        )
        res_root = white_agent_response.root
        assert isinstance(res_root, SendMessageSuccessResponse)
        res_result = res_root.result
        assert isinstance(res_result, Message)
        
        # Maintain conversation context
        if context_id is None:
            context_id = res_result.context_id
        else:
            assert context_id == res_result.context_id, "Context ID mismatch"

        text_parts = get_text_parts(res_result.parts)
        assert len(text_parts) == 1, "Expected single text response"
        white_text = text_parts[0]
        print(f"[Turn {turn_count}] White agent responded")
        
        # Extract action from response
        white_tags = parse_tags(white_text)
        action_json = white_tags["json"]
        action_dict = json.loads(action_json)
        action = Action(**action_dict)

        # Execute action in environment
        env_response = env.step(action)
        reward = env_response.reward
        info = {**info, **env_response.info.model_dump()}

        # Prepare next message based on action type
        if action.name != RESPOND_ACTION_NAME:
            next_green_message = f"""
Tool execution output:
{env_response.observation}
            """
        else:
            next_green_message = f"""
Customer follow-up:
{env_response.observation}
            """
        
        if env_response.done:
            break

    # Store turn count for efficiency metric
    info["turn_count"] = turn_count
    return SolveResult(
        reward=reward,
        info=info,
        messages=[],
        total_cost=total_cost,
    )


class TauGreenAgentExecutor(AgentExecutor):
    """Executor for the green assessment agent."""
    
    def __init__(self):
        pass

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Parse task config and run assessment."""
        print("Assessment started...")
        user_input = context.get_user_input()
        tags = parse_tags(user_input)
        white_agent_url = tags["white_agent_url"]
        env_config_str = tags["env_config"]
        env_config = json.loads(env_config_str)

        # Initialize tau-bench environment
        print("Initializing environment...")
        assert len(env_config["task_ids"]) == 1, "Single task mode only"
        task_index = env_config["task_ids"][0]
        env = get_env(
            env_name=env_config["env"],
            user_strategy=env_config["user_strategy"],
            user_model=env_config["user_model"],
            task_split=env_config["task_split"],
            user_provider=env_config.get("user_provider", None),
            task_index=task_index,
        )
        metrics = {}

        # Run evaluation
        print("Running evaluation...")
        timestamp_started = time.time()
        res = await ask_agent_to_solve(white_agent_url, env, task_index)

        # Collect metrics
        metrics["time_used"] = time.time() - timestamp_started
        result_bool = metrics["success"] = res.reward == 1
        metrics["turns"] = res.info.get("turn_count", 0)
        result_emoji = "✅" if result_bool else "❌"

        print("Assessment complete.")
        await event_queue.enqueue_event(
            new_agent_text_message(
                f"Finished. White agent success: {result_emoji}\nMetrics: {metrics}\n"
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


def start_green_agent(agent_name="tau_green_agent", host="localhost", port=9001):
    """Launch the green assessment agent server."""
    print("Starting green agent...")
    agent_card_dict = load_agent_card_toml(agent_name)
    url = f"http://{host}:{port}"
    agent_card_dict["url"] = url

    request_handler = DefaultRequestHandler(
        agent_executor=TauGreenAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    a2a_app = A2AStarletteApplication(
        agent_card=AgentCard(**agent_card_dict),
        http_handler=request_handler,
    )

    # Build the base app and add endpoints for AgentBeats compatibility
    from starlette.routing import Route
    from starlette.responses import JSONResponse
    
    agent_id = agent_card_dict.get("name", "tau_green_agent")
    
    async def status_endpoint(request):
        return JSONResponse({"status": "server up, with agent running", "pid": port})
    
    async def agents_endpoint(request):
        return JSONResponse({"agents": [{"id": agent_id, "url": url}]})
    
    async def reset_endpoint(request):
        return JSONResponse({"status": "reset successful"})
    
    async def ready_endpoint(request):
        return JSONResponse({"ready": True})
    
    app = a2a_app.build()
    async def agent_status_endpoint(request):
        # AgentBeats checks for "status" == "ready" or similar
        return JSONResponse({"status": "ready", "ready": True})
    
    app.routes.append(Route("/status", status_endpoint, methods=["GET"]))
    app.routes.append(Route("/ready", ready_endpoint, methods=["GET"]))
    app.routes.append(Route("/agents", agents_endpoint, methods=["GET"]))
    app.routes.append(Route("/agents/{agent_id}", agent_status_endpoint, methods=["GET"]))
    app.routes.append(Route("/agents/{agent_id}/reset", reset_endpoint, methods=["POST"]))
    app.routes.append(Route("/agents/{agent_id}/ready", ready_endpoint, methods=["GET"]))

    uvicorn.run(app, host=host, port=port)
