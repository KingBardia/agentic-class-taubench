"""
Tau-Bench White Agent (Assessee)
Target agent being evaluated on airline customer service tasks.
"""

import uvicorn
import dotenv
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill, AgentCard, AgentCapabilities
from a2a.utils import new_agent_text_message
from litellm import completion


dotenv.load_dotenv()

# System prompt optimized for turn efficiency
EFFICIENCY_SYSTEM_PROMPT = """You are a helpful assistant focused on resolving requests efficiently.
Guidelines:
- Keep responses brief and action-oriented
- Complete tasks in as few steps as possible
- Ask clarifying questions only when essential"""


def prepare_white_agent_card(url):
    """Create the agent card for registration."""
    skill = AgentSkill(
        id="task_fulfillment",
        name="Task Fulfillment",
        description="Handles user requests and completes tasks",
        tags=["general"],
        examples=[],
    )
    card = AgentCard(
        name="tau_white_agent",
        description="White agent for Tau-Bench airline assessment",
        url=url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )
    return card


class GeneralWhiteAgentExecutor(AgentExecutor):
    """Executor that handles incoming assessment requests."""
    
    def __init__(self):
        self.ctx_id_to_messages = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Process incoming message and generate response."""
        user_input = context.get_user_input()
        
        # Initialize conversation with system prompt if new context
        if context.context_id not in self.ctx_id_to_messages:
            self.ctx_id_to_messages[context.context_id] = [
                {"role": "system", "content": EFFICIENCY_SYSTEM_PROMPT}
            ]
        
        messages = self.ctx_id_to_messages[context.context_id]
        messages.append({"role": "user", "content": user_input})
        
        # Generate response using Claude Haiku
        response = completion(
            messages=messages,
            model="claude-3-haiku-20240307",
            custom_llm_provider="anthropic",
            temperature=0.0,
        )
        next_message = response.choices[0].message.model_dump()
        messages.append({"role": "assistant", "content": next_message["content"]})
        
        await event_queue.enqueue_event(
            new_agent_text_message(
                next_message["content"], context_id=context.context_id
            )
        )

    async def cancel(self, context, event_queue) -> None:
        raise NotImplementedError


def start_white_agent(agent_name="general_white_agent", host="localhost", port=9002):
    """Launch the white agent server."""
    print("Starting white agent...")
    url = f"http://{host}:{port}"
    card = prepare_white_agent_card(url)

    request_handler = DefaultRequestHandler(
        agent_executor=GeneralWhiteAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    a2a_app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )

    # Add endpoints for AgentBeats compatibility
    from starlette.routing import Route
    from starlette.responses import JSONResponse
    
    async def status_endpoint(request):
        return JSONResponse({"status": "server up, with agent running", "pid": port})
    
    async def agents_endpoint(request):
        return JSONResponse({"agents": [{"id": agent_name, "url": url}]})
    
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
