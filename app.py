from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.base import TaskResult

from dotenv import load_dotenv

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Query
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


# Load environment variables
load_dotenv()


# FastAPI app
app = FastAPI()


# Base directory (Render compatible)
BASE_DIR = Path(__file__).resolve().parent


# Static files
app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static"
)


# Templates
templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)


# OpenAI Client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
    api_key=OPENAI_API_KEY
)



# -----------------------------
# WebSocket Input Handler
# -----------------------------

class WebSocketInputHandler:

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket


    async def get_input(
        self,
        prompt: str,
        cancellation_token: Optional[object] = None
    ) -> str:

        try:
            await self.websocket.send_text(
                "SYSTEM_TURN:USER"
            )

            data = await self.websocket.receive_text()

            return data


        except WebSocketDisconnect:

            print(
                "Client disconnected during input wait."
            )

            return "TERMINATE"



# -----------------------------
# Create Interview Team
# -----------------------------

async def create_interview_team(
    websocket: WebSocket,
    job_position: str
):

    handler = WebSocketInputHandler(websocket)


    interviewer = AssistantAgent(
        name="Interviewer",
        model_client=model_client,

        description=f"Professional interviewer for {job_position}",

        system_message=f"""
You are a professional interviewer for a {job_position} position.

Rules:
- Ask one question at a time.
- Ask exactly 3 questions.
- Question 1: Technical skills.
- Question 2: Problem solving.
- Question 3: Culture and experience.
- Keep each question below 50 words.
- After completing all questions say TERMINATE.
"""
    )


    candidate = UserProxyAgent(
        name="Candidate",

        description="Job candidate",

        input_func=handler.get_input
    )


    evaluator = AssistantAgent(
        name="Evaluator",

        model_client=model_client,

        description="Career feedback evaluator",

        system_message="""
You are a career coach.

Analyze candidate answers.

Give short feedback.
Maximum 40 words.
"""
    )


    termination = TextMentionTermination(
        text="TERMINATE"
    )


    team = RoundRobinGroupChat(
        participants=[
            interviewer,
            candidate,
            evaluator
        ],

        termination_condition=termination,

        max_turns=15
    )


    return team



# -----------------------------
# Home Page
# -----------------------------

@app.get("/")
async def read_root(
    request: Request
):

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request
        }
    )



# -----------------------------
# WebSocket Interview
# -----------------------------

@app.websocket("/ws/interview")
async def websocket_endpoint(
    websocket: WebSocket,
    pos: str = Query("AI Engineer")
):

    await websocket.accept()


    try:

        team = await create_interview_team(
            websocket,
            pos
        )


        await websocket.send_text(
            f"SYSTEM_INFO: Starting interview for {pos}"
        )


        async for message in team.run_stream(
            task="Start the interview."
        ):


            if isinstance(
                message,
                TaskResult
            ):

                await websocket.send_text(
                    f"SYSTEM_END:{message.stop_reason}"
                )


            else:

                await websocket.send_text(
                    f"{message.source}:{message.content}"
                )


    except WebSocketDisconnect:

        print(
            "WebSocket disconnected."
        )


    except Exception as e:

        print(
            "Error:",
            e
        )