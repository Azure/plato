"""
Baobab API module

This module runs a web API that mimics the behavior of Bonsai API.
One can run the API via unicorn or gunicorn.
To test the API locally, first install ``platotk`` and then run

    uvicorn platotk.baobab:app

"""
import asyncio
import os
import uuid

from aiocache import Cache
from fastapi import FastAPI, Request

from platotk.logger import log

namespace = os.getenv("BAOBAB_NAMESPACE", "example")

log.info("BAOBAB namespace: %s", namespace)
app = FastAPI()
cache = Cache(Cache.MEMCACHED, endpoint="localhost", port=11211, namespace=namespace)


class SimState:
    """
    Class used to maintain state for the API.

    Thanks to this class we are able to let the agent and the sim communicate
    through the API. The class uses a cache in the backend to store messages
    between sim and agent.
    """

    sequence_id: int = 0

    async def create_session(self, session_id):
        """Create a sim-API session."""
        self.session_id = session_id
        await self.cleanup()

    async def get(self, key):
        """Get key from the cache."""
        log.debug("Reading '%s' from cache.", key)
        val = await cache.get(key)
        while val is None:
            val = await cache.get(key)
            await asyncio.sleep(0.01)
        await cache.close()
        log.debug("Connection closed. Returning '%s'", val)
        return val

    async def unset(self, key):
        val = await cache.get(key)
        if val is None:
            log.debug("%s not found. Not deleting...", key)
            return
        log.debug("Deleting '%s'...", key)
        await cache.delete(key)
        await cache.close()

    async def set(self, key, val):
        """Save key with value val in the cache."""
        log.debug("Setting %s to %s", key, val)
        await cache.set(key, val)
        await cache.close()

    async def cleanup(self):
        log.debug("Cleaning all keys in cache.")
        for key in ("action", "method", "state"):
            await cache.delete(key)

    def idle_resp(self):
        self.sequence_id += 1
        resp = {
            "type": "Idle",
            "sequenceId": self.sequence_id,
            "idle": {"callbackTime": 1.0},
        }
        log.debug("Send response: %s", resp)
        return resp

    def reset_resp(self, config):
        self.sequence_id += 1
        resp = {
            "type": "EpisodeStart",
            "sequenceId": self.sequence_id,
            "episodeStart": {"config": config},
        }
        log.debug("Send response: %s", resp)
        return resp

    def step_resp(self, action):
        self.sequence_id += 1
        resp = {
            "type": "EpisodeStep",
            "episodeStep": {"action": action},
            "sequenceId": self.sequence_id,
        }
        log.debug("Send response: %s", resp)
        return resp


sim_state = SimState()


@app.post("/reset")
async def reset_sim(request: Request):
    """Agent wants to start a new episode."""
    req = await request.json()
    config = req.get("config")
    log.debug("Agent requested new episode.")
    log.debug("Config is: %s", config)
    await sim_state.set("config", config)
    await sim_state.set("method", "reset_send")
    state = await sim_state.get("state")
    await sim_state.unset("state")
    return state


@app.post("/step")
async def step(request: Request):
    """Agent wants to perform a step in the sim."""
    req = await request.json()
    action = req.get("action")
    log.debug("Requested step with action %s", action)
    await sim_state.set("action", action)
    await sim_state.set("method", "step_send")
    state = await sim_state.get("state")
    await sim_state.unset("state")
    return state


@app.post("/v2/workspaces/{workspace_id}/simulatorSessions", status_code=201)
async def create_session(workspace_id: str, request: Request):
    """Initiate a new sim-API session."""
    session_id = uuid.uuid4()
    await sim_state.create_session(session_id)
    resp = {
        "sessionId": session_id,
        "session_status": 0,
        "registration_time": 0,
        "last_seen_time": 0,
        "last_iterated_time": 0,
    }
    log.debug("Initiate new session %s", session_id)
    return resp


async def method_dispatch(req, method):
    """
    Run the appropriate function for the given method.

    Methods that end with _get do not return a response but have to wait that
    the agent provides new information. This information is fetched from the
    message queue.
    """
    log.debug("method is %s", method)
    if method is None:
        log.debug("Send Idle event.")
        return sim_state.idle_resp()

    if method in ("step_get", "reset_get"):
        state = req.get("state")
        log.debug("Sim sent state: %s", state)
        await sim_state.set("state", state)
        next_method = await sim_state.get("method")
        await sim_state.unset("method")
        return await method_dispatch(req, next_method)

    elif method == "reset_send":
        await sim_state.set("method", "reset_get")
        config = await sim_state.get("config")
        await sim_state.unset("config")
        return sim_state.reset_resp(config)

    elif method == "step_send":
        await sim_state.set("method", "step_get")
        action = await sim_state.get("action")
        await sim_state.unset("action")
        log.debug("Send action to the sim: %s", action)
        return sim_state.step_resp(action)


@app.post("/v2/workspaces/{workspace_id}/simulatorSessions/{session_id}/advance")
async def advance_session(workspace_id: str, session_id: str, request: Request):
    """
    Advance one step in the session.

    The sim polls the API to know what to do next.
    This could be any of the following event:

    - Idle: nothing to do. Ask again after some time
    - Reset: start a new episode
    - Step: perform one step in the sim given the agent's action

    This is the function that required implementing a message queue to
    allow communication between the sim and the agent.
    This function is deeply asynchronous as it has to wait for the agent input
    before replying something to the sim.
    """
    req = await request.json()
    method = None
    for _ in range(100):
        try:
            # Wait that the agent communicates the method
            method = await asyncio.wait_for(sim_state.get("method"), timeout=0.01)
        except asyncio.exceptions.TimeoutError:
            log.debug("Reading for method timed out.")
            pass
        if method is not None:
            await sim_state.unset("method")
            break
    return await method_dispatch(req, method)
