from functools import partial
import json
import random
import math
import time
import os
import pathlib
from dotenv import load_dotenv, set_key
import datetime
from typing import Dict, Any, Union
from microsoft_bonsai_api.simulator.client import BonsaiClientConfig, BonsaiClient
from microsoft_bonsai_api.simulator.generated.models import (
    SimulatorState,
    SimulatorInterface,
    SimulatorSessionResponse,
)
from azure.core.exceptions import HttpResponseError
from sim import SimpleAdder

import argparse

LOG_PATH = "logs"


def main(
    render: bool = False,
    log_iterations: bool = False,
    debug: bool = False,
):
    """Main entrypoint for running simulator connections

    Parameters
    ----------
    render : bool, optional
        visualize steps in environment, by default True, by default False
    log_iterations: bool, optional
        log iterations during training to a CSV file
    config_setup: bool, optional
        if enabled then uses a local `.env` file to find sim workspace id and access_key
    env_file: str, optional
        if config_setup True, then where the environment variable for lookup exists
    workspace: str, optional
        optional flag from CLI for workspace to override
    accesskey: str, optional
        optional flag from CLI for accesskey to override
    """

    # SETUP CONFIG VARS USING CONFIG YML
    # TODO: Remove the need to overwrite environment variables.
    import yaml
    config_yaml = "unmanaged_api_config.yml"
    dir_name = os.path.dirname(os.path.abspath(__file__))
    config_filepath = dir_name+"//"+config_yaml
    with open(config_filepath, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    bonsai_api_url = "http://" + str(config["bonsai_api_url"]) + ":" + str(config["bonsai_api_port"])
    workspace_id = config["workspaceId"]
    os.environ["SIM_API_HOST"] = bonsai_api_url
    os.environ["SIM_WORKSPACE"] = workspace_id
    os.environ["SIM_ACCESS_KEY"] = "unused-password"

    try: 
        workspace = os.environ["SIM_WORKSPACE"]
        accesskey = os.environ["SIM_ACCESS_KEY"]
    except:
        raise IndexError(
            f"Workspace or access key not set or found. Please define them as part of your environment variables."
        )

    # Create sim model object
    sim_model = SimpleAdder(env_config={})
    sim_name = "SimpleAdder"

    # Configure client to interact with Bonsai service
    config_client = BonsaiClientConfig()
    client = BonsaiClient(config_client)

    # Create simulator session and init sequence id
    registration_info = SimulatorInterface(
        name=sim_name,
        timeout=60,
        simulator_context=config_client.simulator_context,
    )

    print(
        "config: {}, {}".format(config_client.server, config_client.workspace)
    )
    
    
    def CreateSession(
        registration_info: SimulatorInterface, config_client: BonsaiClientConfig
    ):
        """Creates a new Simulator Session and returns new session, sequenceId
        """

        try:
            print(
                "config: {}, {}".format(config_client.server, config_client.workspace)
            )
            registered_session: SimulatorSessionResponse = client.session.create(
                workspace_name=config_client.workspace, body=registration_info
            )
            print("Registered simulator. {}".format(registered_session.session_id))

            return registered_session, 1
        except HttpResponseError as ex:
            print(
                "HttpResponseError in Registering session: StatusCode: {}, Error: {}, Exception: {}".format(
                    ex.status_code, ex.error.message, ex
                )
            )
            raise ex
        except Exception as ex:
            print(
                "UnExpected error: {}, Most likely, it's some network connectivity issue, make sure you are able to reach bonsai platform from your network.".format(
                    ex
                )
            )
            raise ex

    #registered_session, sequence_id = CreateSession(registration_info, config_client)
    attempts = 0
    while attempts < 3:
        try:
            registered_session, sequence_id = CreateSession(
                registration_info, config_client
            )
            break
        except Exception as e:
            print(e)
            attempts += 1
            print("\nRetrying to register session in 10 secs...")
            time.sleep(10)
    
    if attempts >= 3:
        raise RuntimeError("Unable to register session after 3 attempts. Aborting.")
    

    episode = 0
    iteration = 0
    sim_model_state = { 'sim_halted': False }
    try:
        while True:
            # Proceed to the next event by calling the advance function and passing the simulation state
            # resulting from the previous event. Note that the sim must always be able to return a valid
            # structure from get_state, including the first time advance is called, before an EpisodeStart
            # message has been received.
            # WARNING: it's risky using `get_state` without first initializing the sim.
            sim_state = SimulatorState(
                sequence_id=sequence_id, state=sim_model_state, halted=sim_model_state.get('sim_halted', False)
            )
    
            try:
                event = client.session.advance(
                    workspace_name=config_client.workspace,
                    session_id=registered_session.session_id,
                    body=sim_state,
                )
                sequence_id = event.sequence_id
                print(
                    "[{}] Last Event: {}".format(time.strftime("%H:%M:%S"), event.type)
                )
            except HttpResponseError as ex:
                print(
                    "HttpResponseError in Advance: StatusCode: {}, Error: {}, Exception: {}".format(
                        ex.status_code, ex.error.message, ex
                    )
                )
                continue
            except Exception as err:
                print("Unexpected error in Advance: {}".format(err))
                continue

            # Event loop
            if event.type == "Idle":
                time.sleep(event.idle.callback_time)
                print("Idling...")
            elif event.type == "EpisodeStart":
                print("Episode config being used:", event.episode_start.config)
                sim_model_state = sim_model.bonsai_reset()
                print(f"state {sim_model_state}")
                episode += 1
            elif event.type == "EpisodeStep":
                iteration += 1
                print(f"action {event.episode_step.action}")
                sim_model_state = sim_model.bonsai_step(event.episode_step.action)
                print(f"state {sim_model_state}")
            elif event.type == "EpisodeFinish":
                print("Episode Finishing...")
                sim_model_state = { 'sim_halted': False }
                iteration = 0
            elif event.type == "Unregister":
                print(
                    "Simulator Session unregistered by platform because '{}', Registering again!".format(
                        event.unregister.details
                    )
                )
                return

    except KeyboardInterrupt:
        # Gracefully unregister with keyboard interrupt
        client.session.delete(
            workspace_name=config_client.workspace,
            session_id=registered_session.session_id,
        )
        print("Unregistered simulator.")
    except Exception as err:
        # Gracefully unregister for any other exceptions
        client.session.delete(
            workspace_name=config_client.workspace,
            session_id=registered_session.session_id,
        )
        print("Unregistered simulator because: {}".format(err))


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Bonsai and Simulator Integration...")
    parser.add_argument(
        "--render", action="store_true", default=False, help="Render training episodes",
    )
    parser.add_argument(
        "--log-iterations",
        action="store_true",
        default=False,
        help="Log iterations during training",
    )
    parser.add_argument(
        "--debug",
        default=False,
        help="Debug sim using default embedded methods in sim.py",
    )

    args, _ = parser.parse_known_args()


    main(
        render=args.render,
        log_iterations=args.log_iterations,
        debug=args.debug,
    )

