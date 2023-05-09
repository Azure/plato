import os
import pandas as pd
from flask import Flask, request, jsonify
import json
from datetime import datetime, timedelta
import yaml
from csv import writer
import numpy as np
import random


class BonsaiAPI:
    def __init__(self, config_yaml = 'unmanaged_api_config.yml'):
        
        # Data reciever API - using SOAP
        self.df = pd.DataFrame()

        # Store config file path
        dir_name = os.path.dirname(os.path.abspath(__file__))
        self.config_yaml = dir_name+"//"+config_yaml
        # Load config file
        self.config = None
        with open(self.config_yaml, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.session_id_count = 2000
        self.session_id_dict = dict()
    

    def init_env_vars(self, bonsai_api_url, bonsai_api_port, workspaceId, accesskey):

        os.environ["SIM_API_HOST"] = f"http://{bonsai_api_url}:{bonsai_api_port}"
        os.environ["SIM_WORKSPACE"] = workspaceId
        os.environ["SIM_ACCESS_KEY"] = accesskey
        
        print("Environment variables set successfully")
        print("SIM_WORKSPACE:", os.environ["SIM_WORKSPACE"])
    

    def instance_api_app(self):
        app = Flask(__name__)

        # EXTRACT API CONFIG
        # Extract bonsai api url and port
        self.bonsai_api_url = self.config["bonsai_api_url"]
        self.bonsai_api_port = self.config["bonsai_api_port"]
        # Extract workspace name
        self.workspaceId = self.config["workspaceId"]
        self.accesskey = self.config["accesskey"]

        # INIT ENVIRONMENT VARIABLES
        self.init_env_vars(self.bonsai_api_url, self.bonsai_api_port, self.workspaceId, self.accesskey)

        # -----------------------------------------------------------------------
        
        # SIMULATOR CREATE: POST METHOD
        @ app.route(f'/v2/workspaces/{self.workspaceId}/simulatorSessions', methods=['POST'])
        def bonsai_create():
            # Get the incoming json data
            json_data = request.get_json(force=True)
            print(json_data)

            # ACKNOWLEDGE CREATE REQUEST
            # Define session id
            session_id = self.session_id_count
            # Increment session id count
            self.session_id_count += 1
            # Add session id to session id dict
            self.session_id_dict[session_id] = 0
            # Define data to return
            data = {
                "sessionId": session_id,
                "session_status": 0,
                "registration_time": 0,
                "last_seen_time": 0,
                "last_iterated_time": 0,
            }

            # Return data
            return jsonify(data), 201

        
        # SIMULATOR DELETE: POST METHOD
        @ app.route(f'/v2/workspaces/{self.workspaceId}/simulatorSessions/<int:session_id>', methods=['POST'])
        def bonsai_delete(session_id):
            # Get the incoming json data
            json_data = request.get_json(force=True)
            print(json_data)

            data = {
                "sessionId": session_id,
            }

            del self.session_id_dict[session_id]

            # Return data
            return jsonify(data), 204

        # SIMULATOR ADVANCE: POST METHOD
        @ app.route(f'/v2/workspaces/{self.workspaceId}/simulatorSessions/<int:session_id>/advance', methods=['POST'])
        def bonsai_advance(session_id):
            # Get the incoming json data
            json_data = request.get_json(force=True)
            print(json_data)

            # Extract state data from json
            sim_state = json_data.get("state", {})
            reward = json_data.get("reward", 2.0)
            terminate = json_data.get("terminate", False)
            truncate = json_data.get("reward", False)

            # Define data to return            
            if self.session_id_dict[session_id] == 0:
                # Respond to stay idle
                response = self.idle_response(session_id)
                # Increment counter for sequence_id
                self.session_id_dict[session_id] = 1
            
            elif self.session_id_dict[session_id] == 1:
                # Respond to episode start
                response = self.episode_start_response(session_id)
                # Increment counter for sequence_id
                self.session_id_dict[session_id] = 2

            else:
                # Example action SimpleAdder
                example_action = {"addend": random.randint(-10, 10)}
                # Respond to episode step
                response = self.episode_step_response(session_id, example_action)
                # Increment counter for sequence_id
                self.session_id_dict[session_id] += 1

                if self.session_id_dict[session_id] > 5:
                    self.session_id_dict[session_id] = 0

            return response

        app.run(host=self.bonsai_api_url, port=self.bonsai_api_port, debug=True)
    

    def idle_response(self, session_id):

        sequence_id = self.session_id_dict[session_id]

        # Define data to return
        data = {
            "type": "Idle",
            "sequenceId": sequence_id,
            "idle": {"callbackTime": 2.0}, # callbackTime is in seconds
            "registration_time": 0,
            "last_seen_time": 0,
            "last_iterated_time": 0,
        }
        return jsonify(data), 200
    

    def episode_start_response(self, session_id):

        sequence_id = self.session_id_dict[session_id]

        # Define data to return
        data = {
            "type": "EpisodeStart",
            "sequenceId": sequence_id,
            "episodeStart": {"config": {}}, # callbackTime is in seconds
            "registration_time": 0,
            "last_seen_time": 0,
            "last_iterated_time": 0,
        }
        return jsonify(data), 200
    

    def episode_step_response(self, session_id, action):

        sequence_id = self.session_id_dict[session_id]

        # Define data to return
        data = {
            "type": "EpisodeStep",
            "sequenceId": sequence_id,
            "episodeStep": {"action": action},
            "registration_time": 0,
            "last_seen_time": 0,
            "last_iterated_time": 0,
        }
        return jsonify(data), 200
    

    def episode_finish_response(self, session_id):

        sequence_id = self.session_id_dict[session_id]

        # Define data to return
        data = {
            "type": "EpisodeFinish",
            "sequenceId": sequence_id,
            "episodeFinish": {"reason": 'Unspecified'},
            "registration_time": 0,
            "last_seen_time": 0,
            "last_iterated_time": 0,
        }
        return jsonify(data), 200



if __name__ == '__main__':
    data = BonsaiAPI()
    data.instance_api_app()

