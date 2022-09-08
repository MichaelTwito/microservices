#!/bin/bash

cd dolores && source env/bin/activate && source .env.example && flask run& 

cd prediction_manager && source env/bin/activate && source .env.example && python grpc_server.py