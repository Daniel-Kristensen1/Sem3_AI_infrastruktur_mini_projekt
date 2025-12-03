Download Docker from "docker pull traneat/mini-ai-server:latest"

Here the final.py file has been made into a docker file for easier testing. 
The file does have some GB in size, which is the LLM model and the requirements.

Starting the dockerfile, will start the local server, then you can start the final.html and use the adress made in the html interface window to get into the server.


If you want to start the server use the "uvicorn main:app --reload --host 127.0.0.1 --port 8000" code and paste in the host adress in the interface or use 127.0.0.1/docs