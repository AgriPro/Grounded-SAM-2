import os
from dotenv import load_dotenv

def setup_environment():
    backend_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_file = ".env"
    if env_file:
        env_path = os.path.join(backend_root, env_file)
        if not os.path.exists(env_path):
            raise FileNotFoundError(f"Environment file {env_path} does not exist")
        load_dotenv(dotenv_path=env_path)