import os
from dotenv import load_dotenv

# .env Datei laden
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
USE_OPENAI = os.getenv("USE_OPENAI", "false").lower() == "true"

print("USE_OPENAI:", USE_OPENAI)
