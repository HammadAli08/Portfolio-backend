
import os
import sys
import asyncio
from dotenv import load_dotenv

# Load env vars first
load_dotenv()

# Add current directory to path so we can import modules
sys.path.append(os.getcwd())

from rag_pipeline import get_rag_pipeline

def test_pipeline():
    print("Initializing pipeline...")
    try:
        pipeline = get_rag_pipeline()
        print("Pipeline initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        return

    print("Testing query 'Hello'...")
    try:
        response = pipeline.get_response("Hello")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Failed to get response: {e}")

if __name__ == "__main__":
    test_pipeline()
