
import sys
import traceback

try:
    print("Importing modules...")
    import os
    from dotenv import load_dotenv
    import langchain_pinecone
    import langchain_groq
    print("Imports successful.")
except Exception:
    traceback.print_exc()
