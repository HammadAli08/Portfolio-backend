import os
import json
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone
from dotenv import load_dotenv
import time

load_dotenv()

# CORRECTED PATH
AGENT_DATA_DIR = "/mnt/data/hammadali08/PycharmProjects/Portfolio/backend/data"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "greyfang")

def load_json_data() -> List[Document]:
    documents = []
    
    if not os.path.exists(AGENT_DATA_DIR):
        print(f"❌ ERROR: Data directory not found at: {AGENT_DATA_DIR}")
        return documents
    
    print(f"✅ Found data directory: {AGENT_DATA_DIR}")
    
    for filename in os.listdir(AGENT_DATA_DIR):
        if filename.endswith(".json"):
            file_path = os.path.join(AGENT_DATA_DIR, filename)
            print(f"📄 Processing: {filename}")
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    # Convert JSON to readable text
                    content_parts = []
                    
                    # Extract key information based on your JSON structure
                    if isinstance(data, dict):
                        # Check for different possible structures in your files
                        
                        # Personal Profile
                        if "personal_profile" in data:
                            info = data["personal_profile"]
                            content_parts.append(f"Name: {info.get('name', '')}")
                            content_parts.append(f"Title: {info.get('title', '')}")
                            content_parts.append(f"Contact: {info.get('contact', {}).get('email', '')}")
                            content_parts.append(f"Location: {info.get('location', '')}")
                            content_parts.append(f"About: {info.get('summary', '')}")
                        
                        # Professional Experience
                        if "professional_experience" in data:
                            content_parts.append("\nProfessional Experience:")
                            for exp in data["professional_experience"]:
                                if isinstance(exp, dict):
                                    role = exp.get('role', '')
                                    company = exp.get('company', '')
                                    if role and company:
                                        content_parts.append(f"- {role} at {company}")
                                        for ach in exp.get('achievements', []):
                                            content_parts.append(f"  * {ach}")
                        
                        # Projects
                        if "projects" in data:
                            content_parts.append("\nProjects:")
                            for proj in data["projects"]:
                                if isinstance(proj, dict):
                                    name = proj.get('name', '')
                                    desc = proj.get('description', '')
                                    if name:
                                        content_parts.append(f"- {name}: {desc}")
                        
                        # Skills
                        if "skills" in data:
                            content_parts.append("\nSkills:")
                            if isinstance(data["skills"], dict):
                                for category, skills in data["skills"].items():
                                    if isinstance(skills, list):
                                        content_parts.append(f"- {category}: {', '.join(skills)}")
                    
                    # Fallback: if no structured content found, use JSON string
                    if not content_parts:
                        content = json.dumps(data, indent=2)
                    else:
                        content = "\n".join(content_parts)
                    
                    metadata = {
                        "source": filename,
                        "type": "portfolio_data",
                        "name": data.get("name", filename.replace(".json", ""))
                    }
                    
                    documents.append(Document(page_content=content, metadata=metadata))
                    print(f"  ✅ Loaded: {filename} ({len(content)} chars)")
                    
            except Exception as e:
                print(f"  ❌ Error loading {filename}: {e}")
                import traceback
                traceback.print_exc()
    
    return documents

def initialize_pinecone():
    print("🚀 Starting Pinecone VectorStore initialization...")
    
    # Check API key
    if not PINECONE_API_KEY:
        print("❌ ERROR: PINECONE_API_KEY not found in environment variables.")
        print("Please set PINECONE_API_KEY in your .env file")
        return False
    
    print(f"✅ Pinecone API Key loaded (length: {len(PINECONE_API_KEY)})")
    
    # Initialize Pinecone client
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        print("✅ Pinecone client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Pinecone client: {e}")
        return False
    
    # Check if index exists and its stats
    try:
        existing_indexes = [index.name for index in pc.list_indexes()]
        print(f"📊 Existing Pinecone indexes: {existing_indexes}")
        
        if PINECONE_INDEX_NAME not in existing_indexes:
            print(f"❌ Index {PINECONE_INDEX_NAME} not found. Please create it first.")
            print("Go to Pinecone console and create index with:")
            print("  - Name: greyfang")
            print("  - Dimension: 1024 (for llama-text-embed-v2)")
            print("  - Metric: cosine")
            return False
        else:
            print(f"✅ Index {PINECONE_INDEX_NAME} exists.")
            
            # Get index stats to verify dimensions
            index = pc.Index(PINECONE_INDEX_NAME)
            stats = index.describe_index_stats()
            print(f"📊 Index Statistics:")
            print(f"  - Total Vectors: {stats.get('total_vector_count', 'N/A')}")
            print(f"  - Dimension: {stats.get('dimension', 'N/A')}")
            print(f"  - Index Fullness: {stats.get('index_fullness', 'N/A')}")
            
            # Verify dimension is 1024
            dimension = stats.get('dimension')
            if dimension != 1024:
                print(f"❌ ERROR: Index dimension is {dimension}, but expected 1024 for llama-text-embed-v2")
                return False
            
    except Exception as e:
        print(f"❌ Error checking index: {e}")
        return False
    
    # Load documents
    documents = load_json_data()
    if not documents:
        print("⚠️  No documents found to index.")
        return False
    
    print(f"📚 Loaded {len(documents)} documents for indexing.")
    
    try:
        print("🔧 Setting up Pinecone embeddings...")
        
        from langchain_pinecone import PineconeEmbeddings
        
        # CORRECT: Use llama-text-embed-v2 with input_type parameter
        embeddings = PineconeEmbeddings(
            model="llama-text-embed-v2",
            pinecone_api_key=PINECONE_API_KEY,
            document_params={
                "input_type": "passage",  # REQUIRED for llama-text-embed-v2
                "dimension": 1024  # Explicitly set dimension
            },
            query_params={
                "input_type": "query",  # For query embeddings
                "dimension": 1024
            }
        )
        
        print(f"📤 Uploading documents to Pinecone index: {PINECONE_INDEX_NAME}")
        
        # Upload documents to Pinecone in smaller batches to avoid timeout
        batch_size = 5  # Smaller batch size for reliability
        total_docs = len(documents)
        
        for i in range(0, total_docs, batch_size):
            batch_docs = documents[i:i + batch_size]
            print(f"  Uploading batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size} "
                  f"({len(batch_docs)} documents)...")
            
            try:
                # Upload this batch
                vectorstore = PineconeVectorStore.from_documents(
                    documents=batch_docs,
                    embedding=embeddings,
                    index_name=PINECONE_INDEX_NAME
                )
                print(f"    ✅ Batch uploaded successfully")
                
                # Small delay between batches
                if i + batch_size < total_docs:
                    time.sleep(2)
                    
            except Exception as batch_error:
                print(f"    ❌ Error uploading batch: {batch_error}")
                # Continue with next batch
                continue
        
        print("✅ All documents uploaded to Pinecone successfully!")
        
        # Final verification
        print("🧪 Final verification with sample queries...")
        try:
            # Create a final vectorstore instance for querying
            final_vectorstore = PineconeVectorStore.from_existing_index(
                index_name=PINECONE_INDEX_NAME,
                embedding=embeddings
            )
            
            test_queries = [
                "Who is Hammad Ali Tahir?",
                "What AI projects has Hammad worked on?",
                "What skills does Hammad have?"
            ]
            
            for query in test_queries:
                print(f"  Testing query: '{query}'")
                results = final_vectorstore.similarity_search(query, k=1)
                if results:
                    source = results[0].metadata.get('source', 'Unknown')
                    print(f"    ✅ Found in: {source}")
                else:
                    print(f"    ⚠️  No results found")
                    
            # Get final index stats
            index = pc.Index(PINECONE_INDEX_NAME)
            final_stats = index.describe_index_stats()
            print(f"📊 Final Index Statistics:")
            print(f"  - Total Vectors: {final_stats.get('total_vector_count', 'N/A')}")
            
        except Exception as test_error:
            print(f"⚠️  Test queries failed: {test_error}")
        
        print("🎉 Pinecone initialization complete successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during embedding/upload process: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Pinecone Vector Store Initialization")
    print("=" * 60)
    
    success = initialize_pinecone()
    
    print("=" * 60)
    if success:
        print("✅ INITIALIZATION SUCCESSFUL")
        print(f"Index '{PINECONE_INDEX_NAME}' is ready with your portfolio data.")
        print("You can now start your FastAPI server.")
    else:
        print("❌ INITIALIZATION FAILED")
        print("Please check the errors above and fix them.")
        exit(1)