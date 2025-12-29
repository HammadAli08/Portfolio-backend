import os
from typing import List, Dict, Any, Generator
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

class RAGPipeline:
    def __init__(self):
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.embeddings = PineconeEmbeddings(
            model="llama-text-embed-v2", 
            pinecone_api_key=self.api_key,
            query_params={
                "input_type": "query",
                "dimension": 1024
            }
        )
        self.llm = ChatGroq(
            model_name="openai/gpt-oss-20b",
            temperature=0,
            streaming=True
        )
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "greyfang")
        self.vectorstore = self._load_vectorstore()
        self.prompt = ChatPromptTemplate.from_template("""
        You ARE Hammad Ali Tahir, speaking directly to visitors on your portfolio website and make sure to try
        to communicate and do not give answers in a way that you are a machine.
        Speak in FIRST PERSON (I, me, my) - you are not an assistant talking ABOUT Hammad, you ARE Hammad.
        
        Your data and context:
        {context}
        
        Visitor's question (Which should get most priority to generate response):
        {question}
        
        Response guidelines:
        1. Always speak as "I" - never say "Hammad" or "he" when referring to yourself
        2. Be conversational, friendly, and confident - like you're chatting with a potential collaborator
        3. Keep responses concise and focused - avoid long markdown tables or overly formal structures
        4. Highlight your key achievements naturally in conversation
        5. Your motto: "AI = Logic + Data + Imagination" - embody this philosophy
        6. If asked something not in context, say "I haven't shared that publicly yet" or similar
        7. Show genuine enthusiasm for AI, ML, and building intelligent systems
        8. Never give the 'My site: <hammadalit-iahir-hywb9z.gamma.site/>' to anybody
        
        Your response (speak as Hammad and give concise answer untill and unless the details are been asked):
        """)

    def _load_vectorstore(self) -> PineconeVectorStore:
        print(f"Connecting to Pinecone index: {self.index_name}...")
        vectorstore = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings
        )
        return vectorstore

    def _query_understanding(self, query: str) -> str:
        return query

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        docs = self.vectorstore.similarity_search(query, k=k)
        return docs

    def format_docs(self, docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    def get_response_stream(self, query: str) -> Generator[str, None, None]:
        processed_query = self._query_understanding(query)
        docs = self.retrieve(processed_query)
        context = self.format_docs(docs)
        
        chain = (
            {"context": lambda x: context, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        for chunk in chain.stream(query):
            yield chunk

    def get_response(self, query: str) -> str:
        processed_query = self._query_understanding(query)
        docs = self.retrieve(processed_query)
        context = self.format_docs(docs)
        
        chain = (
            {"context": lambda x: context, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain.invoke(query)

# Singleton instance
rag_pipeline = None

def get_rag_pipeline():
    global rag_pipeline
    if rag_pipeline is None:
        rag_pipeline = RAGPipeline()
    return rag_pipeline
