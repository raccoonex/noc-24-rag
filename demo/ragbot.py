from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

API_KEY = "PUT_API_KEY_HERE"


class RagBot:
    def __init__(self) -> None:
        self.llm = OpenAI(api_key=API_KEY)
        self.embed_model = OpenAIEmbedding(api_key=API_KEY)

    def ingest(self, input_dir: str) -> None:
        docs = SimpleDirectoryReader(input_dir=input_dir).load_data()
        self.index = VectorStoreIndex.from_documents(
            documents=docs, embed_model=self.embed_model
        )
        self.query_engine = self.index.as_query_engine(llm=self.llm)
        self.chat_engine = self.index.as_chat_engine(
            chat_mode="condense_question", llm=self.llm, verbose=True
        )

    def query(self, question: str) -> str:
        response = self.query_engine.query(question)
        return response.response

    def chat(self, question: str, history: list) -> str:
        response = self.chat_engine.chat(question, chat_history=history)
        return response.response
