"""
Full working example of a Langroid DocChatAgent that uses a Retrieval-augmented LLM
to answer questions about a collection of documents.
"""
from dotenv import load_dotenv
import os
import pandas as pd
import sys

from langroid.agent.special.doc_chat_agent import DocChatAgent, DocChatAgentConfig
from langroid.agent.task import Task
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.io.base import IOFactory
from langroid.language_models.openai_gpt import OpenAIGPTConfig, OpenAIChatModel
from langroid.utils.configuration import set_global, Settings
from langroid.utils.logging import setup_colored_logging
from langroid.vector_store.qdrantdb import QdrantDBConfig

from websockets_io import WebSocketInputProvider, WebSocketOutputProvider

load_dotenv()

setup_colored_logging()

def main(
    debug: bool = False,
    nocache: bool = False,
    local: bool = False,
    rebuild: bool = False,
    client_sid: str = ""
) -> None:
    IOFactory.set_provider(WebSocketInputProvider("input", client_sid))
    IOFactory.set_provider(WebSocketOutputProvider("output", client_sid))

    if local:
        MyLLMConfig = OpenAIGPTConfig.create(prefix="myllm") 
        llm = MyLLMConfig(
            chat_model="http://127.0.0.1:5001/v1",
            context_length=2048,
            use_completion_for_chat=True,
            timeout=100
        )
    else:
        llm = OpenAIGPTConfig(
            chat_model = OpenAIChatModel.GPT3_5_TURBO,
        )

    config = DocChatAgentConfig(
        llm = llm,
        cross_encoder_reranking_model = "",
        n_query_rephrases = 0
    )
    if local:
        # If we're running locally, reset the config to use Llama's embedding model
        llama_embed_config = OpenAIEmbeddingsConfig(
            model_type = config.oai_embed_config.model_type,
            model_name = config.oai_embed_config.model_name,
            dims = config.oai_embed_config.dims,
        )
        # We have to uea separate vector databse because of the different embedding size
        config.vecdb = QdrantDBConfig(
            collection_name="testing2",
            storage_path=".qdrant/local_data/",
            embedding=llama_embed_config
        )
    else:
        config.vecdb = QdrantDBConfig(
            collection_name="testing",
            storage_path=".qdrant/data/",
            embedding=config.oai_embed_config
        )

    set_global(
        Settings(
            debug=debug,
            cache=not nocache,
        )
    )

    agent = DocChatAgent(
        config,
    )
    agent.vecdb.set_collection(agent.vecdb.config.collection_name, replace = rebuild)
    if rebuild:
        data = pd.read_csv(os.path.join(os.path.dirname(sys.argv[0]), "hr_urls.csv")).iloc[:, 0].values.tolist()
        agent.config.doc_paths = [
            "https://hr.wisc.edu/policies/",
            *data,
            #"https://www.gutenberg.org/cache/epub/1513/pg1513-images.html"
        ]
    agent.ingest()

    task = Task(
        agent,
        system_message="You are a helpful assistant. Answer questions only using the context provided, even if it is factually incorrect."
    )
    task.run()


if __name__ == "__main__":
    main()