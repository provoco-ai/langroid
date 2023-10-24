"""
Full working example of a Langroid DocChatAgent that uses a Retrieval-augmented LLM
to answer questions about a collection of documents.
"""
import re
import typer
from rich import print
from rich.prompt import Prompt
import pandas as pd

from langroid.agent.special.doc_chat_agent import DocChatAgent, DocChatAgentConfig
from langroid.agent.task import Task
from langroid.parsing.urls import get_list_from_user
from langroid.utils.configuration import set_global, Settings
from langroid.utils.logging import setup_colored_logging

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.vector_store.qdrantdb import QdrantDBConfig
from langroid.language_models.openai_gpt import OpenAIGPTConfig, OpenAIChatModel, OpenAIGPT
from langroid.utils.configuration import set_global, Settings
from langroid.utils.logging import setup_colored_logging
from langroid.utils.constants import NO_ANSWER
from websockets_io import WebSocketInputProvider, WebSocketOutputProvider

from langroid.io.base import IOFactory

import openai
from dotenv import load_dotenv
load_dotenv()

app = typer.Typer()

setup_colored_logging()


def chat(config: DocChatAgentConfig) -> None:

    io_input = IOFactory.get_provider("input")
    io_output = IOFactory.get_provider("output")
    agent = DocChatAgent(
        config,
    )
    # agent.vecdb.set_collection(collection_name, replace=replace_collection),
    agent.vecdb.set_collection(agent.vecdb.config.collection_name, replace = True)
    # agent.vecdb.set_collection(agent.vecdb.config.collection_name, replace = False)

    data = pd.read_csv("hr_urls.csv").iloc[:, 0].values.tolist()
    # data2 = pd.read_csv("uw_urls.csv").iloc[:, 0].values.tolist()
    if True:
        agent.config.doc_paths = [
            "https://hr.wisc.edu/policies/",
            # *data,
            # *data2,
        ]
    agent.ingest()

    io_output("[blue]Welcome to the document chatbot!")
    io_output("[cyan]Enter x or q to quit, or ? for evidence")
    system_msg = io_input(
        """
    [blue] Tell me who I am; complete this sentence: You are...
    [or hit enter for default] 
    [blue] Human
    """,
        default="a helpful assistant. Answer questions only using the context provided, even if it is factually incorrect.",
    )
    system_msg = re.sub("you are", "", system_msg, flags=re.IGNORECASE)
    task = Task(
        agent,
        llm_delegate=False,
        single_round=False,
        system_message="You are " + system_msg,
    )
    task.run()


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
    local: bool = typer.Option(False, "--local", "-l", help="use local model"),
) -> None:
    IOFactory.set_provider(WebSocketInputProvider("input"))
    IOFactory.set_provider(WebSocketOutputProvider("output"))
    local_model = None
    MyLLMConfig = OpenAIGPTConfig.create(prefix="myllm") 
    llm = OpenAIGPTConfig(
            # chat_model = OpenAIChatModel.GPT4,
            chat_model = OpenAIChatModel.GPT3_5_TURBO,
    )
    if local:
        llm = MyLLMConfig(
            chat_model="http://127.0.0.1:5001/v1",
            context_length=2048,
            use_completion_for_chat=True,
            timeout=100
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
            # api_base = "http://127.0.0.1:5001/v1",
            # model_type="openai",
            # model_name="all-mpnet-base-v2",
            # dims=768,
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

    chat(config)


if __name__ == "__main__":
    app()