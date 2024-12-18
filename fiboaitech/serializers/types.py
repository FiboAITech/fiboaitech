from pydantic import BaseModel, ConfigDict

from fiboaitech import Workflow
from fiboaitech.connections import BaseConnection
from fiboaitech.flows import Flow
from fiboaitech.nodes import Node


class WorkflowYamlData(BaseModel):
    """Data model for the Workflow YAML."""

    connections: dict[str, BaseConnection]
    nodes: dict[str, Node]
    flows: dict[str, Flow]
    workflows: dict[str, Workflow]

    model_config = ConfigDict(arbitrary_types_allowed=True)
