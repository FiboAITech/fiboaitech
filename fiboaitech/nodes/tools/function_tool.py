import inspect
from typing import Any, Callable, Generic, Literal, TypeVar

from pydantic import Field, create_model

from fiboaitech.nodes import ErrorHandling, Node, NodeGroup
from fiboaitech.nodes.node import ensure_config
from fiboaitech.runnables import RunnableConfig
from fiboaitech.utils.logger import logger

T = TypeVar("T")


class FunctionTool(Node, Generic[T]):
    """
    A tool node for executing a specified function with the given input data.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = Field(default="Function Tool")
    description: str = Field(
        default="A tool for executing a function with given input."
    )
    error_handling: ErrorHandling = Field(
        default_factory=lambda: ErrorHandling(timeout_seconds=600)
    )

    def run_tool(self, **kwargs: Any) -> Any:
        """
        Execute the function logic with provided arguments.

        This method must be implemented by subclasses.

        :param kwargs: Arguments to pass to the function.
        :return: Result of the function execution.
        """
        raise NotImplementedError("run_tool must be implemented by subclasses")

    def execute(
        self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        """
        Execute the tool with the provided input data and configuration.

        :param input_data: Dictionary of input data to be passed to the tool.
        :param config: Optional configuration for the runnable instance.
        :return: Dictionary with the execution result.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        try:
            result = self.run_tool(**input_data)
        except TypeError as e:
            logger.error(f"Invalid input parameters: {e}")
            raise ValueError("Invalid input parameters")

        logger.debug(f"Tool {self.name} - {self.id}: finished with result {result}")
        return {"content": result}

    def get_schema(self):
        """
        Generate the schema for the input and output of the tool.

        :return: Dictionary representing the input and output schema.
        """
        cls = self.__class__
        run_tool_method = self.run_tool
        if hasattr(cls, "_original_func"):
            run_tool_method = cls._original_func

        signature = inspect.signature(run_tool_method)
        parameters = signature.parameters

        fields = {}
        for name, param in parameters.items():
            if name == "self":
                continue
            annotation = (
                param.annotation if param.annotation != inspect.Parameter.empty else Any
            )
            default = ... if param.default == inspect.Parameter.empty else param.default
            fields[name] = (annotation, default)

        input_model = create_model(f"{cls.__name__}Input", **fields)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": input_model.schema(),
            "output_schema": {
                "type": "object",
                "properties": {"content": {"type": "any"}},
            },
        }


def function_tool(func: Callable[..., T]) -> type[FunctionTool[T]]:
    """
    Decorator to convert a function into a FunctionTool subclass.

    :param func: Function to be converted into a tool.
    :return: A FunctionTool subclass that wraps the provided function.
    """

    class FunctionToolFromDecorator(FunctionTool[T]):
        name: str = Field(default=func.__name__)
        description: str = Field(
            default=func.__doc__
            or f"A tool for executing the {func.__name__} function."
        )
        _original_func = staticmethod(func)

        def run_tool(self, **kwargs: Any) -> T:
            return func(**kwargs)

    FunctionToolFromDecorator.__name__ = func.__name__
    FunctionToolFromDecorator.__qualname__ = (
        f"FunctionToolFromDecorator({func.__name__})"
    )
    FunctionToolFromDecorator.__module__ = func.__module__

    return FunctionToolFromDecorator
