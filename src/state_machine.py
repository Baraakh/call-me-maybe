"""State Machine is the core of the program"""

from .models import FunctionDefinition, PromptEntry, FunctionCallResult


class StateMachine:
    def __init__(self, funcs_def: list[FunctionDefinition]) -> None:
        self.funcs_def = funcs_def

    def get_func_calls_batch(
        self, prompts: list[PromptEntry]
    ) -> list[FunctionCallResult]:
        pass
