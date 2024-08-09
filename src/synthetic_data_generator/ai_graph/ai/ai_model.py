from pydantic import BaseModel


class InputModel(BaseModel):
    def get_prompt(self) -> str:
        pass


class OutputModel(BaseModel):
    def get_output_description(self) -> str:
        pass
