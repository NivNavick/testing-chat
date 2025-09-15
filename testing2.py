class HelloWorld:
    def __init__(self, name="World"):
        self.name = name

    def say_hello3(self):
        return f"Hello, {self.name}!2"

    def say_hello4(self):
        return f"Hello, {self.name}!2"

    def say_hello5(self, param1: bool):
        return f"Hello, {self.name}!2"

    def say_hello77(self, param1: bool, params: float, param3: int, params4: int):
        return f"Hello, {self.name}!9"

    def say_hello9(self, param1: bool, params: float, param3: int):
        return f"Hello, {self.name}!9"
