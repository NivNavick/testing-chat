class HelloWorld:
    def __init__(self, name="World"):
        self.name = name

    def say_hello(self, param2: str):
        return f"Hello, {self.name}!"

    def say_hello2(self, param_test: str):
        return f"Hello, {self.name}!2"

    def say_hello4(self):
        return f"Hello, {self.name}!2"

    def say_hello6(self, param: int):
        return f"Hello, {self.name}!2"

    def say_hello7(self, param: int, param2: float, param3: float, param4: bool):
        return f"Hello, {self.name}!2"

    def say_hello8(self, param: int, param2: float, param3: float, param4: bool):
        return f"Hello, {self.name}!2"
