from cyclopts import App

app = App()

@app.command
def func1(a: int):
    """do thing"""
    return a

@app.default
def func2(a: int, thing: bool = False):
    """do thing"""
    return a

def cli():
    app()
