from importlib.machinery import SourceFileLoader
import os

# imports the module from the given path
cwd=os.getcwd()
path=cwd + "/app.py"

app = SourceFileLoader("app",path).load_module()

if __name__ == "__main__":
	app.app.run(use_reloader=True, debug=True)
