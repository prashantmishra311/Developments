import importlib, platform


def module_check(lib):
    try:
        # spec = importlib.util.find_spec(lib)
        # print("specs are:", spec)
        module = importlib.import_module(lib)
        print(f"successfully imported {lib} with version: {module.__version__}")
    except ImportError:
        print("failed to import..")


if __name__ == "__main__":
    
    # print("running on machine:", platform.machine())
    # LIB = input("provide lib to check: ")
    # module_check(lib=LIB)
    print(importlib.import_module.__class__.__repr__)
    
