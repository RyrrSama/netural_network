def get_requirements():
    return _get_py_modules("requirements.txt")


def get_dev_requirements():
    return _get_py_modules("dev-requirements.txt")


def _get_py_modules(filename):
    with open(filename, "r") as file:
        modules = []
        for line in file.readlines():
            s_line = line.strip()
            if not s_line or s_line.startswith("#"):
                continue
            modules.append(s_line)
    return modules
