def parse_bool(argument_value: str) -> bool:
    if argument_value.lower() not in {"true", "false"}:
        raise ValueError("Supported values are true or false")

    return argument_value.lower() == "true"


def parse_array(argument_value: str) -> list[str]:
    return argument_value.split(sep=',')
