def contains_expected_entities(answer: str, expected_entities: list[str]) -> bool:
    lowered = answer.casefold()
    return all(entity.casefold() in lowered for entity in expected_entities)
