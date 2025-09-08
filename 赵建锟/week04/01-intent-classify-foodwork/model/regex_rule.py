import re
from typing import Union, List

from config import REGEX_RULE

REGEX_RULE_COMPILED = {}
for category in REGEX_RULE.keys():
    REGEX_RULE_COMPILED[category] = re.compile("|".join(REGEX_RULE[category]))


def model_for_regex(request_text: Union[str, List[str]]) -> Union[str, List[str]]:
    classify_result: Union[str, List[str]] = []

    if isinstance(request_text, str):
        for category in REGEX_RULE_COMPILED.keys():
            if REGEX_RULE_COMPILED[category].findall(request_text):
                classify_result.append(category)
        if not classify_result:
            classify_result.append("Other")
    elif isinstance(request_text, list):
        classify_result = []
        for text in request_text:
            is_classified = False
            for category in REGEX_RULE_COMPILED.keys():
                if REGEX_RULE_COMPILED[category].findall(request_text):
                    classify_result.append(category)
                    is_classified = True

            if not is_classified:
                classify_result.append("Other")
    else:
        raise Exception("格式不支持")

    return classify_result
