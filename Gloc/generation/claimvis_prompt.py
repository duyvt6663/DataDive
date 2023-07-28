"""ClaimVis Prompts."""

import enum
import random
from typing import Dict, Tuple
import pandas as pd
import copy
from utils.table import table_linearization, twoD_list_transpose
from utils.json import NoIndent, MyEncoder
import json
import os
import math
from generation.dater_prompt import PromptBuilder

class TemplateKey(str, enum.Enum):
    SUMMARY = "sum"
    ROW_DECOMPOSE = 'row'
    COL_DECOMPOSE = 'col'
    QUERY_DECOMPOSE = 'que'
    COT_REASONING = 'cot'
    DECOMPOSE_REASONING = 'dec'
    

class Prompter(object):
    def __init__(self) -> None:
        # respectively read pre-prompt files in fewshots folders and 
        # set corresponding attributes
        _path_ = "generation/fewshots/"
        for file_name in os.listdir(_path_):
            attr_name = '_' + file_name.upper()[:-5] + '_'
            with open(_path_ + file_name, "r") as file:
                setattr(self, attr_name, json.loads(file.read()))        

    def _get_template(self, template_key):
        """Returns a template given the key which identifies it."""
        if template_key == TemplateKey.COL_DECOMPOSE:
            return self._COL_DECOMPOSE_
        elif template_key == TemplateKey.DECOMPOSE_REASONING:
            return self._COT_DEC_REASONING_
        elif template_key == TemplateKey.QUERY_DECOMPOSE:
            return self._QUERY_DECOMPOSE_
        elif template_key == TemplateKey.SUMMARY:
            return self._SUMMARY_
        elif template_key == TemplateKey.COT_REASONING:
            return self._COT_REASONING_
        else:
            return self._ROW_DECOMPOSE_        

    def build_prompt(
            self, 
            template_key,
            table: pd.DataFrame,
            question: str = None,
            title: str = None,
            num_rows: int = 10,
            **kwargs
        ):
        """
            Builds a prompt given a table, question and a template identifier.
            This is a wrapper for dater prompt builder's old functions and 
            some new modules 
        """
        pb = PromptBuilder() # dater promptbuilder

        template = self._get_template(template_key)
        if template_key in [TemplateKey.COL_DECOMPOSE, TemplateKey.ROW_DECOMPOSE]:
            template.append({
                "role": "user", 
                "content": pb.build_generate_prompt(
                    table=table,
                    question=question,
                    title=title,
                    num_rows=num_rows,
                    select_type=template_key
                )
            })
        elif template_key == TemplateKey.QUERY_DECOMPOSE:
            template.append({
                "role": "user",
                "content": question
            })

        return template
               