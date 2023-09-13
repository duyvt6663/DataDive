# CoT few-shot prompting 
import sys
import numpy as np
sys.path.append("../Gloc")
sys.path.append("..")

import pandas as pd
from Gloc.generation.claimvis_prompt import Prompter, TemplateKey

# this import also acquires log_decor, don't import it again
from Gloc.utils.async_llm import *
from Gloc.processor.ans_parser import AnsParser
from Gloc.utils.normalizer import post_process_sql
from Gloc.nsql.database import NeuralDB
from Gloc.utils.utils import majority_vote
from rapidfuzz import fuzz
from DataMatching import DataMatcher
from pyinstrument import Profiler
from models import UserClaimBody, ClaimMap, Dataset
from collections import defaultdict
from functools import cache
from itertools import zip_longest
from itertools import product
import json
import random
import re
import spacy
import math

class TableReasoner(object):
	MIN_DATE = 1960
	MAX_DATE = 2020
	INDICATOR = {
		"alternative + complementary metrics": "value",
		"years": "datetime",
		"countries": "country"
	}
	
	def __init__(
			self, 
			temperature=0.0, 
			max_decode_steps=300, 
			samples=1, 
			model=Model.GPT3,
			datamatcher: DataMatcher = None 
		):
		self.prompter = Prompter()
		self.parser = AnsParser()
		self.datamatcher = datamatcher

		self.temperature = temperature # to change
		self.max_decode_steps = max_decode_steps # fixed
		self.samples = samples # to change
		self.model = model # fixed

		self.date_pattern = r"(@\(.*?\)|\d{4})(\s*-\s*(@\(.*?\)|\d{4}))?"
		
	async def _call_api_1(
		self: object, 
		question: str,
		template_key: TemplateKey,
		table: pd.DataFrame = None,
		samples: int = -1,
		temperature: float = -1,
		model: Model = Model.GPT3,
		max_decode_steps: int = -1
	):
		"""
			Call API for few-shot prompting using a question, a template, and a table 
			Input: question, template_key, table
			Output: prompt, response
		"""
		prompt = self.prompter.build_prompt(
			template_key=template_key,
			table=table,
			question=question
		)

		response = await call_model(
						model=model,
						max_decode_steps=max_decode_steps if max_decode_steps > 0 else self.max_decode_steps,
						temperature=temperature if temperature > 0 else self.temperature,
						prompt=prompt,
						samples=samples if samples > 0 else self.samples
					)

		return prompt, response
		
	async def _call_api_2(
			self, 
			prompt: list,
			temperature: float = -1,
			samples: int = -1,
			model: Model = Model.GPT3, 
			max_decode_steps: int = -1
		):
		"""
			Call API using a provide prompt
			Input: prompt
			Output: response
		"""
		response = await call_model(
						model=model,
						temperature=temperature if temperature > 0 else self.temperature,
						max_decode_steps=max_decode_steps if max_decode_steps > 0 else self.max_decode_steps,
						prompt=prompt,
						samples=samples if samples > 0 else self.samples
					)

		return response

	# @log_decorator
	async def _tag_claim(
			self, claim: str, 
			template_key: TemplateKey = TemplateKey.CLAIM_TAGGING,
			model: Model = Model.GPT3,
			verbose: bool = False,
			fewshot_samples: int = 6,
			gen_samples: int = 1
		):
		"""
			Tag claim with claim type
			Input: claim
			Output: claim type
		"""
		if model in [Model.GPT_TAG_2, Model.GPT_TAG_3, Model.GPT_TAG_4, Model.GPT4]:
			msg = [
				{"role": "system", "content": """Tag critical parts of the sentence. Critical parts include:
1. Countries. When there are phrases that represent a groups of countries, tag them with @(<COUNTRY_GROUP>?). For example @(Asian countries?).  
2. Value attributes. Rephrase attribute to be data-related if needed.
3. Datetime. Use 'X' and 'Y' variables to represent the default oldest and newest dates. When a date expression is not interpretable using single number, tag them with @(<EXPRESSION>). For example  @(Y - 2).
4. Also rephrase the sentence into a visualization task using extremal logic if possible."""},
			]
			filedir = "../Gloc/generation/finetune/claim_tagging.jsonl"
			with open(filedir, 'r') as file:
				data = [json.loads(line) for line in file]

				# indices = list(range(0, len(data)))
				# sampled_indices = [random.sample(indices, samples)]
				sampled_indices = range(0, fewshot_samples)
				random_sample = [msg for idx in sampled_indices for msg in data[idx]["messages"][1:]]

				msg.extend(random_sample)
				
			msg.append({"role": "user", "content": claim})
			claim_tag = await self._call_api_2(msg, model=model, max_decode_steps=700, samples=gen_samples, temperature=.8)
		else:
			_, claim_tag = await self._call_api_1(
								question=claim,
								template_key=template_key,
								model=model
							)

		data = [json.loads(sample) for sample in claim_tag]
		if verbose:
			print(f"model: {model}")
			for i, item in enumerate(data, 1):
				print(f"claim tag {i}: {item}")
		# check if the format of the response is correct
		return json.loads(claim_tag[0])
		
	async def _infer_country(
			self, claim: str, 
			dates: list[str], 
			values: list[str], 
			table: pd.DataFrame,
			datasets: list[Dataset], 
			verbose: bool=True,
		):
		queries, combos, cond = [], list(product(dates, values)), any(p in claim for p in ['with', 'Countries of'])
		if cond: # identify the new dataset if needed
			new_value = claim.replace("Countries of ", "").replace("Country", "").replace("with", "").strip()
			scores = []
			for dataset in datasets:
				embed_dict = self.datamatcher.attrs_embeddings[dataset.name]
				embeds = [embed_dict[attr] for attr in dataset.fields]
				scores.append(self.datamatcher.attr_score_batch(new_value, embeds))
			scores = np.array(scores)
			dataset = datasets[scores.argmax()]
			table, _, _ = self.datamatcher.load_table(dataset.name, dataset.fields, infer_date_and_country=True)
		for date, value in combos:
			if '-' in date:
				start, end = date.split('-')
				date = "from {} to {}".format(start, end)
			else:
				date = "in {}".format(date)
			
			if cond: # no need to add value
				queries.append("What are the {} {}?".format(claim, date))
			else: 
				queries.append("What are the {} of {} {}?".format(claim, value, date))

		db = NeuralDB([table], add_row_id=False, normalize=False, lower_case=False)

		if verbose: print(f"queries: {queries}")
		response = await self._exec_sqls_from_sub_queries(db, queries, fuzzy_match=True, verbose=verbose)

		try:
			countries = [item for sublist in response[0] for item in eval(sublist[0])\
							if isinstance(item, str) ]
			return set(countries)
		except Exception as e:
			if verbose: print(e)
			return []

	async def _infer_datetime(
			self, 
			queries: list, 
			start_default: int = 1960,
			end_default: int = 2021,
			use_llm: bool=False):
		"""
			Infer datetime from queries
			Input: queries
			Output: queries with datetime
		"""
		if use_llm:
			prompt = [
				{"role": "system", "content": f"""Please add date or time to the query if needed. 
				 
				 The rules will be demonstrated via the following examples. Let's use default oldest and newest dates as 1960 and 2021, respectively.
					1. Add the most recent date when the query lacks date. E.g "US' GDP > China's GDP" --> "US' GDP > China's GDP in 2021"
					2. Add the most recent date when the query lacks the end date. E.g "US' GDP > China's GDP since 2010" --> "US' GDP > China's GDP since 2010 to 2021"
					3. Add the oldest date when the query lacks the start date. E.g "US' GDP > China's GDP until 2010" --> "US' GDP > China's GDP from 1960 until 2010"
					4. Do not add date when the query already doesn't need more date. E.g "US' GDP > China's GDP in 2010" --> "US' GDP > China's GDP in 2010"

					User will give multiple queries inform of
				 /*
				 Please use the following default dates:
				 OLDEST: <oldest date>
				 NEWEST: <newest date>
					Q1: <query 1>
					Q2: <query 2>
					...
				 */

					Please answer the queries in the following format:
					A1: <answer 1>
					A2: <answer 2>
					..."""},
				{"role": "user", "content": f"""Please use the following default dates: 
				 OLDEST: {start_default}
				 NEWEST: {end_default}""" + "\n".join([f"Q{i+1}: {query}" for i, query in enumerate(queries)])}
			]
			answers = await self._call_api_2(prompt, model=Model.GPT4)
			# parse the answers
			answers = self.parser.parse_sql_2(answers[0])
			return answers
		else:
			self.nlp = spacy.load("en_core_web_sm")
			# use spacy dependency parsing
			for idx, query in enumerate(queries):                    
				doc, dates = self.nlp(query), []
				# count the number of dates within the query
				for ind, token in enumerate(doc):
					if token.ent_type_ == 'DATE' and token.head.pos_ in ['ADP', 'SCONJ']:
						dates.append(ind)
				
				if len(dates) == 0: # add the most recent date
					query = f"In {end_default}, {query}"
				elif len(dates) == 1: # rules
					ind = dates[0]-1
					if doc[ind].text.lower() in ['since', 'from']:
						query = f"{doc[:ind]} from {doc[ind+1]} to {end_default} {doc[ind+2:]}"
					elif doc[ind].text.lower() in ['til', 'until']:
						query = f"{doc[:ind]} from {start_default} {doc[ind:]}"
				
				queries[idx] = query

			return queries
					
	async def _suggest_queries(self, claim: str, table: pd.DataFrame=None, more_attrs: list=None):
		"""
			Suggest queries given a claim (and a table)
			Input: @claim, @table
			Output: 
				@query: list of suggested queries
				@vis: a list of vis tasks
				@explain: a list of exlanation for why the queries are suggested
				@attributes: list of attribute used in the queries
		"""
		# suggest different queries in form of "[{query: ...}, {query: ...}}]"
		template = TemplateKey.QUERY_GENERATION_2 if table is None \
												else TemplateKey.QUERY_GENERATION_3
		_, suggestions = await self._call_api_1(
								question=claim,
								template_key=template,
								table=table,
								model=Model.GPT3 # 4
							)
		queries, vis_tasks, reasons, attributes = self.parser.parse_gen_query(suggestions[0])

		attributes, col_set = set(attributes + more_attrs), set(table.columns)
		# add datefield if exist and infer datetime from the queries if needed
		time_batch = self.datamatcher.encode(["time", "date", "year"])
		col_embeds = list(self.datamatcher.attrs_embeddings[table.name].values())
		datefields, start_index = [], 1 if table.columns[0] == "row_id" else 0 # row_id added to table so index starts at 1
		for col, embed in zip(table.columns[start_index:], col_embeds): 
			score = self.datamatcher.attr_score_batch(embed, time_batch)
			if score > .5: datefields.append((col, score))
		best_datefield = max(datefields, key=lambda x: x[1])[0] if datefields else None   
		# print("best datefield: ", best_datefield)

		if best_datefield: # infer datetime
			oldest_date, newest_date = table[best_datefield].min(), 2020 # 2020 has the most data
			answers = await self._infer_datetime(queries + vis_tasks, oldest_date, newest_date, use_llm=True)
			# chop to queries and vis tasks
			queries, vis_tasks = answers[:len(queries)], answers[len(queries):]

		attributes.add(best_datefield)
		attributes = list(attributes)
		# further process the attributes
		for idx, attr in reversed(list(enumerate(attributes))):
			# fuzzy match when GPT hallucinating attributes
			if attr not in col_set: 
				lower_attr = attr.lower()
				similar_attr = max(table.columns, key=lambda col: fuzz.ratio(col, lower_attr))
				if fuzz.ratio(similar_attr.lower(), lower_attr) > 50 and similar_attr not in attributes:
					attributes[idx] = similar_attr
				else: # delete from tail
					del attributes[idx]       

		# assert every column name in attribute is unique
		assert len(attributes) == len(set(attributes)), "Column names in attributes are not unique"     
		return queries, vis_tasks, reasons, attributes

	async def _suggest_variable(self, claim: UserClaimBody, variable: str, verbose: bool=True):
		prompt = [
			{"role": "system", "content": """Given a given statement, a context paragraph, and an indicator, please suggest different sets of values for the indicator that could explore the context of the statement. Provide a one-sentence explanation for each set of values. Respond as JSON in following format:
    [{
		"values": [<TODO: list of values>],
		"explain": "<TODO: explanation>"
		}, 
	...]
	
	Each suggested set of value must be CONSISTENT with the indicator!"""},
			{"role": "user", "content": f"""Context: {claim.paragraph}\nStatement: "{claim.userClaim}"\nIndicator: "{variable}" """}
		]
		response = await self._call_api_2(prompt, model=Model.GPT3, temperature=.8, max_decode_steps=600)
		# if verbose: print(f"response: {response}")
		res = json.loads(response[0])
		new_res = list(map(lambda x: {"field": self.INDICATOR[variable],\
									  "values": [str(val) for val in x["values"]], \
									  "explain": x["explain"]}, res))
		return new_res
	
	async def _suggest_exploration(self, claim: UserClaimBody, verbose: bool=True):
		prompt = [
			{"role": "system", "content": """You are a keen learner. You are given a statement, and a context paragraph. Please suggest a list of questions that could explore the context of the statement. Provide a one-sentence explanation for each question and respond as JSON in following format:
			[{
				"question": "<question 1>",
				"explain": "<explanation>"
				},{
				"question": "<question 2>",
				"explain": "<explanation>"
			},...]"""},
			{"role": "user", "content": f"""Context: {claim.paragraph}\nStatement: "{claim.userClaim}" """}
		]
		response = await self._call_api_2(prompt, model=Model.GPT3, temperature=.8)
		# if verbose: print(f"response: {response}")
		return json.loads(response[0])
	
	async def _suggest_queries_2(self, body: UserClaimBody, verbose: bool=True, model: Model=Model.GPT_TAG_4):
		tasks = [self._suggest_variable(body, ind, verbose=verbose) \
	   						for ind in self.INDICATOR] \
				+ [self._tag_claim(
					body.userClaim, TemplateKey.CLAIM_TAGGING_2, 
		      		model=model, verbose=verbose
				)]
		attributes, years, countries, claim_tag = await asyncio.gather(*tasks)

		variables, claim_tag['cloze_vis'] = { "X": self.MIN_DATE, "Y": self.MAX_DATE }, claim_tag["vis"]
		for idx, tagged_date in enumerate(claim_tag["datetime"]):
			match = re.match(self.date_pattern, tagged_date)
			start_end = [match.group(1), match.group(3)] if match.group(3) else [match.group(1)]
			for date in start_end:
				if date.startswith("@("):
					val = eval(date[2:-1], variables)
					claim_tag["vis"] = claim_tag["vis"].replace(f"{{{date}}}", f"{{{str(val)}}}")
					claim_tag["datetime"][idx] = claim_tag["datetime"][idx].replace(f"{date}", f"{str(val)}")
				else:
					val = date

				claim_tag["rephrase"] = claim_tag["rephrase"].replace(f"{{{date}}}", f"{{{str(val)}}}")
				claim_tag["cloze_vis"] = claim_tag["cloze_vis"].replace(f"{{{date}}}", "{date}")
		for tagged_country in claim_tag["country"]:
			claim_tag["cloze_vis"] = claim_tag["cloze_vis"].replace(f"{{{tagged_country}}}", "{country}")
			if tagged_country.startswith("@(") and "countries" not in tagged_country.lower():
				new_tagged_country = f"@(Countries of {tagged_country[2:]}"
				claim_tag["vis"] = claim_tag["vis"].replace(f"{{{tagged_country}}}", f"{{{new_tagged_country}}}")
			
		for tagged_attr in claim_tag["value"]:
			claim_tag["cloze_vis"] = claim_tag["cloze_vis"].replace(f"{{{tagged_attr['rephrase']}}}", "{value}")

		claim_tag["suggestion"] = attributes + years + countries
		claim_tag["mapping"] = dict()
		claim_tag["value"] = [attr["rephrase"] for attr in claim_tag["value"]]
		claim_tag["date"] = claim_tag["datetime"]
		del claim_tag["datetime"]
		# if verbose: print(f"claim tag: {claim_tag}\n{'@'*75}")
		return claim_tag

	async def _decompose_query(self, query: str):
		"""
			Decompose query into subqueries
			Input: query
			Output: list of subqueries
		"""
		_, decomposed_ans = await self._call_api_1(
									question=query,
									template_key=TemplateKey.QUERY_DECOMPOSE
								)

		return self.parser.parse_dec_reasoning(decomposed_ans[0])
	
	async def _decompose_colunns(self, claim: str, table: pd.DataFrame):
		"""
			Decompose table into subtable using column decomposition
			Input: claim, table
			Output: subtable
		"""
		_, decomposed_cols = await self._call_api_1(
									question=claim,
									template_key=TemplateKey.COL_DECOMPOSE,
									table=table
								)

		cols = self.parser.parse_col_dec(decomposed_cols[0])
		return table.loc[:, cols]

	async def _generate_sql(
			self, query: str, 
			table: pd.DataFrame,
			template_key: TemplateKey,
			samples: int = 15,
			temperature: float = 0.6,
			fuzzy_match: bool = False,
			max_decode_steps: int = 700
		):
		"""
			Generate SQL queries based on the provided query and table.
			The type of SQL generation is determined by the template_key.
			The number of samples and the temperature can be adjusted.
			If fuzzy_match is set to True, the function will return post-processed SQL queries.

			Parameters:
				query (str): The query based on which SQL queries are generated.
				table (pd.DataFrame): The table used for SQL generation.
				template_key (TemplateKey): The key determining the type of SQL generation.
				samples (int, optional): The number of samples to generate. Defaults to 5.
				temperature (float, optional): The temperature for generation. Defaults to 0.4.
				fuzzy_match (bool, optional): Whether to return post-processed SQL queries. Defaults to False.

			Returns:
				list: A list of generated SQL queries.
		"""
		
		if template_key not in [TemplateKey.NSQL_GENERATION, TemplateKey.SQL_GENERATION, TemplateKey.SQL_GENERATION_2]:
			raise ValueError("Invalid template key for SQL generation")
		
		_, sqls = await self._call_api_1(
							question=query,
							template_key=template_key,
							table=table,
							samples=samples, # need samples to aggregate
							temperature=temperature, # need some creativity
							max_decode_steps=max_decode_steps # need to be long enough
						)

		if template_key == TemplateKey.NSQL_GENERATION:
			psqls = [self.parser.parse_nsql(sql) for sql in sqls]
		elif template_key == TemplateKey.SQL_GENERATION:
			psqls = [self.parser.parse_sql(sql) for sql in sqls]
		# list of list of sqls --> be careful when handling this case
		elif template_key == TemplateKey.SQL_GENERATION_2:
			psqls = [self.parser.parse_sql_2(sql) for sql in sqls]
			# transpose psqls, pad with "SELECT" if needed
			psqls = list(map(list, zip_longest(*psqls, fillvalue="SELECT")))
		
		if fuzzy_match:
			# bottle neck due to fuzzy matching on big tables
			sql_cache, value_map = set(), defaultdict(set) # control the number of distinct sqls
			def process_psqls(psqls):
				processed_psqls = []
				for psql in psqls:
					if isinstance(psql, str) and psql not in sql_cache:
						sql_cache.add(psql)
						new_sql_str, new_val_map = post_process_sql(
														sql_str=psql, 
														df=table,
														matcher=self.datamatcher,
														process_program_with_fuzzy_match_on_db=fuzzy_match,
														verbose=False
													)
						processed_psqls.append(new_sql_str)
						for k, v in new_val_map.items():
							value_map[k].update(v)
					elif isinstance(psql, list):
						processed_psqls.append(process_psqls(psql))
				return processed_psqls
			
			return process_psqls(psqls), value_map
		else:
			return psqls, None
	
	async def _exec_sqls_from_sub_queries(
			self,
			db: NeuralDB,
			queries: list,
			is_sequential: bool=False, 
			verbose: bool=False,
			fuzzy_match: bool=False
		):
		answers, value_map = [], None
		if is_sequential: # sequential prompting
			sqlss = [await self._generate_sql(
								query=query, 
								table=db.get_table(), 
								template_key=TemplateKey.SQL_GENERATION,
								fuzzy_match=fuzzy_match
							) for query in queries]
		else: # parallel prompting
			sqlss, value_map = await self._generate_sql(
										query=queries,
										table=db.get_table(),
										template_key=TemplateKey.SQL_GENERATION_2,
										fuzzy_match=fuzzy_match
									)
						
		for idx, (sqls, query) in enumerate(zip(sqlss, queries)):
			if verbose: print(f"Q{idx+1}: {query}\nGenerated SQLs: {sqls}")

			preds = []
			for sql in sqls:
				try:
					res = db.execute_query(sql)
					refined_res = self.parser.parse_sql_result(res)
					# if verbose: print(f"refined: {refined_res}")
					preds.append(refined_res)
				except Exception as e:
					continue
			
			top_ans, pred_sqls = majority_vote(
				nsqls=sqls,
				pred_answer_list=preds
			)
			top_ans = self._process_list(top_ans, verbose)
			unit = self.parser.parse_sql_unit(pred_sqls[0][0])
			if verbose: print(f"A{idx+1}: {top_ans}. {unit}\n{'*'*75}")

			answers.append((top_ans, unit))

		return answers, value_map
	
	async def _evaluate_soundness(self, reasoning:str): 
		evaluation = await self._call_api_2(
			prompt = [
				{"role": "system", "content": """You are an amazing logician. You are given a sequence of logical deduction based on real-world data. 
				You need to evaluate the soundness of the reasoning and fix the reasoning while still RETAIN the core idea and be as informative as possible in the following format.
				\{
					explain: "<TODO: explain why the reasoning is sound or not sound>"   
					revised: "<TODO: revised reasoning>"
				\}"""},
				{"role": "user", "content": reasoning},
			],
			model=Model.GPT4 # 4
		)

		return self.parser.parse_evaluation(evaluation[0])
	
	def _build_dec_prompt(self, sub_queries: list, answers: list, table: pd.DataFrame, question: str):
		dec_prompt = self.prompter.build_prompt(
						template_key=TemplateKey.DEC_REASONING_2,
						table=table,
						question=question,
					)
		dec_prompt.extend([
			{ "role": "user", "content": "\n".join(sub_queries[:-1]) },
			{ "role": "assistant", "content": "\n".join(answers) },
			{ "role": "user", "content": sub_queries[-1] }
		])
		return dec_prompt

	def _process_list(self, lst: list, verbose=False):
		try:
			if len(lst) > 30:
				# sometimes the answer is too long to fit into the prompt
				new_lst = [x for x in lst if isinstance(x, (int, float)) and not math.isnan(x)] 
				if not new_lst and isinstance(lst[0], str): 
					return str(random.sample(lst, 10)) # array full of string
				return f"Ranging from {str(min(new_lst))} to {str(max(new_lst))}, with average {str(sum(new_lst)/len(new_lst))}"
			return str(lst)
		except Exception as e:
			if verbose: print("error with list ans: ", e)
			return "[]"

	# @log_decorator
	async def reason(
			self, claim: str, 
			table: pd.DataFrame, 
			verbose=False, 
			fuzzy_match=False,
			more_attrs: list = []
		):
		"""
			Reasoning pipeline for CoT
			Input: claim, table
			Output: justification
		"""			
		db = NeuralDB(
			tables=[table],
			add_row_id=False,
			normalize=False,
			lower_case=False
		)
		# take first query from suggested queries
		suggestions, vis_tasks, _, attributes = await self._suggest_queries(
														claim=claim, 
														table=db.get_table_df(), 
														more_attrs=more_attrs
														)
		# update table with relevant attributes
		if attributes: 
			db.update_table(attributes) 
		if verbose: 
			print(f"generated queries: {suggestions}")
			if attributes: print(f"mapped attributes: {attributes}")

		reason_map = []
		for idx, query in enumerate(suggestions[:1]):
			# decompose queries
			sub_queries = await  self._decompose_query(query)
			if verbose: print(f"steps of reasoning: {sub_queries}")

			# execute sql corresponding to each subquery (up to the second last one)
			answers, value_map = await self._exec_sqls_from_sub_queries(
										db=db, queries=sub_queries[:-1], 
										is_sequential=False,
										verbose=verbose,
										fuzzy_match=fuzzy_match
									)
			sub_queries = [f"Q{i+1}: {query}" for i, query in enumerate(sub_queries)]
			answers = [f"A{i+1}: {ans}. {unit}" for i, (ans, unit) in enumerate(answers)]
			# generate prompt for decomposed reasoning
			dec_prompt = self._build_dec_prompt(sub_queries, answers, db.get_table_df(), query)
			# if verbose: print(f"full prompt:\n{dec_prompt}")
			answers.extend(await self._call_api_2(dec_prompt))

			response = await self._call_api_2(
								prompt = [
									{"role": "system", "content": """You are an amazing rhetorician. You are given a sequence of questions and answers that aims to tackle an ultimate question step by step. 
									You need to reframe the sequence to make it look like a coherent, smooth paragraph of logical deduction."""},
									{"role": "user", "content": "\n".join(query + "\n" + answer for query, answer in zip(sub_queries, answers))},
								],
								model=Model.GPT3 # 4
							)
			justification = response[0]

			# use GPT4 to evaluate whether the reasoning is sound or not, then revise the reasoning if needed
			# justification = await self._evaluate_soundness(justification)
			reason_map.append({
				"query": query,
				"visualization": vis_tasks[idx],
				"reasoning_steps": sub_queries,
				"justification": justification,
				"value_map": value_map
			})

		if verbose: print(f"final justifications: {reason_map}\n{'@'*75}")
		
		return {
			"suggestions": reason_map,
			"sub_table": {
				"data": db.get_table_df()
			},
			"attributes": attributes
		}
	
	async def reason_2(
				self, claim_map: ClaimMap, 
				df: pd.DataFrame,
				verbose=False, 
				fuzzy_match=True,
			):
		country_attr, date_attr = claim_map.mapping["country"], claim_map.mapping["date"]
		# claim_map already contains the elements constituting the queries
		queries, answers = [], []
		for country in claim_map.country:
			for datetime in claim_map.date:
				for category in claim_map.value:
					if '-' in datetime:
						start, end = datetime.split('-')
						date_mask = (df[date_attr] >= int(start)) & (df[date_attr] <= int(end))
						date_name = f"from {start} to {end}"
					else:
						date_mask = df[date_attr] == int(datetime)
						date_name = f"in {datetime}"
					
					if country.startswith("@("):
						country_name = claim_map.mapping[country]
						country_mask = df[country_attr].isin(country_name)
					else:
						country_mask = df[country_attr] == country
						country_name = country
					
					category_name = claim_map.mapping[category] 

					val = df[date_mask & country_mask][category_name].values
					val = self._process_list(val, verbose)
					query = f"Q{len(queries)+1}: What is the {category_name} of {country_name} {date_name}?"
					queries.append(query)
					answer = f"A{len(answers)+1}: {str(val)}"
					answers.append(answer)
		queries.append(f"Q{len(queries)+1}: {claim_map.rephrase}")
		# if verbose: print(f"queries: {queries}\nanswers: {answers}")
		dec_prompt = self._build_dec_prompt(queries, answers, df, claim_map.rephrase)
		# if verbose: print(f"full prompt:\n{dec_prompt}")
		answers.extend(await self._call_api_2(dec_prompt))

		msg = [
				{"role": "system", "content": """You are an amazing rhetorician and logician. You are given a sequence of questions and answers that aims to tackle an ultimate question step by step. 
				You need to reframe the sequence to make it look like a coherent, smooth paragraph of logical deduction. Make adjustment to the logic if needed."""},
				{"role": "user", "content": "\n".join(query + "\n" + answer for query, answer in zip(queries, answers))},
			]
		print(f"msg: {msg}")
		response = await self._call_api_2(
								prompt = msg,
								model=Model.GPT3 # 4
							)
		justification = response[0]
		# justification = await self._evaluate_soundness(justification)
		if verbose: print(f"justification: {justification}")
		return justification
						

async def main():
	data_matcher = DataMatcher(datasrc="../Datasets")
	table_reasoner = TableReasoner(datamatcher=data_matcher)
	query = "What was the population of South Korea in 2020?"
	# await table_reasoner._suggest_queries_2(query, verbose=True)
	data = await table_reasoner._tag_claim(
				query, TemplateKey.CLAIM_TAGGING_2, 
				model=Model.GPT_TAG_4, 
				verbose=True, 
				fewshot_samples=6, 
				gen_samples=7
			)

	# tasks = [
	# 		table_reasoner._tag_claim(
	# 				query, TemplateKey.CLAIM_TAGGING_2, 
	# 				model=Model.GPT4, verbose=True, samples=5
	# 			), 
	# 		table_reasoner._tag_claim(
	# 				query, TemplateKey.CLAIM_TAGGING_2, 
	# 				model=Model.GPT_TAG_3, verbose=True, samples=10
	# 			),
	# 		table_reasoner._tag_claim(
	# 			query, TemplateKey.CLAIM_TAGGING_2, 
	# 			model=Model.GPT_TAG_4, verbose=True, samples=10
	# 		)]
	# await asyncio.gather(*tasks)

if __name__ == "__main__":
	asyncio.run(main())
	
