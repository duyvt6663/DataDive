import asyncio
import json
import sys

from matplotlib import pyplot as plt
sys.path.append("../Gloc")
sys.path.append("..")

from Gloc.utils.normalizer import post_process_sql
from Pipeline.DataMatching import DataMatcher
from Gloc.nsql.database import NeuralDB
from Gloc.processor.ans_parser import AnsParser
from AutomatedViz import AutomatedViz
from llama_index.query_engine import PandasQueryEngine
import openai
import pandas as pd
import os
from ClaimDetection import ClaimDetector
from TableReasoning import TableReasoner
from Gloc.utils.async_llm import Model
from models import *
from api import *

class Tester():
    def __init__(self, datasrc: str = None):
        self.datasrc = datasrc
        self.dm = DataMatcher(datasrc=datasrc)

    def test_post_process_sql(self):
        sql = """No, we cannot determine if the share of children in work dropped by one percentage point between 2012 and 2016 as the 
data provided does not include the necessary information """
        table = pd.read_csv(os.path.join(self.datasrc, "Social Development.csv"))
        # table.columns = table.columns.str.lower()
        # table = table.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        table.reset_index(inplace=True)
        table.rename(columns={'index': 'row_id'}, inplace=True)
        matcher = DataMatcher()

        new_sql, value_map = post_process_sql(
            sql_str=sql,
            df=table,
            verbose=True,
            matcher=matcher
        )
        print(new_sql, value_map)
    
    def test_filter_data(self):
        table = pd.read_csv(os.path.join(self.datasrc, "Social Protection & Labor.csv"))
        # table.columns = table.columns.str.lower()
        # table = table.applymap(lambda x: x.lower() if isinstance(x, str) else x)

        table = table[table['country_name'].isin(['united states', 'china'])]
        table = table[table['date'].isin([2022])]
        print(table)
    
    def test_retrieve_data_points(self):
        table = pd.read_csv(os.path.join(self.datasrc, "Social Protection & Labor.csv"))
        db = NeuralDB(
            tables=[table],
            add_row_id=False,
            normalize=False,
            lower_case=False
        )
        sql = """SELECT "country_name" FROM w WHERE "Employment to population ratio, 15+, total (%) (national estimate)" > 7  AND "date" = 2020 """
        print(db.execute_query(sql))

    def test_parse_ans(self):
        parser = AnsParser()
        # message = """SELECT "country" , "Annual total production-based emissions of carbon dioxide (CO₂), including land-use change (million tonnes)" FROM w WHERE "country" IS NOT null and "Annual total production-based emissions of carbon dioxide (CO₂), including land-use change (million tonnes)" IS NOT null and "Annual total production-based emissions of carbon dioxide (CO₂), including land-use change (million tonnes)" <> \'nan\'"""
        message = """country" , "Annual total production-based emissions of carbon dioxide (CO₂), including land-use change (million tonnes)"""
        # print(parser.parse_sql_unit(message))
        print(parser.parse_unit(message))

    def test_tag_date_time(self):
        import spacy
        query = "China had the best coal production against VAL_1 in 2011."
        if query.endswith(('.', '?')): # remove the period at the end
            query = query[:-1]
        # Load the spacy model
        nlp = spacy.load("en_core_web_sm")
        # Create a Doc object
        doc = nlp(query)
        # Dependency parsing
        for token in doc:
            print(token.text, token.pos_, token.ent_type_, token.dep_, token.head.text, token.head.pos_,
                [child for child in token.children])
            
        # start_default, end_default = 2010, 2020
        # dates = []
        # for ind, token in enumerate(doc):
        #     if token.ent_type_ == 'DATE' and token.head.pos_ in ['ADP', 'SCONJ']:
        #         dates.append(ind)
        # print(dates)
        
        # if len(dates) == 0: # add the most recent date
        #     query += f" in {end_default}"
        # elif len(dates) == 1: # rules
        #     ind = dates[0]-1
        #     if doc[ind].text.lower() in ['since', 'from']:
        #         query = f"{doc[:ind]} from {doc[ind+1]} to {end_default} {doc[ind+2:]}"
        #     elif doc[ind].text.lower() in ['til', 'until']:
        #         query = f"{doc[:ind]} from {start_default} {doc[ind:]}"
        
        print(query)
    
    def test_tag_attr_gpt(self):
        datasrc = "../Datasets/owid-co2.csv"
        autoviz = AutomatedViz(attributes=['year', 'Population of each country or region', 'country'], datasrc=datasrc)
        tag_map = autoviz.tag_attribute_gpt("Show the population of China and the USA in 2020.")
        print(tag_map)

    def test_llama_index(self):
        table = pd.read_csv(os.path.join(self.datasrc, "Private Sector.csv"))
        engine = PandasQueryEngine(table)
        response = engine.query("China has the highest Import value index (2000= 100).")
        print(response)

    def test_open_finetune_gpt(self):
        file_path = "../Gloc/generation/finetune/claim_tagging.jsonl"
        t = openai.File.create(
            file = open(file_path, 'rb'),
            purpose = 'fine-tune',
            user_provided_filename= "claim_tagging_fine_tune_3"
        )
        print(t)

    def test_list_files(self):
        files = openai.File.list()
        print(files)
    
    def test_create_fine_tune(self):
        # t = openai.FineTuningJob.create(
        #     training_file="file-nGHKtMTQqYdKfJU0uXRq06W9",
        #     model = 'gpt-3.5-turbo',
        #     hyperparameters = {
        #         "n_epochs": 4,
        #     }
        # )
        # print(t)
        t = openai.FineTuningJob.list()
        print(t)
    
    def test_infer_country(self):
        from TableReasoning import TableReasoner
        dm = DataMatcher(datasrc="../Datasets")
        tb = TableReasoner(datamatcher=dm)
        pass
    
    async def test_process_data(self):
        string = """Wealthy countries have been trying to boost their birthrates for decades. The results have been pretty similar.
        China's population has begun to decline, a demographic turning point for the country that has global implications. Experts had long anticipated this moment, but it arrived in 2022 several years earlier than expected, prompting hand-wringing among economists over the long-term impacts given the country's immense economic heft and its role as the world's manufacturer.
        With 850,000 fewer births than deaths last year, at least according to the country's official report, China joined an expanding set of nations with shrinking populations caused by years of falling fertility and often little or even negative net migration, a group that includes Italy, Greece and Russia, along with swaths of Eastern and Southern Europe and several Asian nations like South Korea and Japan.
        Even places that have not begun to lose population, such as Australia, France and Britain, have been grappling with demographic decline for years as life expectancy increases and women have fewer children.
        History suggests that once a country crosses the threshold of negative population growth, there is little that its government can do to reverse it. And as a country's population grows more top-heavy, a smaller, younger generation bears the increasing costs of caring for a larger, older one.
        Even though China's birthrate has fallen substantially over the last five decades, it was long a country with a relatively young population, which meant it could withstand those low rates for a long time before starting to see population losses. Like many developed countries, China's older population is now swelling — a consequence of its earlier boom — leaving it in a position similar to that of many wealthy nations: in need of more young people.
        Countries such as the U.S. and Germany have been able to rely on robust immigration, even with relatively low birthrates. But for countries with negative net migration, such as China, more people requires more babies.  
        "The good news is that the Chinese government is fully aware of the problem," said Yong Cai, a sociologist at the University of North Carolina, Chapel Hill, who specializes in Chinese demographics. "The bad news is, empirically speaking, there is very little they can do about it."
        That's because the playbook for boosting national birthrates is rather thin. Most initiatives that encourage families to have more children are expensive, and the results are often limited. Options include cash incentives for having babies, generous parental leave policies, and free or subsidized child care.
        Two decades ago, Australia tried a "baby bonus" program that paid the equivalent of nearly 6,000 U.S. dollars per child at its peak. At the time the campaign started in 2004, the country's fertility rate was around 1.8 children per woman. (For most developed nations, a fertility rate of 2.1 is the minimum needed for the population to remain steady without immigration.) By 2008, the rate had risen to a high of around 2, but by 2020, six years after the program had ended, it was at 1.6 — lower than when the cash payments were first introduced.
        By one estimate, the initiative led to an additional 24,000 births.
        Dr. Liz Allen, a demographer at the Australian National University, said that the program was largely ineffective and that publicly funded paternity leave and child care would have been a more effective use of taxpayer money. "Government intervention to increase fertility rates is best focused on addressing the issues that prevent people from having their desired family size," she said. Experts say the most effective initiatives address social welfare, employment policy, and other underlying economic issues. France, Germany, and Nordic countries like Sweden and Denmark have had notable success in arresting the decline in birthrates, often through government-funded child care or generous parental leave policies.
        But even the success of those efforts has had limits, with no country able to reach a sustained return to the 2.1 replacement rate. (The U.S. rate fell below 2.1 in the 1970s, slowly rose back up to the replacement rate by 2007, then collapsed again after the Great Recession, to a current level just below 1.7.)
        "You're not going to reverse the trend, but if you throw in the kitchen sink and make childbearing more attractive, you may be able to prevent the population from falling off a cliff," said John Bongaarts, a demographer at the Population Council, a research institution in New York.
        Sweden is often cited as a model for increasing fertility rates, thanks to a government-boosted jump in its birthrate. After introducing nine months of parental leave in the 1970s and implementing a "speed premium" in 1980 (which incentivized mothers to have multiple children within a set period), Sweden saw fertility rise from around 1.6 early in the decade to a peak just above the replacement rate by 1990. (The country has since increased its parental leave to 16 months, among the highest in the world.)
        After that uptick, however, Sweden's birthrate fell through the '90s. Over the last 50 years, its fertility rate has fluctuated significantly, rising roughly in tandem with economic booms. And while the country still has one of the highest fertility rates among the most advanced economies, over the past decade it has followed a trajectory similar to that of most developed nations: down.
        Recent research suggests a reason Sweden's fertility spikes were only temporary: Families rushed to have children they were already planning to have. Stuart Gietel-Basten, a demographer at the Hong Kong University of Science and Technology, said financial incentives seldom increase the overall number of children born, but instead encourage families to take advantage of benefits that may not last. The spikes, he added, can have unforeseen consequences. "When you have 50,000 children born one year, 100,000 the next, and then 50,000 the year after that, it is really bad for planning and education," he said.
        Few countries have embraced pronatalist policies as vigorously as Hungary, whose right-wing populist leader, Viktor Orban, is dedicating 5 percent of the nation's G.D.P. toward increasing birthrates. The government encourages procreation through generous loans that become gifts upon the birth of multiple children, tax forgiveness for mothers who have three children, and free fertility treatments.
        "You're not going to reverse the trend, but if you throw in the kitchen sink and make childbearing more attractive, you may be able to prevent the population from falling off a cliff," said John Bongaarts, a demographer at the Population Council, a research institution in New York.
        Sweden is often cited as a model for increasing fertility rates, thanks to a government-boosted jump in its birthrate. After introducing nine months of parental leave in the 1970s and implementing a "speed premium" in 1980 (which incentivized mothers to have multiple children within a set period), Sweden saw fertility rise from around 1.6 early in the decade to a peak just above the replacement rate by 1990. (The country has since increased its parental leave to 16 months, among the highest in the world.)
        After that uptick, however, Sweden's birthrate fell through the '90s. Over the last 50 years, its fertility rate has fluctuated significantly, rising roughly in tandem with economic booms. And while the country still has one of the highest fertility rates among the most advanced economies, over the past decade it has followed a trajectory similar to that of most developed nations: down.
        Recent research suggests a reason Sweden's fertility spikes were only temporary: Families rushed to have children they were already planning to have. Stuart Gietel-Basten, a demographer at the Hong Kong University of Science and Technology, said financial incentives seldom increase the overall number of children born, but instead encourage families to take advantage of benefits that may not last. The spikes, he added, can have unforeseen consequences. "When you have 50,000 children born one year, 100,000 the next, and then 50,000 the year after that, it is really bad for planning and education," he said.
        Few countries have embraced pronatalist policies as vigorously as Hungary, whose right-wing populist leader, Viktor Orban, is dedicating 5 percent of the nation's G.D.P. toward increasing birthrates. The government encourages procreation through generous loans that become gifts upon the birth of multiple children, tax forgiveness for mothers who have three children, and free fertility treatments.
        "You're not going to reverse the trend, but if you make childbearing more attractive, you may be able to prevent the population from falling off a cliff," said John Bongaarts, a demographer at the Population Council, a research institution in New York.
        Sweden is often cited as a model for increasing fertility rates, thanks to a government-boosted jump in its birthrate. After introducing nine months of parental leave in the 1970s and implementing a "speed premium" in 1980 (which incentivized mothers to have multiple children within a set period), Sweden saw fertility rise from around 1.6 early in the decade to a peak just above the replacement rate by 1990. (The country has since increased its parental leave to 16 months, among the highest in the world.)
        After that uptick, however, Sweden's birthrate fell through the '90s. Over the last 50 years, its fertility rate has fluctuated significantly, rising roughly in tandem with economic booms. And while the country still has one of the highest fertility rates among the most advanced economies, over the past decade it has followed a trajectory similar to that of most developed nations: down.
        Recent research suggests a reason Sweden's fertility spikes were only temporary: Families rushed to have children they were already planning to have. Stuart Gietel-Basten, a demographer at the Hong Kong University of Science and Technology, said financial incentives seldom increase the overall number of children born, but instead encourage families to take advantage of benefits that may not last. The spikes, he added, can have unforeseen consequences. "When you have 50,000 children born one year, 100,000 the next, and then 50,000 the year after that, it is really bad for planning and education," he said.
        Few countries have embraced pronatalist policies as vigorously as Hungary, whose right-wing populist leader, Viktor Orban, is dedicating 5 percent of the nation's G.D.P. toward increasing birthrates. The government encourages procreation through generous loans that become gifts upon the birth of multiple children, tax forgiveness for mothers who have three children, and free fertility treatments.
        Around the time these efforts began under Mr. Orban in 2010, Hungary's fertility rate was just over 1.2, among the lowest in Europe. Over the 2010s, that rate climbed to around 1.6 - a modest improvement at a high cost.
        It remains to be seen how far China will go to stem its decline in population, which was set in motion when the country's fertility rate began to plummet decades ago. That drop began even before the country's family planning policies limiting most families to a single child, introduced in 1979. Those who defied the rules were punished with fines and even forced abortions.
        The official end of Beijing's one-child policy in 2016, however, has not led to a rise in births, despite cash incentives and tax cuts for parents. The country's fertility rate rose slightly around that time, but has fallen since, according to data from the United Nations: from around 1.7 children per woman, on par with Australia and Britain, to around 1.2, among the lowest in the world. That recent drop could be a result of unreliable data from China or a technical effect of delays in childbearing, but it likely also reflects a combination of various pressures that have mounted in the country over time.
        Even though they are now allowed to, many young Chinese are not interested in having large families. Vastly more young Chinese people are enrolling in higher education, marrying later and having children later. Raised in single-child households, some have come to see small families as normal. But the bigger impediment to having a second or third child is financial, according to Lauren A. Johnston, an economist at the University of Sydney who studies Chinese demographics. She said many parents cite the high cost of housing and education as the main obstacle to having more children. "People can't afford to buy space for themselves, let alone for two kids," she said.
        Few countries have embraced pronatalist policies as vigorously as Hungary, whose right-wing populist leader, Viktor Orban, is dedicating 5 percent of the nation's G.D.P. toward increasing birthrates. The government encourages procreation through generous loans that become gifts upon the birth of multiple children, tax forgiveness for mothers who have three children, and free fertility treatments.
        Around the time these efforts began under Mr. Orban in 2010, Hungary's fertility rate was just over 1.2, among the lowest in Europe. Over the 2010s, that rate climbed to around 1.6 - a modest improvement at a high cost.
        It remains to be seen how far China will go to stem its decline in population, which was set in motion when the country's fertility rate began to plummet decades ago. That drop began even before the country's family planning policies limiting most families to a single child, introduced in 1979. Those who defied the rules were punished with fines and even forced abortions.
        The official end of Beijing's one-child policy in 2016, however, has not led to a rise in births, despite cash incentives and tax cuts for parents. The country's fertility rate rose slightly around that time, but has fallen since, according to data from the United Nations: from around 1.7 children per woman, on par with Australia and Britain, to around 1.2, among the lowest in the world. That recent drop could be a result of unreliable data from China or a technical effect of delays in childbearing, but it likely also reflects a combination of various pressures that have mounted in the country over time.
        Even though they are now allowed to, many young Chinese are not interested in having large families. Vastly more young Chinese people are enrolling in higher education, marrying later and having children later. Raised in single-child households, some have come to see small families as normal. But the bigger impediment to having a second or third child is financial, according to Lauren A. Johnston, an economist at the University of Sydney who studies Chinese demographics. She said many parents cite the high cost of housing and education as the main obstacle to having more children. "People can't afford to buy space for themselves, let alone for two kids," she said.
        Few countries have embraced pro-natalist policies as vigorously as Hungary, whose right-wing populist leader, Viktor Orban, is dedicating 5 percent of the nation's GDP toward increasing birthrates. The government encourages procreation through generous loans that become gifts upon the birth of multiple children, tax forgiveness for mothers who have three children, and free fertility treatments.
        Around the time these efforts began under Mr. Orban in 2010, Hungary's fertility rate was just over 1.2, among the lowest in Europe. Over the 2010s, that rate climbed to around 1.6 - a modest improvement at a high cost.
        It remains to be seen how far China will go to stem its decline in population, which was set in motion when the country's fertility rate began to plummet decades ago. That drop began even before the country's family planning policies limiting most families to a single child, introduced in 1979. Those who defied the rules were punished with fines and even forced abortions.
        The official end of Beijing's one-child policy in 2016, however, has not led to a rise in births, despite cash incentives and tax cuts for parents. The country's fertility rate rose slightly around that time, but has fallen since, according to data from the United Nations: from around 1.7 children per woman, on par with Australia and Britain, to around 1.2, among the lowest in the world. That recent drop could be a result of unreliable data from China or a technical effect of delays in childbearing, but it likely also reflects a combination of various pressures that have mounted in the country over time.
        Even though they are now allowed to, many young Chinese are not interested in having large families. Vastly more young Chinese people are enrolling in higher education, marrying later and having children later. Raised in single-child households, some have come to see small families as normal. But the bigger impediment to having a second or third child is financial, according to Lauren A. Johnston, an economist at the University of Sydney who studies Chinese demographics. She said many parents cite the high cost of housing and education as the main obstacle to having more children. "People can't afford to buy space for themselves, let alone for two kids," she said.
        China's government could ease the burden on young families through housing subsidies, extended parental leave and increased funding for education and pensions, experts say. Other policy changes, like reforming the country's restrictive household registration system and raising the official retirement age — female blue-collar workers must retire at 50, for example — could boost the nation's working-age population, alleviating some of the economic strain that comes with population decline.
        Though the Chinese are unlikely to find more success than the Swedes in recovering a high fertility rate, "there is low-hanging fruit that can allow them to squeeze more productivity and higher labor force participation from the population," said Gerard DiPippo, a senior fellow at the Center for Strategic and International Studies. All this points to a Chinese population, currently 1.4 billion, that is likely to continue shrinking. In contrast to economists who have cast China's population decline as a grim sign for global growth, many demographers have been more sanguine, noting the benefits of a smaller population.
        John Wilmoth, director of the Population Division at the United Nations, said that after decades of exponential growth in which the world's population doubled to more than seven billion between 1970 to 2014, the doom-and-gloom assessments about declining fertility rates and depopulation tend to be overstated. Japan has been battling population decline since the 1970s, he noted, but it remains one of the world's largest economies. "It has not been the disaster that people imagined," Mr. Wilmoth said. "Japan is not in a death spiral."
        Worldwide, fertility remains above the replacement rate, which means that allowing more immigration will continue to be an option for many developed nations, even those that historically haven't relied on it: Before the pandemic, net migration into Japan, while relatively low, had been increasing steadily.
        Without immigration, pragmatic and non-coercive measures that encourage parents to have families while pursuing careers — as well as policies that allow people in their 60s and 70s to keep working — are the key to managing negative population growth, Mr. Wilmoth said. "Population stabilization is overall a good thing," he said. "All societies need to adapt to having older populations. What really matters is the speed of change, and how fast we get from here to there." """


        async def process_sentence(sentence, paragraph):
            claim, score = await dt.detect_2(sentence, score_threshold=THRE_SHOLD, llm_classify=True)
            if score > THRE_SHOLD:
                print(claim, score)
                return {
                    "paragraph": paragraph,
                    "claim": claim,
                    "score": score
                }
            return None

        import nltk
        dt = ClaimDetector()
        paragraphs = string.split("\n")
        tasks, THRE_SHOLD = [], .5
        for paragraph in paragraphs:
            sentences = nltk.sent_tokenize(paragraph)
            for sentence in sentences:
                task = asyncio.create_task(process_sentence(sentence, paragraph))
                tasks.append(task)
        claims = await asyncio.gather(*tasks)
        claims = [claim for claim in claims if claim is not None]
        with open("claims_1.json", "w") as f:
            json.dump(claims, f, indent=4)
        print(claims)
        
    async def test_arrange_article(self):
        with open("claims_2.json", "r") as f:
            claims = json.load(f)
            ans = {
                "text": [],
                "sentences": [],
                "url": "https://www.nytimes.com/2023/02/09/upshot/china-population-decline.html?unlocked_article_code=ZJe01vsF8q8o0tGVjIawT7oDvY2MT5KpRX3t9NpEsVvOqWVn1bUM0ZnGen_UVO2bMYVvJHLwrqRDIdRXd9eDt8gKc3k55Q8zSEhgAj6ZIXQxCgQvsqblkANQObymBFajH4lYg1SJOJx0CEgP1YSbPVgDN-buekUI5bK6VcMlStncXtjWxZQz-dV78mViLkjICZcyMCgz6Z2TfspMii4PWMyU5mvtFXN5SrgNnk_x_Ut4-z4LTYqi40jjzkExqA0A6oeq44qsi6PwLN-TCm5H7vc63EYRXevjAs0AsNHTGEmj2Xh9nSo5dX09pFhCnw6WlYuNQMFCpg-rFDvJNlKYvkA3&amp;smid=url-share",
                "type": "article",
                "title": "Can China Reverse Its Population Decline? Just Ask Sweden."
            }

            for claim in claims:
                if claim["paragraph"] not in ans["text"]:
                    ans["text"].append(claim["paragraph"])
                ans["sentences"].append(claim["claim"])
            
            with open("claims_2.json", "w") as f:
                json.dump(ans, f, indent=4)
            
    async def get_suggested_queries_and_write(self, claim, model, ind, lock):
        async with lock:
            with open('./processed_files/Recommendations.json', 'r+') as f:
                data = json.load(f)
                if str(ind) not in data:
                    result = await get_suggested_queries(claim, model)
                    data[str(ind)] = result['claim_map'].to_json()
                    f.seek(0)
                    json.dump(data, f)
                    f.truncate()
                else:
                    print(f"Already processed {ind}")

    async def test_recommend(self, n=1, rset=None):
        filename = "./processed_files/Pipeline Evaluation - Ground Truth - Datasets.csv"
        df = pd.read_csv(filename)
        lock = asyncio.Lock()
        sem = asyncio.Semaphore(n)  # Limit to n concurrent tasks
        tasks = []
        for ind, row in df.iterrows():
            if rset and ind not in rset:
                continue
            if pd.notna(row["Sentence"]):
                sentence = row["Sentence"]
            else: continue
            if pd.notna(row["Comment"]):
                paragraph = row["Comment"]
            else: continue
            claim = UserClaimBody(userClaim=sentence, paragraph=paragraph)
            task = asyncio.create_task(self.bounded_get_suggested_queries_and_write(claim, Model.GPT_TAG_4, ind, lock, sem))
            tasks.append(task)
        await asyncio.gather(*tasks)

    async def bounded_get_suggested_queries_and_write(self, claim, model, ind, lock, sem):
        async with sem:  # This will block if there are already n running tasks
            await self.get_suggested_queries_and_write(claim, model, ind, lock)



    async def test_potential_datapoints(self, n=2, rset=None):
        recfilename = "./processed_files/Recommendations.json"
        recdf = json.load(open(recfilename, 'r'))
        lock = asyncio.Lock()
        tasks = []
        for ind, claim_map in recdf.items():
            if rset and int(ind) not in rset:
                continue
            claim_map = ClaimMap(**claim_map)
            task = asyncio.create_task(self.get_potential_datapoints_and_write(claim_map, ind, lock))
            tasks.append(task)
        await asyncio.gather(*tasks)
    
    async def find_top_k_datasets(self, claim_map: ClaimMap, verbose=True):
        value_keywords = [keyword for sublist in claim_map.suggestion for keyword in sublist.values if sublist.field == "value" or keyword.startswith("@(")]
        country_keywords = [keyword[2:-2].replace("Country", "").replace("Countries", "").strip() for keyword in claim_map.country if keyword.startswith("@(")]
        keywords = country_keywords + claim_map.value + value_keywords
        print("keywords:", keywords)
        top_k_datasets = await self.dm.find_top_k_datasets("", k=8, method="gpt", verbose=verbose, keywords=keywords)
        datasets = [Dataset(name=name, description=description, score=score, fields=fields) 
            for name, description, score, fields in top_k_datasets]
        return datasets
    
    async def get_potential_datapoints_and_write(self, claim_map: ClaimMap, ind, lock):
        async with lock:
            try:
                with open('./processed_files/DatapointSets.json', 'r+') as f:
                    data = json.load(f)
                    if str(ind) not in data:
                        datasets = await self.find_top_k_datasets(claim_map, verbose=True)
                        result = await potential_data_point_sets_2(claim_map, datasets, True)
                        data[str(ind)] = [result[0].to_json()]
                        f.seek(0)
                        json.dump(data, f)
                        f.truncate()
                    else:
                        print(f"Already processed {ind}")
            except Exception as e:
                print("Super error: ", e)
        
    async def merge_datapoints_recommend(self, rset=None):
        df = pd.read_csv("./processed_files/Pipeline Evaluation - Ground Truth - Datasets.csv")
        datapoint_df = json.load(open("./processed_files/DatapointSets.json", 'r'))
        rec_df = json.load(open("./processed_files/Recommendations.json", 'r'))
        merged_df = []
        for ind, row in df.iterrows():
            if not pd.notna(row["Sentence"]) or not pd.notna(row["Comment"]):
                continue
            if rset and ind not in rset:
                continue
            idx = str(ind)
            merged_df.append({
                "snippet": {
                    "sentences": [row["Sentence"]],
                    "text": [row["Comment"]],
                    "url": row["Link"],
                    "title": None,
                    "type": row["Type"],
                    "header": None,
                },
                "claimMap": rec_df[idx] if idx in rec_df else {},
                "dataPointSet": datapoint_df[idx] if idx in datapoint_df else {}
            })
        
        with open("./processed_files/Merged.json", "w") as f:
            json.dump(merged_df, f, indent=4)
            
    async def play_around(self, write=False):
        df = pd.read_csv("./processed_files/Pipeline Evaluation - Ground Truth - Datasets.csv")
        # Exploratory analysis on the topic
        topics = df['Topic'].value_counts()
        print("Topics distribution:\n", topics)
        
        # Task: Visualize the distribution of topics
        async def visualize_topic_distribution(self, topics):
            topics.plot(kind='bar')
            plt.title('Distribution of Topics')
            plt.xlabel('Topics')
            plt.ylabel('Count')
            plt.show()
        vis_task = asyncio.create_task(visualize_topic_distribution(self, topics))

        # check if merged files include or exclude correctly
        with open("./fresh files/Merged.json", 'r') as f:
            mf = json.load(f)
            sents = set(df['Sentence'].tolist())
            # count how many snippets are included\
            count, removed = 0, []
            for idx, item in reversed(list(enumerate(mf))):
                sent = item['snippet']['sentences'][0]
                if sent in sents:
                    count += 1
                    # add title
                    item['snippet']['title'] = df[df['Sentence'] == sent]['Title'].tolist()[0]
                else:
                    removed.append(sent)
                    mf.pop(idx)
                    
            print("Count of snippets included:", count)
            print("Count of total snippets:", len(df['Sentence'].tolist()))
            print("Removed snippets:", removed)

            # write to file
            if write:
                with open("./fresh files/Merged.json", 'w') as f:
                    json.dump(mf, f, indent=4)
            
    async def add_index(self, write=False):
        with open("./fresh files/Merged.json", 'r') as f:
            mf = json.load(f)
            for idx, item in enumerate(mf):
                item['id'] = idx
            
            if write:
                with open("./fresh files/Merged.json", 'w') as f:
                    json.dump(mf, f, indent=4)
    
    async def statistics(self):
        import sqlite3
        conn = sqlite3.connect('database/sql_app.db')
        cursor = conn.cursor()
        table_name = "logs"
        cursor.execute("SELECT * FROM {}".format(table_name))
        rows = cursor.fetchall()
        
        event_column = [row[1] for row in rows]  # assuming 'event' is the column name
        unique_events = set(event_column)
        print(f"Unique events: {unique_events}")
        
        # Count occurrences of each event
        event_counts = {event: event_column.count(event) for event in unique_events}
        print(f"Event counts: {event_counts}")

        async def plot_event_counts(event_counts):
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _plot_event_counts, event_counts)

        def _plot_event_counts(event_counts):
            plt.barh(list(event_counts.keys()), list(event_counts.values()))
            plt.tick_params(axis='y', labelsize=5)  # Adjust the size of the y-axis labels
            plt.xlabel('Event')
            plt.ylabel('Count')
            plt.title('Event counts')
            plt.show()
       
        plot_task = asyncio.create_task(plot_event_counts(event_counts))
        await plot_task

async def main():
    tester = Tester(datasrc="../Datasets")
    # rset = [57, 14, 72]
    # # await tester.test_recommend(rset=rset, n=2)
    # # await tester.test_potential_datapoints(rset=rset, n=2)
    # await tester.add_index(write=True)
    await tester.statistics()

if __name__ == "__main__":
    asyncio.run(main())
    # pass