query_generation = [
    {
        "role": "system",
        "content": """From a claim, suggest me a list of at max 3 suggestions, each containing:
        1. a query that can verified using a database.
        2. a visualization task that is consistent with the query."""
    },
    {
        "role": "user",
        "content": "statement: The number of COVID in US has dwindled to below 2000 cases per day in June 26, 2023."
    },
    {
        "role": "assistant",
        "content": """Here are some suggestions:
        [
            {
                query: "Is the number of COVID cases in US below 2000 per day in 26th June 2023?",
                vis: "Show the number of COVID cases in US per day in 26th June 2023."
                explain: "Showing the number of COVID cases in US per day in 26th June 2023 helps to verify the claim."
            }
        ]"""
    },

    {
        "role": "user",
        "content": "statement: US'economy has outgrown China's in 2023."
    },
    {
        "role": "assistant",
        "content": """Here are some suggestions:
        [
            {
                query: "Is the GDP of US greater than China in 2023?",
                vis: "Show the GDP of US and China in 2023.",
                explain: "Since GDP is a quantitative metric that measures economy, \
                    showing the GDP of US and China in 2023 helps to verify the claim."
            },
            {
                query: "Is the Trade Volume of US greater than China in 2023?",
                vis: "Show the Trade Volume of US and China in 2023.",
                explain: "Trade Volume is also a good economic metric, showing it could give \
                more information to verify the claim."
            }
        ]"""
    },

    {
        "role": "user",
        "content": "statement: US'economy outweighs the whole South Asia's."
    },
    {
        "role": "assistant",
        "content": """Here are some suggestions:
        [
            {
                query: "Is the GDP of US greater than the sum of GDP of all South Asian countries in 2023?",
                vis: "Show the GDP of US and all South Asian countries in 2023.",
                explain: "Since GDP is a quantitative metric that measures economy, \
                    showing the GDP of US and all South Asian countries in 2023 helps to verify the claim."
            }
        ]"""
    },

    {
        "role": "user",
        "content": "statement: All countries in East Asia produce less than 200 kWh of electricity per capita per day in 2019."
    },
    {
        "role": "assistant",
        "content": """Here are some suggestions:
        [
            {
                query: "Is the electricity production per capita of all East Asian countries less than 200 kWh per day in 2019?",
                vis: "Show the electricity production per capita per day of all East Asian countries in 2019.",
                explain: "Showing the electricity production per capita per day of all East Asian countries in 2019 helps to verify the claim."
            }
        ]"""
    },

    {
        "role": "user",
        "content": "statement: As the temperature in China rises, the number of COVID cases in China decreases."
    },
    {
        "role": "assistant",
        "content": """Here are some suggestions:
        [
            {
                query: "Is the temperature in China and the number of COVID cases in China negatively correlated?",
                vis: "Show the relationship between the temperature in China and the number of COVID cases in China.",
                explain: "Showing the relationship between the temperature in China and the number of COVID cases in China helps to verify the claim."
            }
        ]"""
    },

    {
        "role": "user",
        "content": "statement: The movie production of US is better than that of China in 2023."
    },
    {
        "role": "assistant",
        "content": """Here are some suggestions:
        [
            {
                query: "Is the total box office of US greater than China in 2023?",
                vis: "Show the total box office of US and China in 2023.",
                explain: "Showing the total box office of US and China in 2023 helps to give more information to verify the claim."
            },
            {
                query: "Is the number of movies produced in US greater than China in 2023?",
                vis: "Show the number of movies produced in US and China in 2023.",
                explain: "Showing the number of movies produced in US and China in 2023 helps to give more information to verify the claim."
            }
        ]"""
    },
]

dec_reasoning_2 = [
    {
        "role": "system",
        "content": "You will be given a table and a question associated with them. Try to break the question down and answer them step by step."
    },
    {
        "role": "system",
        "content": "/*\ntable caption : jeep grand cherokee.\ncol : years | displacement | engine | power | torque\nrow 1 : 1999 - 2004 | 4.0l (242cid) | power tech i6 | - | 3000 rpm\nrow 2 : 1999 - 2004 | 4.7l (287cid) | powertech v8 | - | 3200 rpm\nrow 3 : 2002 - 2004 | 4.7l (287cid) | high output powertech v8 | - | -\nrow 4 : 1999 - 2001 | 3.1l diesel | 531 ohv diesel i5 | - | -\nrow 5 : 2002 - 2004 | 2.7l diesel | om647 diesel i5 | - | -\n*/\nquestion : Does the powertech v8 engine have a torque of 6000rpm?"
    },
    {
        "role": "user",
        "content": "Q1: What is torque of the powertech v8 engine?"
    },
    {
        "role": "assistant",
        "content": "A1: Let's find the column and row of powertech v8. Then let's extract the torque, it's 3200 rpm. The anwer is 3200."
    },
    {
        "role": "user",
        "content": "Q2: Does the powertech v8 engine have a torque of 6000 rpm?"
    },
    {
        "role": "assistant",
        "content": "A2: Since the torque of powertech v8 is 3200 rpm, and (3200 != 6000). The answer is no."
    },

    {
        "role": "system",
        "content": "/*\ncol: Year | Democrats | Republicans | Independents\nrow 1: 2004 | 68.1% | 45.0% | 53.0%\nrow 2: 2006 | 58.0% | 42.0% | 53.0%\nrow 3: 2007 | 59.0% | 38.0% | 45.0%\nrow 4: 2009 | 72.0% | 49.0% | 60.0%\nrow 5: 2011 | 71.0% | 51.2% | 58.0%\nrow 6: 2012 | 70.0% | 48.0% | 53.0%\nrow 7: 2013 | 72.0% | 41.0% | 60.0%\n*/\nquestion: Do republicans' and democrats' lowest vote rate have a wide difference by more than 30%??"
    },
    {
        "role": "user",
        "content": """Q1: What is the lowest rate of republicans?
        Q2: What is the lowest rate of democrats?"""
    },
    {
        "role": "assistant",
        "content": """A1: Let's find the column of republicans. Then let's extract the favor rates, they are [45.0, 42.0, 38.0, 49.0, 51.2, 48.0, 41.0]. The smallest number is 38.0. The answer is 38.0.
        A2: Let's find the column of democrats. Then let's extract the favor rates, they are [68.1, 58.0, 59.0, 72.0, 71.0, 70.0, 72.0]. The smallest number is 58.0. The answer is 58.0."""
    },
    {
        "role": "user",
        "content": "Q3: Do republicans' and democrats' lowest vote rate have a wide difference by more than 30%?"
    },
    {
        "role": "assistant",
        "content": "A3: Since Republicans' lowest favor rate is 38.0%, Democrats' lowest favor rate is 58.0%, and (58.0 - 38.0 < 30), the difference must be less than 30. The answer is no."
    },

    {
        "role": "system",
        "content": "/*\ntable_caption: south wales derby,\ncol : competition | total matches | cardiff win | draw | swansea win\nrow1 : league | 55 | 19 | 16 | 20\nrow2 : fa cup | 2 | 0 | 27 | 2\nrow3 : league cup | 5 | 2 | 0 | 3\n*/\n question : Cardiff wins in fa cup more than Swansea does in all competitions?"
    },
    {
        "role": "user",
        "content": """Q1: What is the cardiff win in fa cup?
        Q2: What is the swansea total win in all competitions?"""
    },
    {
        "role": "assistant",
        "content": """A1: Let's find the row of fa cup. Then let's extract the cardiff win, it's 0. The answer is 0.
        A2: Let's find Swansea's win in all competitions, it's [20, 2, 3]. Then let's sum them to obtain the total win, it's 25. The answer is 25."""
    },
    {
        "role": "user",
        "content": "Q3: Does Cardiff win in fa cup more than Swansea does in all competitions?"
    },
    {
        "role": "assistant",
        "content": "A3: Since Cardiff's win in fa cup is 0, Swansea's total win in all competitions is 25, and (0 < 25), the answer is no."
    }
]

import json
# save pro to query_generation_2.json
with open("fewshots/dec_reasoning_2.json", "w") as f:
    json.dump(dec_reasoning_2, f, indent=4)