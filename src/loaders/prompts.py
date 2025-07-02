

###### Prompts for labeling datasets with specific concepts ######

CEBAB_LABELING_PROMPT = """
In a dataset of restaurant reviews there are 4 possible concepts: 

- Good Food,
- Good Ambiance,
- Good Service,
- Good Noise. 

Given a certain review, you have to detect if those concepts are present or not in the review.

Answer format: 
Good Food: score, Good Ambiance: score, Good Service: score, Good Noise: score. 

Do not add any text other than that specified by the answer format. 
The score should be equal to 1 if the concept is present or zero otherwise, no other values are accepted.

The following are examples:

Review: "The food was delicious and the service fantastic".
Answer: Good Food: 1, Good Ambiance: 0, Good Service: 1, Good Noise: 0

Review: "The staff was very rough but the restaurant decorations were great. Other than that there was a very relaxing background music".
Answer: Good Food: 0, Good Ambiance: 1, Good Service: 0, Good Noise: 1

Now it's your turn: 

Review: <review>       
Answer:
"""

IMDB_LABELING_PROMPT = """
In a dataset of film reviews (IMDb), there are 4 possible concepts: 

- Good Acting,
- Good Storyline,
- Good Emotional Arousal,
- Good Cinematography.

Given a certain review, you have to detect if those concepts are present or not in the review.

Answer format: 
Good Acting: score, Good Storyline: score, Good Emotional Arousal: score, Good Cinematography: score. 

Do not add any text other than that specified by the answer format. 
The score should be equal to 1 if the concept is present and zero if , no other values are accepted.

The following are examples:

Review: "The performances were outstanding, especially the lead actor. The story dragged in the middle though."
Answer: Good Acting: 1, Good Storyline: 0, Good Emotional Arousal: 0, Good Cinematography: 0

Review: "This film moved me to tears. The plot was very touching, and the visual effects were just stunning."
Answer: Good Acting: 0, Good Storyline: 1, Good Emotional Arousal: 1, Good Cinematography: 1

Now it's your turn:

Review: <review>       
Answer:
"""

TREC_LABELING_PROMPT = """
In a dataset of questions, there are 6 possible concepts: 

- Definition Request,
- Person Entity,
- Location Reference,
- Numeric Expectation,
- Abbreviation or Acronym,
- Object Reference.

Given a certain question, you have to detect if those concepts are present or not in the question.

Answer format: 
Definition Request: score, Person Entity: score, Location Reference: score, Numeric Expectation: score, Abbreviation or Acronym: score, Object Reference: score.

Do not add any text other than that specified by the answer format. 
The score should be equal to 1 if the concept is present or zero otherwise, no other values are accepted.

The following are examples:

Question: "What is the capital of France?"
Answer: Definition Request: 0, Person Entity: 0, Location Reference: 1, Numeric Expectation: 0, Abbreviation or Acronym: 0, Object Reference: 0

Question: "Who discovered penicillin?"
Answer: Definition Request: 0, Person Entity: 1, Location Reference: 0, Numeric Expectation: 0, Abbreviation or Acronym: 0, Object Reference: 0

Now it's your turn:

Question: <review>
Answer:
"""

WOS_LABELING_PROMPT = """
In a dataset of scientific paper abstracts, there are 7 possible concepts:

- Mentions statistical methods,
- Focus on human subjects,
- Uses neural networks,
- Mentions DNA/genetics,
- Refers to electrical components,
- Mentions software systems,
- Uses clinical trial terminology.

Given a certain abstract, you have to detect if those concepts are present or not in the abstract.

Answer format:
Mentions statistical methods: score, Focus on human subjects: score, Uses neural networks: score, Mentions DNA/genetics: score, Refers to electrical components: score, Mentions software systems: score, Uses clinical trial terminology: score.

Do not add any text other than that specified by the answer format.
The score should be equal to 1 if the concept is present and zero otherwise. No other values are accepted.

The following are examples:

Abstract: "This study develops a neural network model to predict disease progression using clinical trial data."
Answer: Mentions statistical methods: 0, Focus on human subjects: 1, Uses neural networks: 1, Mentions DNA/genetics: 0, Refers to electrical components: 0, Mentions software systems: 0, Uses clinical trial terminology: 1

Abstract: "We propose a software system that leverages statistical inference to optimize electrical component performance."
Answer: Mentions statistical methods: 1, Focus on human subjects: 0, Uses neural networks: 0, Mentions DNA/genetics: 0, Refers to electrical components: 1, Mentions software systems: 1, Uses clinical trial terminology: 0

Now it's your turn:

Abstract: <review>
Answer:
"""

CLINC_LABELING_PROMPT = """
You are given a user query to a task-oriented dialog system. The system supports multiple domains and intents, but some queries may be out-of-scope (OOS), meaning they do not fall into any supported intent.

Your task is to detect the presence or absence of the following concepts in the query. For each concept, answer with a score of 1 if the concept is present, or 0 if it is absent. Do not add any text other than the answer format.

Concepts:
- Domain Mention: Does the query explicitly mention or imply a supported domain or topic?
- Intent Specific Keywords: Does the query contain keywords or phrases related to any specific intent?
- Action Request: Does the query ask to perform an action or service?
- Out-of-Scope Indicators: Does the query contain terms or topics unrelated to any supported domain or intent, indicating it is out-of-scope?

Answer format:
Domain Mention: score, Intent Specific Keywords: score, Action Request: score, Out-of-Scope Indicators: score

Examples:

Query: "Can you help me book a flight to New York?"
Answer: Domain Mention: 1, Intent Specific Keywords: 1, Action Request: 1, Out-of-Scope Indicators: 0

Query: "What's the capital of France?"
Answer: Domain Mention: 0, Intent Specific Keywords: 0, Action Request: 0, Out-of-Scope Indicators: 1

Now it's your turn:

Query: <review>
Answer:

"""

BANK_LABELING_PROMPT = """
In a dataset of user queries related to banking and financial services, there are 4 possible concepts:

- Transaction Mention
- Issue/Problem Description
- Account Reference
- Request for Help or Clarification

Given a user query, you have to detect if each of these concepts is present or not in the query.

Answer format:  
Transaction Mention: score, Issue/Problem Description: score, Account Reference: score, Request for Help or Clarification: score.

Do not add any text other than that specified by the answer format.  
The score should be 1 if the concept is present or 0 otherwise. No other values are accepted.

The following are examples:

Query: "A card payment on my account is shown as pending."  
Answer: Transaction Mention: 1, Issue/Problem Description: 1, Account Reference: 1, Request for Help or Clarification: 0

Query: "I can't seem to make a standard bank transfer. I have tried at least five times already but none of them are going through. Please tell me what is wrong?"  
Answer: Transaction Mention: 1, Issue/Problem Description: 1, Account Reference: 0, Request for Help or Clarification: 1

Now it's your turn:

Query: <review>  
Answer:
"""


###### Prompts Zero-Shot Classification ######