
CEBAB_LABELING_PROMPT = """
In a dataset of restaurant reviews there are 4 possible concepts: 

- concept_1: Good Food,
- concept_2: Good Ambiance,
- concept_3: Good Service,
- concept_4: Good Noise. 

Given a certain review, you have to detect if those concepts are present or not in the review.

Answer format: 
concept_1: score, concept_2: score, concept_3: score, concept_4: score. 

Do not add any text other than that specified by the answer format. 
The score should be equal to 1 if the concept is present or zero otherwise,
no other values are accepted.

The following are examples:

Review: "The food was delicious and the service fantastic".
Answer: concept_1: 1, concept_2: 0, concept_3: 1, concept_4: 0

Review: "The staff was very rough but the restaurant decorations were great. Other than that there was a very relaxing background music".
Answer: concept_1: 0, concept_2: 1, concept_3: 0, concept_4: 1

Now it's your turn: 

Review: <review>       
Answer:
"""

IMDB_LABELING_PROMPT = """

"""