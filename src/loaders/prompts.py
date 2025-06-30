
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