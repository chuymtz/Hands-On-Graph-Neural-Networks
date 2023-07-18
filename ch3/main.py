import numpy as np
from pprint import pprint

CONTEXT_SIZE = 2
text = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
Nunc eu sem scelerisque, dictum eros aliquam, accumsan quam. 
Pellentesque tempus, lorem ut semper fermentum, ante turpis accumsan ex, sit amet ultricies tortor erat quis nulla. 
Nunc consectetur ligula sit amet purus porttitor, vel tempus tortor scelerisque. 
Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; 
Quisque suscipit ligula nec faucibus accumsan. Duis vulputate massa sit amet viverra hendrerit. 
Integer maximus quis sapien id convallis. Donec elementum placerat ex laoreet gravida. 
Praesent quis enim facilisis, bibendum est nec, pharetra ex. 
Etiam pharetra congue justo, eget imperdiet diam varius non. 
Mauris dolor lectus, interdum in laoreet quis, faucibus vitae velit. 
Donec lacinia dui eget maximus cursus. 
Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. 
Vivamus tincidunt velit eget nisi ornare convallis. 
Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. 
Donec tristique ultrices tortor at accumsan.
""".split()

text[:12]

skipgrams = []
for i in range(CONTEXT_SIZE, len(text) - CONTEXT_SIZE):
    # grab words around the target word text[i] but skip the ith
    array = [text[j] for j in np.arange(i - CONTEXT_SIZE, i + CONTEXT_SIZE + 1 ) if j != i]
    # text[i] is the target word is the input and we want to predict the array 
    skipgrams.append((text[i], array))

pprint(skipgrams)


vocab = set(text)
VOCAB_SIZE = len(vocab)
print(f"{VOCAB_SIZE = }")

# the dimension of the embedding is defined by the user














