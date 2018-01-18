# Multilingual Training of Crosslingual Word Embeddings

Long Duong et al.

Core is to map multiple languages into the same vector space without any direct bilingual signal between them. The paper proposes several algorithm in training or post-processing, like applying linear transformation or modifying the objective function to jointing build embedding. 



The base model for bilingual embedding is an extension of CBOW with negative sampling. Duong et al. takes not only the middle word but also its translation in other languages. 







### Thoughts 