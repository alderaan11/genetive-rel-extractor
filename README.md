# genetive-rel-extractor


This project explores **automatic rule extraction for determining semantic relation types in French genitive constructions**.  
It combines **large language models, lexical resources, and machine learning techniques** to build a pipeline from corpus creation to classification.

## Main Contributions

- **Corpus generation with ChatGPT**  
  Synthetic French sentences with genitive constructions are generated to ensure sufficient data coverage and balanced semantic relations.

- **Semantic signature extraction with JeuxDeMots**  
  For each lexical unit in the corpus, semantic signatures are retrieved and enriched using the JeuxDeMots lexical network.  
  These signatures capture semantic traits that guide the relation typing.

- **Information encoding**  
  Semantic and syntactic features are encoded into machine-readable representations suitable for learning and inference.  
  This step bridges raw linguistic information and classification models.

- **Relation type classification**  
  Classifiers are trained to predict the semantic relation (e.g. possession, part-whole, origin, quantity, etc.) expressed in the genitive constructions.  
  Both rule-based and learning-based approaches are considered.

## Objectives

- Build a **reproducible pipeline** for studying semantic relations in French genitives.  
- Investigate the synergy between **LLM-based corpus generation**, **lexical semantic networks**, and **automatic classification methods**.  
- Provide a **benchmark corpus and evaluation framework** for future research on French semantic parsing and relation extraction.

