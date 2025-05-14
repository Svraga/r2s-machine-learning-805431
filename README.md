# Rosetta Stone  
**Brandetti Claudia, Mammetti Francesco, Sarcina Daniele**  

# Libraries used
* Counter: used to count word frequencies - https://docs.python.org/3/library/collections.html#counter-objects
* pandas: for handling and processing tabular data - https://pandas.pydata.org/docs/
* numpy: for numerical operations and array handling - https://numpy.org/doc/
* torch: PyTorch, used for building and training neural networks - https://docs.pytorch.org/docs/stable/index.html
* tqdm: adds progress bars to loops (e.g., during training) - https://tqdm.github.io/
* warnings: to suppress or handle warning messages - https://docs.python.org/3/library/warnings.html
* fasttext: library for efficient word embeddings and language detection - https://fasttext.cc/docs/en/supervised-tutorial.html
* nltk: natural language processing tools (e.g., tokenization, stopwords) - https://www.nltk.org/
* sklearn: scikit-learn, used here for data splitting and evaluation metrics - https://scikit-learn.org/stable/
* random: for random number generation and reproducibility - https://docs.python.org/3/library/random.html

### Transformers
* BertTokenizer, BertModel: BERT tokenizer and model used for sentence encoding
* MarianMTModel, MarianTokenizer: machine translation model/tokenizer from HuggingFace for back-translation
* AutoTokenizer, AutoModelForCausalLM: generic tokenizer and causal language model interface from HuggingFace
* SentenceTransformer (SBert): to calculate the similarity score for Japanese

More in-depth explanation of how the transfomers have been used can be found later on.

## INTRODUCTION  
The goal of the project is to develop a model for semantic similarity between two sentences in different languages, trained on the augmented multilingual dataset from the Rosetta Stone.  

## METHODS  
The provided dataset includes eleven languages: English (en), Spanish (es), French (fr), Italian (it), Japanese (ja), Dutch (nl), Polish (pl), Portuguese (pt), Russian (ru), Chinese (zh), and German (de). It consists of 959,080 rows and 5 columns: `'sentence1'`, `'sentence2'`, `'lang1'`, `'lang2'`, and `'similarity score'`, where sentences in each language are translated into all other languages. Some sentences share the same meaning, resulting in a high similarity score, while others differ and therefore have a low score.  

We divided the process into three phases: **Data Processing**, **Data Augmentation**, and **Model Development and Evaluation**.  

#### 1. Data Processing  
The first step was cleaning the dataset. Specifically, we checked for missing values (NA) or empty sentences, but none were found. We then removed duplicates, totaling 8,542 rows (~0.9% of the dataset). Next, we normalized the sentences by:  
- Removing leading/trailing whitespaces  
- Replacing multiple spaces and tabs  
- Converting text to lowercase  
- Removing commas  

The score column was converted to a numeric format, with invalid values set to `NaN`. Scores were filtered to ensure they fell within a defined range (0 to 5).  

The resulting dataset, named `clean_df`, consists of **940,538 rows** and the **5 columns** specified above. 

## 2. Data Augmentation

To expand the dataset, we implemented three distinct approaches to effectively handle all languages. In each case, we only paraphrased the "sentence1" column while preserving both "sentence2" and the original similarity scores, as the semantic meaning remained unchanged through rephrasing.

The languages were processed in three phases:  
- **First Phase**: English (en), Spanish (es), French (fr), Italian (it), German (de), Chinese (zh), Dutch (nl), Russian (ru)  
- **Second Phase**: Portuguese (pt), Polish (pl)  
- **Third Phase**: Japanese (ja)  

### FIRST PHASE

For the initial eight languages, we employed backtranslation using English as the pivot language (except for English itself, where French served as the intermediate language). This process leveraged Hugging Face's MarianMTModel transformer ([documentation](https://huggingface.co/docs/transformers/model_doc/marian)), though the remaining three languages required alternative methods due to unsupported translation combinations.

The workflow began by creating eight language-specific subsets from clean_df (df_en, df_es, df_fr, etc.), each filtered by their respective lang1 values. Using MarianMTModel, we first translated sentences from the original language to the pivot language, then back to the source language. This two-step process maintained semantic meaning while varying lexical choices.

To optimize performance, we implemented batch translation monitored via tqdm, supported by two core functions:  
1. `translate_batch()` handles batch conversions between language pairs  
2. `back_translate()` manages the complete two-step process  

We utilized pretrained Hugging Face translation models (e.g., Helsinki-NLP/opus-mt-en-fr) with automatic GPU acceleration when available. The output generated augmented DataFrames (df_en_aug, df_es_aug, etc.) containing paraphrased sentences that enhanced both dataset size and NLP model robustness.


### SECOND PHASE

For Portuguese and Polish, we implemented a word-level substitution method based on synonymic and semantic replacement strategies. This approach was necessary because MarianMTModel didn't adequately support these languages for backtranslation.

We utilized pretrained word vectors from the FastText library ([available here](https://fasttext.cc/docs/en/crawl-vectors.html)), an open-source tool for text representation and classification. While FastText offers both binary and text formats for word vectors, we selected the binary version to avoid dependency conflicts with NumPy that occur when using the text format with Gensim.

#### Implementation Approach

After installing FastText, we temporarily loaded the pretrained models for Portuguese (`cc.pt.300.bin`) and Polish (`cc.pl.300.bin`). These models enabled us to find semantically similar words for substitution.

The process began with tokenizing sentences, through NLTK, from the "sentence1" column for both languages. We focused on the most relevant vocabulary by extracting the top 500 most frequent words. For synonym replacement, we applied three key rules:
1. Only words with similarity ≥ 0.75 were retained
2. Words were never replaced with themselves
3. Original words were kept when no suitable synonym was found

Each sentence underwent tokenization, lowercasing, and word-by-word analysis. Words found in our substitution dictionary were replaced, while others remained unchanged. The modified sentences were stored in a new column named `sentence1_aug`, and the paraphrased outputs for both languages were combined into a single DataFrame called `df_paraphrased`.
  
### THIRD PHASE  

For Japanese (ja), we employed **Rinna's Japanese GPT-2 Medium** (AutoModelForCausalLM) due to NLTK's limited support for this language. The model documentation can be found on Hugging Face: [Rinna Japanese GPT-2 Medium](https://huggingface.co/rinna/japanese-gpt2-medium). Unlike our approach with other languages, this method didn't paraphrase sentences but instead generated continuations of the input sentences.

#### Implementation Process  

After importing the necessary libraries, we loaded Rinna's pretrained Japanese GPT-2 model. The tokenizer was configured to use left-padding with the EOS token, which proved crucial for handling variable-length inputs. We then implemented two key functions:  

- The `generate_batch()` function processed lists of Japanese sentences, generating continuations with temperature sampling (set to 0.9, to add variability to the responses) while avoiding repetitions through no-repeat n-gram constraints.
- The `paraphrase_dataframe()` function managed the workflow, splitting the "sentence1" column into batches, processing them through `generate_batch()`, and storing the cleaned results in a new `sentence1_aug` column within the `df_ja_aug` DataFrame.

#### Similarity Score Recalculation  

Since the GPT-2 generated continuations rather than direct paraphrases, often altering the original meaning, we needed to recompute the similarity scores. We used the multilingual `paraphrase-multilingual-mpnet-base-v2` (SBERT) model, specifically optimized for cross-lingual semantic similarity tasks. This model produces scores between 0 (completely dissimilar) and 1 (identical meaning).

The `compute_similarity(row)` function handled the core calculation, encoding both sentences into dense vectors and computing their cosine similarity. We applied this function across `df_ja_aug` using `tqdm.progress_apply` for progress tracking, storing results in a new `score` column. Finally, we rescaled these values to a 0-5 range to maintain consistency with our original dataset's scoring system.

## 3. Model Development and Evaluation

To create our final dataset, we merged `clean_df` with all newly created datasets (`df_en_aug`, `df_es_aug`, etc.), resulting in the `augmented_df` containing 1,881,076 rows. After removing duplicates and NA values, we obtained the cleaned `final_df` with 1,847,162 rows, representing a 96.5% increase from the original `clean_df`.

### Model Development

We implemented BERT (Bidirectional Encoder Representations from Transformers) using `BertTokenizer` and `BertModel` ([documentation](https://huggingface.co/docs/transformers/model_doc/bert)). BERT's pretraining on masked token prediction and next-sentence tasks provides deep linguistic understanding through its bidirectional architecture.

**Initial Prototyping Phase**  
We first tested our approach on a 50,000-row subset (`small_df`) for faster iteration. Using a fixed random seed (42, as it is the [Ultimate Question of Life, the Universe, and Everything](https://en.wikipedia.org/wiki/Phrases_from_The_Hitchhiker%27s_Guide_to_the_Galaxy#The_Answer_to_the_Ultimate_Question_of_Life,_the_Universe,_and_Everything_is_42)), we split the data 90-10 for train-test validation. The `bert-base-multilingual-cased` tokenizer handled our multilingual requirements, supporting over 100 languages. Our `CustomDataset` class processed sentence pairs into BERT-compatible inputs with padding/truncation (max length: 64 tokens) and normalized similarity scores to 0-5 range.

**Full Dataset Implementation**  

For the complete `final_df`, we maintained the same seed (42) for an 80-20 train-test split, removing score values from test data. The model accepted tokenized sentences truncated to 32 tokens for efficiency, paired with similarity scores. 

The `BERTSimilarityModel` architecture is built upon BERT with:
- Dropout layers for regularization  
- Linear regression head for score prediction  
- Input handling for `input_ids`, `attention_mask`, and `token_type_ids`  

Training utilized:
- AdamW optimizer
- MSELoss function  
- Mixed precision training (autocast + GradScaler)  
- 3-batch rounds via DataLoader  

Predictions on test data preserved the original structure while generating similarity scores. The complete workflow balanced computational efficiency with model performance, leveraging BERT's multilingual capabilities while adapting to our specific semantic similarity task.

## Evaluation

We employed three complementary evaluation metrics to thoroughly assess model performance:

**MSE (Mean Squared Error)** strongly penalizes larger errors, making it sensitive to significant prediction deviations. **MAE (Mean Absolute Error)** provides an intuitive, real-scale measurement of average error magnitude. **R² Score** quantifies how much variance in the actual data is explained by the model, indicating overall predictive capability.

### Results on Subset (50,000 rows)
- **MSE**: 0.6595  
- **MAE**: 0.6221  
- **R² Score**: 0.6966  

### Results on Full Dataset
- **MSE**: 0.0339  
- **MAE**: 0.1062
- **R² Score**: 0.9844  

## 4. Results

The obtained results demonstrate strong model performance on the training data. However, the high scores may also suggest potential overfitting, indicating the model might not generalize optimally to new, unseen data. Further validation with additional test sets would be valuable to verify true generalization capability.

## 5. Conclusions

The project successfully developed a multilingual BERT model for semantic similarity prediction, achieving strong performance metrics (MSE: 0.0339, MAE: 0.1062, R²: 0.9844) that demonstrate its effectiveness in capturing cross-lingual semantic relationships. These results validate our approach, combining data augmentation techniques with transformer architecture.

While the current implementation shows promising results, several improvements could enhance the model's robustness. For example:

- Expanding the dataset with original sentences, rather than relying solely on paraphrased versions, would likely improve generalization.
- Implementing more rigorous validation through cross-validation or using completely independent test sets could provide stronger evidence of model performance.

The architecture could also benefit from controlled fine-tuning of specific BERT layers, potentially with increased dropout rates and the use of early stopping to prevent overfitting. Future work might explore alternative model architectures or ensemble approaches that combine multiple models for improved prediction stability.

This project establishes a reliable foundation for multilingual semantic similarity analysis while identifying clear pathways for future refinement. The demonstrated results, coupled with the outlined improvement strategies, position this work as a strong starting point for more advanced implementations in this domain.
