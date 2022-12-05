# ConWea Baseline Model Replication

## Data
Each dataset should contain following files:
1) DataFrame pickle file \
    ex) ```data/nyt/coarse/df.pkl```
      * Dataset has TWO columns - sentence & label
      * Sentence contains text, label contains corresponding label
      * MUST be named as df.pkl
2) Seedwords JSON file \
    ex) ```data/nyt/coarse/seedwords.json```
      * This JSON file contains seed words for each label
      * MUST be named seedwords.json

## Running the Project
To run this project, run command;
```
python run.py [test | data] [dataset] 
```
Note: If running ```python run.py test``` \
No dataset needs to be specified. 

If running ```python run.py data```: \
ONLY datasets of **nyt** and **20news** are supported. 

Example commands include: \
``` python run.py test ``` \
``` python run.py data nyt coarse ``` \
``` python run.py data 20news fine ``` 

## File Structure Rundown
* **data** ---- contains data to run this project, compressed in form of .pkl files for data and .json for seed words
    * **nyt**
        * **coarse**
            * df.pkl
            * seedwords.json
        * **fine**
            * ...
    * **20news**
        * ...
* **src**
    * **dataset**
        * read_data.py ---- contains method to read data
    * **models**
        * TFIDF.py ---- TFIDF implementation and prediction
        * Word2Vec.py ---- Word2Vec implementation and prediction
    * **util** ---- contains helper functions
        * util.py ---- insurance that stopwords from nltk are downloaded
* **test**
    * **testdata**
        * data.pkl
        * seedwords.json
* Dockerfile ---- environment image
* run.py ---- main run script for project
