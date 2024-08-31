# DL_Spam_filter
Quora question Spam classifier project 

Objective : 
To identify whether questions that are being posted in a forum like Quora are they spam questions?

Approach :
We have to solve this problem with Neural networks.
1. Data Analayis
2. Tokenization
3. Glove Embeddings 100 dimensions
4. Model Training
5. Model prediction

**Data Analysis:**
I started with data analysis,checked whether there are any Null values in the table and then removed extra blankspaces from the data.
I didn't remove any other details from data as special characters,uppercase letters etc., may be useful for identifying spam questions.

split the data into three chunks Train, validation and test datasets with 80%,10% and 10% distribution.

**Tokenization:**
Used tokenizer from tensorflow, to convert the question words into text tokens and then converted them to integer sequences.
I then worked on padding to make all sequences of same length for which,I checked max length of my sequences through histogram distribution to understand where majority of entries lying and fixed max_length as 100 so all sequences which are less than 100 are padded with zero and more than 100 are truncated.

**Glove Embeddings:**
I used 100 dimensions glove embeddings from glove and created embedding matrix 

**Model Training**
Model is then trained with embedding layer first then with bi-directional LSTM to get context from both directions i added dropout layer to avoid overfitting and then read the data from one direction using LSTM and then flattned using dense layers at last used sigmoid for classification.
I used adam optimizer.
Saved Model, Tokenizer and Max Length

**Model Prediction**
I have used tokenizer and max length additionally as we need tokenizer to convert new questions to tokens and max length to pad the sequences to max 100, length on which model is built.
After loading model, i have some examples pre loaded where we could see both working and not working cases.
Then i have also enabled option for user to input the question and see if posted question is spam or not as per mode1.

