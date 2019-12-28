import pandas as pd, string, re, numpy as np, collections, nltk, time, spacy
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from nltk.corpus import stopwords
from collections import Counter

def question_one(df):
    # First, count all queriesby ID
    val = df.groupby('ID').count()
    # Then find the mean of that list
    return val['text'].mean()

def question_two(df):
    master_list = []
    query_lengths = []
    char_lengths = []

    # Loop down each row, look at the text
    for col, row in df.iterrows():
        curr_list = []        
        curr_list.append(row['text'])
        query = str(curr_list[0])
        words = query.split()
        # Figure out how many words are in the text and how many characters
        # Append those numbers to lists
        characters = [char for char in query]
        char_lengths.append(len(characters))
        query_lengths.append(len(words))

    # Find mean and median of the lists
    mean_query_length = np.mean(query_lengths)
    median_query_length = np.median(query_lengths)

    mean_chars = np.mean(char_lengths)
    median_chars = np.median(char_lengths)

    # Return it
    return mean_query_length, median_query_length, mean_chars, median_chars

def question_four(df):
    #Question 4: What percent of the time does a user request only the top 10 results?
    # Create a list of all of the ranges
    results = list(df['range'])
    total = len(results)
    count_10 = 0

    # Loop through, see if its equal to 0, if it is, then they want top 10
    for cnt in results:
        if cnt == 0:
            count_10 += 1
        else:
            continue

    # Return the %
    return (count_10 / total) * 100
    
def question_five(df):
    # Create variables
    count_questions = 0
    who_count = 0
    what_count = 0
    where_count = 0
    why_count = 0
    when_count = 0

    new_df = df['text'].unique()

    # Loop down each row, look at the text
    for item in new_df:
    # Only count a query once
        query = str(item)
        words = query.lower().split()
        # What defines a question
        question_words = ['who', 'what', 'where', 'when', 'why', '?']

        # If any of those appear, count it as a question
        if(any(x in words for x in question_words)):
            count_questions += 1

        # Also count a question if there is an ? in the query
        characters = [char for char in query]
        if('?' in characters):
            count_questions += 1            
        
        # Create counts by query type
        if('who' in words):
            who_count += 1
        
        elif ('what' in words):
            what_count += 1
        
        elif('where' in words):
            where_count += 1
        
        elif('when' in words):
            when_count += 1
        
        elif('why' in words):
            why_count += 1     

    # Return the vals
    return (count_questions / len(new_df)) * 100, who_count, what_count, where_count, when_count, why_count

def question_six(df):
    # Count the amount of times each value in the text column occurs
    ans = df['text'].value_counts()
    # Only select the top 20
    new_ans = ans.head(20)
    # Return the answer
    return new_ans

def question_seven(df):
    # Create a counter object
    collectionFreq = Counter()
    # Use NLTK's stop words
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

    # Loop down the dataframe
    for col, row in df.iterrows():
        # Create a string oject of the text
        query = str(row['text'])
        # Split it
        tokens = query.split()
        # Make sure it isnt just a character
        real_tokens = [data for data in tokens if len(data) > 1]
        # print(real_tokens)

        # Loop through the words
        for curr in real_tokens:
            # If its not a stop word
            if curr.lower() not in stop_words:
                # Update its counter
                collectionFreq[curr] += 1

    # Return the 20 most common    
    return collectionFreq.most_common(20)

def question_eight(df):
    # Use NLTK's stop words
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    
    # Create counts
    contain_count = 0
    total_count = 0
    # Loop down the dataframe
    for col, row in df.iterrows():
        # Create a string oject of the text
        query = str(row['text'])
        # Split it
        tokens = query.split()
        # Make sure it isnt just a character
        real_tokens = [data for data in tokens if len(data) > 1]
        
        # Loop through the words
        for curr in real_tokens:
            # If any word is in the stop words list
            if curr.lower() in stop_words:
                # Count it
                contain_count += 1

        # Increment the total observed samples
        total_count += 1

    return ((contain_count / total_count) * 100)

def question_nine(df):
    # NLTK stopwords
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    collectionFreq = Counter()

    # Loop down the dataframe
    for col, row in df.iterrows():
        # Create a string oject of the text
        query = str(row['text'])
        # Split it
        tokens = query.split()
        # Make sure it isnt just a character
        real_tokens = [data for data in tokens if len(data) > 1]

        if('download' in real_tokens):
            # Loop through the words
            for curr in real_tokens:
                # If its not a stop word
                if curr.lower() not in stop_words:
                    # Update its counter
                    collectionFreq[curr] += 1
    
    # Return the 10 most common
    return collectionFreq.most_common(10)

def question_twelve(df):
    # Specified terms
    search_terms = ['Al Gore', 'Johns Hopkins', 'John Hopkins']
    # Create a counter
    termFreq = Counter()

    # Loop down the dataframe
    for col, row in df.iterrows():
        # Create a string oject of the text
        query = str(row['text'])
        # Loop through the search terms
        for x in search_terms:
            # If its in the query
            if(x in query):
                # Update its count
                termFreq[x] += 1
    
    # Return it
    return termFreq
    
def question_thirteen(df):
    # Specified terms
    search_terms = ['www', '.com', '.edu', '.gov', '.net', 'http:', 'https:']
    # Create a counter
    url_count = 0
    total_count = 0

    # Loop down the dataframe
    for col, row in df.iterrows():
        # Create a string oject of the text
        query = str(row['text'])
        # Loop through the search terms
        for x in search_terms:
            # If the query contains something a url would
            if(x in query):
                # Update its count
                url_count += 1
        
        # Update the total
        total_count += 1
    
    # Return it
    return ((url_count / total_count) * 100)

def question_fourteen(df):
    # Load the spacy model
    nlp = spacy.load('en_core_web_sm')

    # Create counts
    person_count = 0
    company_count = 0
    total_count = 0

    # Turn the text column into a numpy array for faster processing
    rows = df['text'].to_numpy()

    # Loop down the row
    for row in rows:
        # Turn current row to a string
        curr = str(row)
        # Run the model on the row
        doc = nlp(curr)
        # Find all proper entities
        proper = [ent for ent in doc.ents]

        # If the model classified as a word as a proper noun
        if len(proper) >= 1:
            # Loop through the proper nouns
            for entity in proper:
                # If it was an organization, increase the organization count
                if(entity.label_ == 'ORG'):
                    company_count += 1
                # If it was a person, increase the person count
                elif(entity.label_ == 'PERSON'):
                    person_count += 1
        else:
            continue
    
    # Return the %s of queries that contained someones name or a company
    return ((person_count / len(rows)) * 100), ((company_count / len(rows)) * 100)


def question_sixteen(df):
    # Specified terms
    search_terms = ["'", '"', '+', '-', 'and', 'or', 'nor']
    # Create a counter
    search_engine_count = 0
    total_count = 0

    # Loop down the dataframe
    for col, row in df.iterrows():
        # Create a string oject of the text
        query = str(row['text'])
        # Loop through the search terms
        for x in search_terms:
            # If the query contains something a url would
            if(x in query.lower()):
                # Update its count
                search_engine_count += 1
        
        # Update the total
        total_count += 1
    
    # Return it
    return ((search_engine_count / total_count) * 100)

def main():
    t0 = time.time()

    tsv_file = "D:\\Grad School\\Fall 2019\\605.744.81.FA19 - Information Retrieval\\Programming Assignment 5\\19991220-Excite-QueryLog.tsv"
    csv_table=pd.read_csv(tsv_file, sep='\t', header=None)
    csv_table.columns = ['time', 'ID', 'range', 'text']

    # q1_ans = question_one(csv_table)
    # print("The average number of queries per user ID is: {}".format(int(q1_ans)))
    # t1 = time.time()
    # print("Time to complete: {}".format(t1-t0))

    # print("---------------------------------------------------------------------------")

    # q2_mean_query_length, q2_median_query_length, q2_mean_char_length, q2_median_char_length = question_two(csv_table)
    # print("Mean Query Length: {} | Median Query Length: {} | Mean Chars: {}  | Median Chars: {}".format(q2_mean_query_length, q2_median_query_length, q2_mean_char_length, q2_median_char_length))
    # t2 = time.time()
    # print("Time to complete: {}".format(t2-t1))

    # print("---------------------------------------------------------------------------")

    # q4_ans = question_four(csv_table)
    # print("The percent of the time that a user requests only the top 10 results is: {}".format(q4_ans))
    # t3 = time.time()
    # print("Time to complete: {}".format(t3-t2))

    # print("---------------------------------------------------------------------------")

    # q5_total_questions, q5_who, q5_what, q5_where, q5_when, q5_why = question_five(csv_table)
    # print("Total Number of Questions: {} | Total Number of 'Who' Questions: {} | Total Number of 'What' Questions: {} | Total Number of 'Where' Questions: {} | Total Number of 'When' Questions: {} | Total Number of 'Why' Questions: {}".format(q5_total_questions, q5_who, q5_what, q5_where, q5_when, q5_why))
    # t4 = time.time()
    # print("Time to complete: {}".format(t4-t3))

    # print("---------------------------------------------------------------------------")

    # q6_ans = question_six(csv_table)
    # print("The top 20 most common queries were: ")
    # print(q6_ans)
    # t5 = time.time()
    # print("Time to complete: {}".format(t5-t4))

    # print("---------------------------------------------------------------------------")

    # q7_ans = question_seven(csv_table)
    # print("The top 20 most common search terms that were not stop words were: ")
    # print(q7_ans)
    # t6 = time.time()
    # print("Time to complete: {}".format(t6-t5))

    # print("---------------------------------------------------------------------------")

    # q8_ans = question_eight(csv_table)
    # print("The percentage of queries containing stopwords is: {}".format(q8_ans))
    # t7 = time.time()
    # print("Time to complete: {}".format(t7-t6))

    # print("---------------------------------------------------------------------------")

    # q9_ans = question_nine(csv_table)
    # print("The 10 most common non-stopwords appearing in queries that contain the word download are: ")
    # print(q9_ans)
    # t8 = time.time()
    # print("Time to complete: {}".format(t8-t7))

    # print("---------------------------------------------------------------------------")

    # q12_ans = question_twelve(csv_table)
    # print("The counts for John Hopkins, Al Gore, and Johns Hopkins are: ")
    # print(q12_ans)
    # t9 = time.time()
    # print("Time to complete: {}".format(t9-t8))

    # print("---------------------------------------------------------------------------")

    # q13_ans = question_thirteen(csv_table)
    # print("URLs appear in queries {} percent of the time".format(q13_ans))
    # t10 = time.time()
    # print("Time to complete: {}".format(t10-t9))

    # print("---------------------------------------------------------------------------")

    # q14_person, q14_company = question_fourteen(csv_table)
    # print("People's names appear in {} percent of queries | Company names appear in {} percent of queries".format(q14_person, q14_company))
    # t11 = time.time()
    # print("Time to complete: {}".format(t11-t0))

    # print("---------------------------------------------------------------------------")

    q16_ans = question_sixteen(csv_table)
    print("Search engine query syntax is used in {} percent of queries".format(q16_ans))
    t12 = time.time()
    print("Time to complete: {}".format(t12-t0))

    print("---------------------------------------------------------------------------")



main()