import numpy as np
import pandas as pd
import re
import warnings
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
import pickle
from nltk.stem import SnowballStemmer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
from os import path
from keras.models import model_from_json
from sklearn.ensemble import RandomForestClassifier


# Scrape data from pokemmondb.net using BeautifulSoup. This took a lot
# of trial and error but eventually found the right id's. Take this data
# and put it into a Panda DataFrame. Then save it to a csv file.
def create_csv():
    
    pokemon_columns = ['Number', 'Name', 'Type1', 'Type2', 'Pokedex']
    
    df = pd.DataFrame(columns = pokemon_columns)
    
    # There are 898 Pokemon
    for number in range(1, 899):
    
        url = 'https://pokemondb.net/pokedex/' + str(number)
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        name_html = str(soup.find('title'))
        name_html_endpoint = str(name_html).find(' Pokédex')
        name = name_html[7: name_html_endpoint]
        
        pokedex_html = soup.find_all('td', { 'class' : 'cell-med-text' })
        pokedex = '\n'.join([p.text for p in pokedex_html])
        
        type1 = soup.find_all('a', {'class': re.compile('.*itype.*')})[0].text
        type2 = None
        
        try:
            
            type2 = soup.find_all('a', {'class': re.compile('.*itype.*')})[1].text
            
            if type1 == type2:
                
                type2 = None
                
        except:
            
            pass
        
        temp = pd.DataFrame(data = {pokemon_columns[0]:[number],
                                    pokemon_columns[1]:[name],
                                    pokemon_columns[2]:[type1],
                                    pokemon_columns[3]:[type2],
                                    pokemon_columns[4]: [pokedex]})
        
        df = df.append(temp, ignore_index = True)
    
    df.to_csv('./csv/pokedex.csv', index = False)


# Clean up the csv file. Add an encoded column to represent the primary
# type. As described below, an adjusted label column is also created.
# Afterwards, execute some regex to clean up the pokedex strings.
def clean_csv():
    
    pokemon = pd.read_csv('./csv/pokedex.csv')
    
    num_of_pokemon = len(pokemon)
    
    label_enc = LabelEncoder()
    
    pokemon['Label'] = label_enc.fit_transform(pokemon[['Type1']])
    
    
    # Flying is a very uncommon primary type, but very common secondary type.
    # Normal is a very common primary type, and even moreso when Flying is the
    # secondary type. Therefore, I am setting those Pokemon to have a primary
    # type of Flying.
    pokemon['Label_adjusted'] = [7
                               if 
                                   pokemon['Type1'].loc[poke] == 'Normal' and pokemon['Type2'].loc[poke] == 'Flying'
                               else
                                   pokemon['Label'].loc[poke] for poke in range(num_of_pokemon)]

    pokemon['Pokedex'] = pokemon['Pokedex'].str.lower()
    
    pokemon['Pokedex'] = pokemon['Pokedex'].replace(to_replace = "['’-]",
                                                    value = ' ',
                                                    regex = True)
    
    pokemon['Pokedex'] = pokemon['Pokedex'].replace(to_replace = 'it s ',
                                                    value = 'it is ',
                                                    regex = True)
    
    pokemon['Pokedex'] = pokemon['Pokedex'].replace(to_replace = ' s ',
                                                    value = ' ',
                                                    regex = True)
    
    pokemon['Pokedex'] = pokemon['Pokedex'].replace(to_replace = 'é',
                                                    value = 'e',
                                                    regex = True)
    
    pokemon['Pokedex'] = pokemon['Pokedex'].replace(to_replace = '\n',
                                                    value = ' ',
                                                    regex = True)
    
    pokemon['Pokedex'] = pokemon['Pokedex'].replace(to_replace = '[^a-zA-Z0-9 ]',
                                                    value = '',
                                                    regex = True)
    
    pokemon['Pokedex'] = pokemon['Pokedex'].replace(to_replace = 'pokemon',
                                                    value = '',
                                                    regex = True)
    
    pokemon['Pokedex'] = pokemon['Pokedex'].replace(to_replace = [pokemon_name.lower() for pokemon_name in pokemon.Name.values],
                                                    value = '',
                                                    regex = True)
    
    pokemon['Pokedex'] = pokemon['Pokedex'].replace(to_replace = '[0-9]+',
                                                    value = '',
                                                    regex = True)
    
    pokemon['Pokedex'] = pokemon['Pokedex'].replace(to_replace = ' +',
                                                    value = ' ',
                                                    regex = True)

    pokemon.to_csv('./csv/cleaned_pokedex.csv', index = False)


# Take the csv file (from the other file) as input and create a stemmed list of tokens and a
# stemmed version of the pokedex. This will help identify the same words
# with different endings (eg. swimmimg -> swim, swim -> swim).
def create_tokens():
    
    pokemon = pd.read_csv('./csv/cleaned_pokedex_bert.csv')
    stem = SnowballStemmer('english')
    stop_words = stopwords.words('english')
    
    tokens = []
    s = []
    stemmed_tokens = []
    stemmed_pokedex = []
    temp = []
    
    num_of_pokemon = len(pokemon)
    
    for i in range(num_of_pokemon):
        
        tokens.append(pokemon.Pokedex.values[i].split(' '))
    
    num_of_tokens = len(tokens)
    
    for j in range(num_of_tokens):
        
        s += tokens[j]
        
    s = sorted(list(set(s)))
    
    tokens_without_stopwords = np.setdiff1d(s, stop_words)
    
    for token in tokens_without_stopwords:
        
        stemmed_tokens.append(stem.stem(token))
    
    for i in range(num_of_pokemon):
        
        temp = np.setdiff1d(pokemon.Pokedex.values[i].split(' '), stop_words)
        
        token_indices = [ list(tokens_without_stopwords).index(token) for token in temp ]
        
        stemmed_pokedex.append([stemmed_tokens[token_index] for token_index in token_indices])
        
        stemmed_pokedex[i] = list(set(stemmed_pokedex[i]))
    
    with open('./stemmed/stemmed_tokens', 'wb') as st_file:
        pickle.dump(stemmed_tokens, st_file)
        
    with open('./stemmed/stemmed_pokedex', 'wb') as sp_file:
        pickle.dump(stemmed_pokedex, sp_file)


# Helper function to gather and split the data into train and test
# sets, and complete one hot encoding.
def gathering_splitting_data():
    
    pokemon = pd.read_csv('./csv/cleaned_pokedex_bert.csv')
    
    with open ('./stemmed/stemmed_tokens', 'rb') as st_file:
        stemmed_tokens = pickle.load(st_file)
    
    with open('./stemmed/stemmed_pokedex', 'rb') as sp_file:
        stemmed_pokedex = pickle.load(sp_file)
    
    binarizer = MultiLabelBinarizer()
    binarizer = binarizer.fit([list(set(stemmed_tokens))])
    
    sampler = RandomOverSampler(sampling_strategy = 'not majority', random_state = 42)
    
    one_hot_encoding = binarizer.transform(stemmed_pokedex)
    values = pokemon.Label_adjusted.values
    
    train_x, test_x, train_y, test_y = train_test_split(one_hot_encoding,
                                                        values,
                                                        test_size = 0.2,
                                                        random_state = 42,
                                                        stratify = values)
    
    train_x, train_y = sampler.fit_resample(train_x, train_y)
    
    return stemmed_tokens, train_x, train_y, test_x, test_y


# Create a Bernoulli Naive Bayes object to train and make a prediction.
def bernoulli_naive_bayes():
    
    _, train_x, train_y, test_x, test_y = gathering_splitting_data()
    
    if not path.exists('./models/bernoulli_naive_bayes'):
    
        model = BernoulliNB()
        
        model.fit(train_x, train_y)
        
        with open('./models/bernoulli_naive_bayes', 'wb') as model_file:
            pickle.dump(model, model_file)
            
    else:
        
        with open('./models/bernoulli_naive_bayes', 'rb') as model_file:
            model = pickle.load(model_file)

    pred_y = model.predict(test_x)
    
    print('\nBernoulli Naive Bayes\n')
    print('Training: ' + str(model.score(train_x, train_y)))
    print('Test: ' +  str(model.score(test_x, test_y)))
    
    display_results(pred_y, test_y, 'bnb')


# Create a Sequential object to train and make a prediction.
# I used relu and softmax as the activation functions.
# I used categorical cross entropy as the loss parameter with an adam
# optimizer and used the accuracy as the metric.
def neural_network():
    
    stemmed_tokens, train_x, train_y, test_x, test_y = gathering_splitting_data()
    
    if not path.exists('./models/neural_network_model.json'):
        
        model = Sequential()
        
        model.add(Dense(12, input_dim = len(list(set(stemmed_tokens))), activation = 'relu'))
        model.add(Dense(8, activation = 'relu'))    
        model.add(Dense(18, activation = 'softmax'))
        
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        
        model.fit(train_x, to_categorical(train_y), epochs = 1000, verbose = 0)
        
        json_model = model.to_json()
        
        with open('./models/neural_network_model.json', 'w') as json_file:
            json_file.write(json_model)
            
        model.save_weights('./models/neural_network_model.h5')
        
    else:
        
        json_file = open('./models/neural_network_model.json', 'r')
        json_model = json_file.read()
        json_file.close()
        
        model = model_from_json(json_model)
        model.load_weights('./models/neural_network_model.h5')
        
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
            
    _, score = model.evaluate(test_x, to_categorical(test_y), verbose = 0)
    
    pred_y = np.argmax(model.predict(test_x), axis = 1)
    
    print('\nNeural Network\n')
    print('Score: ' + str(score))
    
    display_results(pred_y, test_y, 'nn')
    

# Create a Random Forest Classifier object to train and make a prediction.
def random_forest():
    
    _, train_x, train_y, test_x, test_y = gathering_splitting_data()
    
    if not path.exists('./models/random_forest'):
    
        model = RandomForestClassifier(random_state = 42)
        
        model.fit(train_x, train_y)
        
        with open('./models/random_forest', 'wb') as model_file:
            pickle.dump(model, model_file)
            
    else:
        
        with open('./models/random_forest', 'rb') as model_file:
            model = pickle.load(model_file)

    pred_y = model.predict(test_x)
    
    print('\nRandom Forest\n')
    print('Test: ' +  str(model.score(test_x, test_y)))
    
    display_results(pred_y, test_y, 'rf')


# Helper function to plot the confusion matrix with sns
# heatmap and print the classification report.
def display_results(pred_y, true_y, model_type):
    
    cleaned_pokedex = pd.read_csv('./csv/cleaned_pokedex_bert.csv')
    
    labels = sorted(cleaned_pokedex['Type1'].unique())
        
    confusion = confusion_matrix(true_y, pred_y)
    classification = classification_report(true_y, pred_y, target_names = labels)

    _, axis = plt.subplots(figsize = (12, 10))
    
    sns.heatmap(confusion,
                annot = True,
                fmt = 'd',
                xticklabels = labels,
                yticklabels = labels,
                ax = axis)
    
    plt.xlabel('Prediction')
    plt.ylabel('True')
    
    plt.show()
    
    print()
    print(classification)


def main():
    
    warnings.simplefilter(action = 'ignore', category = FutureWarning)
    
    # Uncomment as needed.
    
    #create_csv()
    #clean_csv()
    #create_tokens()
    #bernoulli_naive_bayes()
    #neural_network()
    #random_forest()
    
    print()
    
    
if __name__ == '__main__':
    
    main()