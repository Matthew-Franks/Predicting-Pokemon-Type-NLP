import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
import torch
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW
import time
import datetime
from transformers import get_linear_schedule_with_warmup
import random
from tensorflow import nn


# Read in the pokedex.csv file. I made this in the other file when I scraped
# it from the website using BeautifulSoup. The difference this time is I'm making
# a row for each pokedex line, which is why I'm splitting by newline characters.
# After doing most of the same operations from the other file, like adding an
# adjusted label column, and cleaning up the text in pokedex_adjusted, save this
# new csv file for later use.
def read_and_clean_csv():    
    
    pokemon_columns = ['Number','Name','Type1','Type2','Pokedex','Label']
    pokemon = pd.DataFrame()
    
    pokemon_csv = pd.read_csv('./csv/pokedex.csv')
    pokemon_csv['Label'] = LabelEncoder().fit_transform(pokemon_csv.Type1.values)
    
    for i in range(len(pokemon_csv)):
        
        for pokedex_line in pokemon_csv.iloc[i]['Pokedex'].split('\n'):
            
            pokemon = pd.concat([pokemon,
                                 pd.DataFrame(
                                             [[pokemon_csv.iloc[i]['Number'],
                                               pokemon_csv.iloc[i]['Name'],
                                               pokemon_csv.iloc[i]['Type1'],
                                               pokemon_csv.iloc[i]['Type2'],
                                               pokedex_line,
                                               pokemon_csv.iloc[i]['Label']]])])
            
    pokemon.columns = pokemon_columns
    pokemon = pokemon.reset_index(drop = True)
    
    pokemon['Label_adjusted'] = [7
                               if 
                                   pokemon['Type1'].loc[poke] == 'Normal' and pokemon['Type2'].loc[poke] == 'Flying'
                               else
                                   pokemon['Label'].loc[poke] for poke in range(len(pokemon))]

    pokemon['Pokedex_adjusted'] = pokemon['Pokedex'].str.lower()
    
    pokemon['Pokedex_adjusted'] = pokemon['Pokedex_adjusted'].replace(to_replace = "['’-]",
                                                                      value = ' ',
                                                                      regex = True)
    
    pokemon['Pokedex_adjusted'] = pokemon['Pokedex_adjusted'].replace(to_replace = 'it s ',
                                                                      value = 'it is ',
                                                                      regex = True)
    
    pokemon['Pokedex_adjusted'] = pokemon['Pokedex_adjusted'].replace(to_replace = ' s ',
                                                                      value = ' ',
                                                                      regex = True)
    
    pokemon['Pokedex_adjusted'] = pokemon['Pokedex_adjusted'].replace(to_replace = 'é',
                                                                      value = 'e',
                                                                      regex = True)
    
    pokemon['Pokedex_adjusted'] = pokemon['Pokedex_adjusted'].replace(to_replace = '\n',
                                                                      value = ' ',
                                                                      regex = True)
    
    pokemon['Pokedex_adjusted'] = pokemon['Pokedex_adjusted'].replace(to_replace = '[^a-zA-Z0-9 ]',
                                                                      value = '',
                                                                      regex = True)
    
    pokemon['Pokedex_adjusted'] = pokemon['Pokedex_adjusted'].replace(to_replace = 'pokemon',
                                                                      value = '',
                                                                      regex = True)
    
    pokemon['Pokedex_adjusted'] = pokemon['Pokedex_adjusted'].replace(to_replace = [i.lower() for i in pokemon.Name.values],
                                                                      value = '',
                                                                      regex = True)
    
    pokemon['Pokedex_adjusted'] = pokemon['Pokedex_adjusted'].replace(to_replace = '[0-9]+',
                                                                      value = '',
                                                                      regex = True)
    
    pokemon['Pokedex_adjusted'] = pokemon['Pokedex_adjusted'].replace(to_replace = ' +',
                                                                      value = ' ',
                                                                      regex = True)
    
    pokemon.to_csv('./csv/cleaned_pokedex_bert.csv', index = False)
    

# Helper function to calculate the accuracy given the predicted and true values
def accuracy_flattened(predicted_values, true_values):
    
    predicted_values_flattened = np.argmax(predicted_values, axis = 1).flatten()
    true_values_flattened = true_values.flatten()
    
    accuracy = np.sum(predicted_values_flattened == true_values_flattened) / len(true_values_flattened)

    return accuracy


# Helper function to provide an easy way to print the time that has elapsed.
# This was just used for debugging and to give myself an estimate of how
# long something would take (which was usually a very long time).
def print_time(time):
    
    rounded_time = int(round((time)))
    
    proper_time = str(datetime.timedelta(seconds = rounded_time))
    
    return proper_time


# Helper function to create attention masks based on our tokens to return
# for our train/test data
def create_attention_masks(tokens):
    
      attention_masks = []
      
      for token in tokens:
          
          attention_masks.append( [int(t > 0) for t in token] )
      
      return attention_masks


# Both train_model and test_model need all this data, so I separated it
# into its own function so that I wouldn't have to repeat it and waste space.
# Essentially, this function uses Bert to encode tokens that will be used when
# we train test split our data. Afterwards, we collect the inputs, attention
# masks, labels, and data, create a sampler, and create our dataloader.
# Depending on the input to this function, it will return the train set or
# the test set.
def gather_model_parameters(train_test):
    
    device = torch.device('cpu')
    
    cleaned_pokedex_bert = pd.read_csv('./csv/cleaned_pokedex_bert.csv')
    
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)
    
    tokens = []
    
    for i in range(len(cleaned_pokedex_bert)):
        
        tokens.append(bert_tokenizer.encode(cleaned_pokedex_bert['Pokedex_adjusted'][i], add_special_tokens = True))
    
    train_x, test_x, train_y, test_y = train_test_split(tokens,
                                                        cleaned_pokedex_bert.Label_adjusted.values,
                                                        test_size = 0.2,
                                                        random_state = 42,
                                                        stratify = cleaned_pokedex_bert.Label_adjusted.values)
    
    max_length = 64
    batch_length = 16
    
    if train_test == 'train':
        
        train_inputs = pad_sequences(train_x,
                                     maxlen = max_length,
                                     dtype = 'long',
                                     value = 0,
                                     truncating = 'post', 
                                     padding = 'post')

        train_attention_masks = torch.tensor(create_attention_masks(train_inputs))
        train_inputs = torch.tensor(train_inputs)
        train_labels = torch.tensor(train_y, dtype = torch.long)
        train_data = TensorDataset(train_inputs, train_attention_masks, train_labels)
        train_random_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data,
                                      sampler = train_random_sampler,
                                      batch_size = batch_length)
        
        return device, train_dataloader
        
    else:
        
        test_inputs = pad_sequences(test_x,
                                    maxlen = max_length,
                                    dtype = 'long',
                                    value = 0,
                                    truncating = 'post',
                                    padding = 'post')
        
        test_attention_masks = torch.tensor(create_attention_masks(test_inputs))
    
        test_inputs = torch.tensor(test_inputs)
        
        test_labels = torch.tensor(test_y, dtype = torch.long)
        
        test_data = TensorDataset(test_inputs, test_attention_masks, test_labels)
        test_random_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data,
                                     sampler = test_random_sampler,
                                     batch_size = batch_length)
        
        return device, test_inputs, test_dataloader
    
    #train_masks = torch.tensor(train_masks)
    #test_masks = torch.tensor(test_masks)
    

# Helper function to display the confusion matrix and classification report
# for our model. Had to adjust the pred_y and test_y values to get them in
# a shape that was useful for the analysis. The confusion matrix uses a
# seaborn heatmap to display the data.
def display_results(pred_y, true_y):
    
    cleaned_pokedex_bert = pd.read_csv('./csv/cleaned_pokedex_bert.csv')
    
    predicted_probabilities = []
    flattened_true_y = []
    
    for pred in pred_y:
        
        temp = nn.softmax(pred).numpy()
        
        for t in temp:
            
            predicted_probabilities.append(t)
            
    for i in range(len(true_y)):
        
        for label in true_y[i]:
            
            flattened_true_y.append(label)
    
    labels = sorted(cleaned_pokedex_bert['Type1'].unique())
    
    confusion = confusion_matrix(flattened_true_y,
                                 np.argmax(predicted_probabilities,axis = 1))
    
    _, axis = plt.subplots(figsize = (12, 10))
    
    sns.heatmap(confusion,
                annot = True,
                fmt = 'd',
                xticklabels = labels,
                yticklabels = labels,
                ax = axis)
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.show()
    
    classification = classification_report(flattened_true_y, np.argmax(predicted_probabilities, axis = 1), target_names = labels)
    
    print()
    print(classification)
    

# This function is used to train the Bert model. I chose the AdamW optimizer
# for my model. I set the epochs to 7, just to make sure I was getting the
# best result that I could achieve. Set the seed value to 42 so I would have
# the same result consistently. Then ran a classic Bert training loop. Kept
# track of the loss and the time elapsed for my own record to give perspective
# on how well my parameters were doing, which I had to switch around several
# times until I landed on what you see below. I saved this model using Torch
# to reuse it easily, as it took several hours to train.
def train_model():
    
    device, train_dataloader = gather_model_parameters('train')
    
    train_dataloader_length = len(train_dataloader)
    
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          num_labels = 18)

    adamw_optimizer = AdamW(model.parameters(), lr = 1e-5, eps = 1e-8)
    
    epochs = 7
    
    num_training_steps =  epochs * train_dataloader_length
    
    linear_schedule = get_linear_schedule_with_warmup(adamw_optimizer, 
                                                      num_warmup_steps = 0,
                                                      num_training_steps = num_training_steps)
    
    seed_value = 42
    
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    
    loss_list = []
    
    for epoch in range(epochs):
        
        print('\nEpoch: ' + str(epoch))
    
        start_time = time.time()
        loss_total = 0
    
        model.train()
    
        for step, batch in enumerate(train_dataloader):
            
            if step % 50 == 0 and not step == 0:
                
                t = print_time(time.time() - start_time)
                
                print('\nBatch: ' + str(step) + ' | Time: ' + str(t))
            
            batch_inputs, batch_mask, batch_labels = tuple(t.to(device) for t in batch)
            
            model.zero_grad()
            
            output_tuple = model(batch_inputs, 
                                 token_type_ids = None, 
                                 attention_mask = batch_mask, 
                                 labels = batch_labels)
            
            loss = output_tuple[0]
            loss_total += loss.item()
            loss.backward()
    
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
            adamw_optimizer.step()
            linear_schedule.step()
    
        average_loss = loss_total / train_dataloader_length            
        
        loss_list.append(average_loss)
        
        t = time.time() - start_time
        
        print('\nAverage Loss: ' + str(round(average_loss, 2)))
        print('Time: ' + print_time(t))
    
    try:
        
        torch.save(model, './models/torch_model')
        
    except:
        
        print('Torch Save did not work.')
        

# This function takes the model created and saved in train_model and evaluates
# the test data. Worth noting that this model has not touched any of the data
# inside the test set, so there is no risk of overfitting. Displays the accuracy
# and time elapsed at the end. Afterwards, calls display_results to show the
# confusion matrix and classification report.
def test_model():
    
    
    device, test_inputs, test_dataloader = gather_model_parameters('test')
    
    model = torch.load('./models/torch_model')
    
    accuracy = 0
    steps = 0
    
    model.eval()
    
    prediction_labels = []
    true_labels = []
    
    start_time = time.time()
    
    for batch in test_dataloader:
        
        batch_inputs, batch_mask, batch_labels = tuple(b.to(device) for b in batch)
                
        with torch.no_grad():
            
            output_tuple = model(batch_inputs,
                                 token_type_ids = None,
                                 attention_mask = batch_mask)
      
        logits = output_tuple[0]
        logits = logits.detach().cpu().numpy()
        
        labels = batch_labels.to('cpu').numpy()
        
        prediction_labels.append(logits)
        true_labels.append(labels)
        
        accuracy += accuracy_flattened(logits, labels)
        steps += 1
    
    t = time.time() - start_time
    
    print('Accuracy: ' + str(accuracy/steps))
    print('Time: ' + print_time(t))
    
    display_results(prediction_labels, true_labels)


def main():
    
    warnings.simplefilter(action = 'ignore', category = FutureWarning)
    
    # Uncomment as needed.
    
    #read_and_clean_csv()
    #train_model()
    #test_model()    
    
    print()
    
    
if __name__ == '__main__':
    
    main()