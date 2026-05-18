import os
import numpy as np
import pandas as pd
import kagglehub

def load_ravdess_dataframe():
    """Pobiera zbiór RAVDESS i tworzy DataFrame z ścieżkami, emocjami i aktorami."""
    ravdess = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
    
    file_path = []
    emotions_numbers = []
    actors_numbers = []

    for actor in os.listdir(ravdess):
        actor_path = os.path.join(ravdess, actor)
        for file in os.listdir(actor_path):
            if not file.endswith('.wav'):
                continue
            file_path.append(os.path.join(actor_path, file))
            emotions_numbers.append(int(file.split('-')[2]))
            actors_numbers.append(file.split('-')[6])

    emotions_df = pd.DataFrame(emotions_numbers, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    actors_df = pd.DataFrame(actors_numbers, columns=['Actor'])
    
    emot_path_df = pd.concat([path_df, emotions_df, actors_df], axis=1)
    
    # Podmienienie etykiet numerycznych na tekstowe
    emot_path_df['Emotions'] = emot_path_df['Emotions'].replace(
        {1:'neutral', 2:'neutral', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}
    )
    
    emot_path_df = emot_path_df.drop_duplicates()
    return emot_path_df

def split_by_actors(emot_path_df, test_size=0.2, random_seed=42):
    """Dzieli zbiór na treningowy i testowy bez wycieku danych (aktorzy się nie pokrywają)."""
    actors = emot_path_df['Actor'].unique()
    rng = np.random.default_rng(seed=random_seed)
    
    test_actors = rng.choice(actors, size=int(test_size * len(actors)), replace=False)
    train_actors = [a for a in actors if a not in test_actors]

    train_df = emot_path_df[emot_path_df['Actor'].isin(train_actors)]
    test_df = emot_path_df[emot_path_df['Actor'].isin(test_actors)]
    
    return train_df, test_df