import pytest
import os
import useful_functions as uf
import pandas as pd

project_path = r"D:\adham-till-code"

data = pd.read_pickle(os.path.join(project_path, 'file_path.pkl'))
color = pd.read_pickle(os.path.join(project_path,'color_histogram.pkl'))
df = pd.read_pickle(os.path.join(project_path,'embeddings.pkl'))

def test_embedding_similarity():
    '''
    Test function to test if the ImageRecommender returns a similar recommendation
    example:
    2681 and 2611 in dew are most likely exact the same
    '''
    input_image = r"\images\weather_image_recognition\dew\2681.jpg"
    output_image = r"\images\weather_image_recognition\dew\2611.jpg"

    input_val = data[data["file_path"] == input_image].index.values
    embeddings_similarity = uf.calculate_similarity(df, input_val[0])
    top_embeddings = uf.top_path(embeddings_similarity)

    assert output_image in top_embeddings

def test_color_similarity():
    '''
    Test function to test if the ImageRecommender returns a similar recommendation
    example:
    2681 and 2611 in dew are most likely exact the same
    '''
    input_image = r"\images\weather_image_recognition\dew\2681.jpg"
    output_image = r"\images\weather_image_recognition\dew\2611.jpg"

    input_val = data[data["file_path"] == input_image].index.values
    embeddings_similarity = uf.calculate_similarity(color, input_val[0])
    top_embeddings = uf.top_path(embeddings_similarity)

    assert output_image in top_embeddings

def test_file_length():
    assert len(data) == len(color) == len(df)

def test_all_index():
    assert data.index.all() == color.index.all() == df.index.all()