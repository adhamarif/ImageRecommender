#!/usr/bin/env python
# coding: utf-8

# In[20]:

import os
import pandas as pd
import numpy as np
import sqlite3

# library for similarities calculation
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance

project_path = r"D:\adham-till-code"

# setup a database
conn = sqlite3.connect(os.path.join(project_path, "databases/file_path.db"))

# setup a cursor object
curs = conn.cursor()


def calculate_similarity(dataframe, img_to_compare, img_input=False):
    """
    This function is used to calculate the Euclidean distance of the images
    """
    if img_input:
        input_vector = img_to_compare
    else:
        input_vector = dataframe.loc[img_to_compare].values

    distances = distance.cdist([input_vector], dataframe.values, metric='euclidean')[0]
    return distances


# In[24]:


def top_path(result, n_path=10):
    """
    This function is used to retrieve path of the top similar images from the database.
    n_path can be adjusted with the desired number of paths.
    """
    sorted_index = np.argsort(result)[:n_path]
    ids_list = [f"{i:07}_" for i in sorted_index]

    # Construct the SQL query
    query = "SELECT * FROM image_path WHERE image_id IN ({})".format(','.join(['?']*len(ids_list)))

    # Execute the query with the ID list
    curs.execute(query, ids_list)

    # Fetch all the results
    results = curs.fetchall()

    # Create a dictionary to map image IDs to file paths
    file_path_dict = {row[0]: row[1] for row in results}

    # Create a list of file paths in the correct order
    file_paths = [file_path_dict[image_id] for image_id in ids_list]
    
    return file_paths