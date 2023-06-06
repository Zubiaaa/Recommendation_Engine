# Content-based Recommendation Engine API with Flask

Below is a step-by-step guide to building a simple content-based Netflix shows recommendation engine, running it as an API with Flask:

1.	Create new folder named “Netflix Shows Recommendation API”
2.	Open a new command (or Anaconda) prompt inside the folder, or point terminal directory to its path:
cd /d <parent directory path>\Netflix Shows Recommendation API
3.	Create a new Python virtual environment:
virtualenv recommendation_api_env
4. Activate recommendation_api_env:
recommendation_api_env\Scripts\activate
5. The packages required for this project are pandas, scikit-learn, flask and gunicorn. Run the following command to batch install them:
pip install pandas sklearn flask gunicorn
6. After it has finished installing, save the project’s list of packages to a text file with this command. Heroku uses this file as reference to what packages to install:
pip freeze > requirements.txt
7. If you haven’t already, install Git:
Git - Downloading Package (git-scm.com)
8.	Download all the files in Git repo:
MAbdElRaouf/Content-based-Recommendation-Engine (github.com)
9.	Needed packages for this project:
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
10.	Getting to know the dataset:
netflix_titles_df = pd.read_csv('netflix_titles.csv')
netflix_titles_df.head()
11.	Some of the columns would have little to no impact on the user’s preference of shows (ex: country or date_added) and are better off dropped:
netflix_titles_df.drop(netflix_titles_df.columns[[0,1,5,6,7,9]], axis=1, inplace=True)
12.	Now to inspect the remaining columns for NaNs:
netflix_titles_df.count()
13.	Commonly, the next step would be to drop all rows with NaN/missing values. However, let’s first see what is the proportion of rows having NaN in any of the columns:
null_rows = len(netflix_titles_df[netflix_titles_df.isna().any(axis=1)])
print(f'Rows with NaNs: {null_rows} ({(null_rows/netflix_titles_df.shape[0])*100:.0f}%)')
14.	That’s too big a chunk of the dataset to let go of, especially when the missing values are in either director or cast columns, while description column (which has the most keywords) is always intact.
15.	Instead, we will replace NaNs with blank strings. This way we get to keep and pass row values to a text vectorizer:
netflix_titles_df.fillna('', inplace=True)
netflix_titles_df.head()
16.	This code will remove spaces between names, replace commas with spaces and finally retain only the first 3 names in the list (since the top actors/directors with most screen time are who relate the most to the work):
netflix_titles_df[['director','cast']] = netflix_titles_df[['director','cast']].applymap(lambda x: ' '.join(x.replace(' ', '').split(',')[:3]))
netflix_titles_df.head()
17.	Simply add a duplicate title column to the dataframe, making titles show twice in the vector, doubling their importance:
netflix_titles_df['title_dup'] = netflix_titles_df['title']
18.	Next, building the corpus. We do so by turning the dataframe into a series of just the index and a concatenated string of each show’s column values (features):
titles_corpus = netflix_titles_df.apply(' '.join, axis=1)
19.	The final remaining steps are to convert the strings of features to lowercase (so that words are treated equally regardless of letters case), tokenize the corpus and discard stop words (“the”, “for”, “that”, etc..). Luckily, sklearn TfidfVectorizer already takes care of this when given the right parameters:
tfidf_vectorizer_params = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 3), max_df = .5)
tfidf_vectorizer = tfidf_vectorizer_params.fit_transform(titles_corpus)
20.	This generates a large sparse matrix where every token is given its own column filled with the token’s TF-IDF score for each show. Here’s a visualization of how it looks like:
pd.DataFrame(tfidf_vectorizer.toarray(), columns=tfidf_vectorizer_params.get_feature_names())
21.	Saving it to disk so that the recommendation engine wouldn’t have to go through the steps of generating the matrix with every request:
pickle.dump(tfidf_vectorizer, open('tfidf_vectorizer.pickle', 'wb'))
22.	To determine the similarity between vectors/shows, we measure the cosine of angle between the TF-IDF vectors.
 
23.	We get the cosine similarity by taking the dot product of TF-IDF vectors:
vects_cos_sim = cosine_similarity(tfidf_vectorizer, tfidf_vectorizer)
24.	Which gives us this pairwise cosine similarity matrix:
pd.DataFrame(data=vects_cos_sim, index=netflix_titles_df['title'], columns=netflix_titles_df['title']).head()


25.	Let’s test our logic:
def recommended_shows(title):
    
    #Get show index
    title_iloc = netflix_titles_df.index[netflix_titles_df['title'] == title][0]
    
    #Get cosine similarity
    show_cos_sim = cosine_similarity(tfidf_vectorizer[title_iloc], tfidf_vectorizer).flatten()
    
    #Get the top 5 most similar shows
    sim_titles_vects = sorted(list(enumerate(show_cos_sim)), key=lambda x: x[1], reverse=True)[1:6]
    
    #Return result
    response = '\n'.join([f'{netflix_titles_df.iloc[t_vect[0]][0]} --> confidence: {round(t_vect[1],1)}' for t_vect in sim_titles_vects])
    
    return responseprint(recommended_shows('The Matrix'))

print(recommended_shows('Breaking Bad'))

26.	We’ll be saving the engine in its own module file recommendation_engine.py. Here is how it looks like after tweaking the function to return a response in JSON format:
from sklearn.metrics.pairwise import cosine_similarity	
	

	def recommended_shows(title, shows_df, tfidf_vect):
	

	    '''
	    Recommends the top 5 similar shows to provided show title.
	
	            Arguments:
	                    title (str): Show title extracted from JSON API request
	                    shows_df (pandas.DataFrame): Dataframe of Netflix shows dataset
	                    tfidf_vect (scipy.sparse.matrix): sklearn TF-IDF vectorizer sparse matrix
	
	            Returns:
	                    response (dict): Recommended shows and similarity confidence in JSON format
	    '''
	

	    try:
	

	        title_iloc = shows_df.index[shows_df['title'] == title][0]
	

	    except:
	        
	        return 'Movie/TV Show title not found. Please make sure it is one of the titles in this dataset: https://www.kaggle.com/shivamb/netflix-shows'
	

	    show_cos_sim = cosine_similarity(tfidf_vect[title_iloc], tfidf_vect).flatten()
	

	    sim_titles_vects = sorted(list(enumerate(show_cos_sim)), key=lambda x: x[1], reverse=True)[1:6]
	

	    response = {'result': [{'title':shows_df.iloc[t_vect[0]][0], 'confidence': round(t_vect[1],1)} for t_vect in sim_titles_vects]}
	    
	    return response

Flask:
1.	Download Flask:
https://www.postman.com/downloads/
2.	Open Workspaces and select Collections
3.	Create a request and accept and parse POST API requests routed through ‘/api/’ endpoint in JSON format. Example of acceptable request:
 

4.	Calls the recommendation engine and pass the above as arguments
5.	Respond to the request with the engine’s output.

from flask import Flask, request, jsonify	
	from recommendation_engine import recommended_shows #Importing engine function
	import pandas as pd
	import pickle
	

	app = Flask(__name__)
	

	#Avoid switching the order of 'title' and 'confidence' keys
	app.config['JSON_SORT_KEYS'] = False
	

	netflix_titles_df = pd.read_csv('netflix_titles.csv', usecols=[2])
	

	tfidf_vect_pkl = pickle.load(open('tfidf_vectorizer.pickle', 'rb'))
	

	#API endpoint
	@app.route('/api/', methods=['POST'])
	def process_request():
	

	    #Parse received JSON request
	    user_input = request.get_json()
	

	    #Extract show title
	    title = user_input['title']
	

	    #Call recommendation engine
	    recommended_shows_dict = recommended_shows(title, netflix_titles_df, tfidf_vect_pkl)
	

	    return jsonify(recommended_shows_dict)
	

	

	if __name__ == '__main__':
	

	    app.run()

6.	Save to recommendation_api.py and run this command:
python recommendation_api.py

7.	To test the API running locally, send a test request to the shown address (http://127.0.0.1:5000/api/ ). I did so using Postman:
 
