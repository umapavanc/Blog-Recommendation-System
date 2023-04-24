from flask import Flask, render_template,request
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

app = Flask(__name__)
# app.model = pickle.load('pipeline.pkl')
df_path  = 'C:/Users/student/Desktop/blog_recommendation/cleaned_data.csv'
model_path = 'C:/Users/student/Desktop/blog_recommendation/classifier_model.sav'
vec_parth = "C:/Users/student/Desktop/blog_recommendation/vectorizer.sav"
df = pd.read_csv(df_path)

model = pickle.load(open(model_path, 'rb'))
vectorizer = pickle.load(open(vec_parth, 'rb'))

def get_top_10(similarities, filtered_df):
    # Get the indices of the top 10 highest cosine similarity values
    similar_indices = (-similarities[0]).argsort()[:10]
    # Get the corresponding texts from your dataframe
    similar_blogs = filtered_df.iloc[similar_indices]
    return similar_blogs

def recommend_blogs(user_input):
    test = vectorizer.transform([user_input])
    prediction = model.predict(test)
    filtered_df = df[df['category'] == prediction[0]]
    filtered_df = filtered_df.reset_index(drop=True)
    filtered_df_tr = vectorizer.transform(filtered_df['title'])
    similarities = cosine_similarity(test, filtered_df_tr)
    similar_blogs = get_top_10(similarities, filtered_df)
    #print(similar_blogs['URL'])
    #print(similar_blogs['title'])
    print(similar_blogs.shape)
    return similar_blogs['title'], similar_blogs['URL']

def pre_process(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_input = request.form.get("blog_title")
        blogs, urls = recommend_blogs(user_input)
        print(blogs)
        print("blogs type:",type(blogs))
        print("urls type:",type(urls))
        return render_template("index.html", blogs=blogs.tolist(), urls=urls.tolist())

    else:
        return render_template("index.html")