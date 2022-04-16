import joblib
import requests
import os

def get_model(model_path):
    
    try:
        with open(model_path, "rb") as mh:
            rf = joblib.load(mh)
    except:
        print("Cannot fetch model from local downloading from drive")
        if not 'extree_tuned_classifier.joblib' in os.listdir('.'):
            # example url: "https://drive.google.com/u/1/uc?id=18IxYOI-whucBTZmt5qTvvYgjlxleaSqO&export=download&confirm=t"
            url = "https://drive.google.com/file/d/1B5F8sfWGutGG9DoA0zBtobJ0GLzYi5W5/view?usp=sharing"
            r = requests.get(url, allow_redirects=True)
            open(r"extree_tuned_classifier.joblib", 'wb').write(r.content)
            del r
        with open(r"extree_tuned_classifier.joblib", "rb") as m:
            rf = joblib.load(m)
    return rf
get_model(model_path = r'Model/extree_tuned_classifier.joblib')
