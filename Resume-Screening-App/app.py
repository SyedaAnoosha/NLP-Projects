import streamlit as st
import pickle
import re
import nltk

# Load the model and the tfidf vectorizer from the disk
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

def cleanResume(resumeText):
    resumeText = re.sub(r'http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub(r'RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub(r'#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub(r'@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) # remove special characters
    resumeText = re.sub(r'\s+', ' ', resumeText)  # remove extra whitespace
    resumeText = resumeText.lower()  # convert to lowercase
    return resumeText

def main():
    st.title('Resume Screening App')
    st.write('This app predicts the category of a resume based on its content.')

    resume = st.file_uploader('Upload a resume', type=['txt', 'pdf'])

    if resume is not None:  
        try:
            resume_text = resume.read().decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume.read().decode('latin-1')
        
        cleanedResume = cleanResume(resume_text)
        input_features = tfidf.transform([cleanedResume])
        prediction = model.predict(input_features)[0]
        category_map = {
            6: 'Data Science',
            12: 'HR',
            0: 'Advocate',
            1: 'Arts',
            24: 'Web Designing',
            16: 'Mechanical Engineer',
            22: 'Sales',
            14: 'Health and fitness',
            5: 'Civil Engineer',
            15: 'Java Developer',
            4: 'Business Analyst',
            21: 'SAP Developer',
            2: 'Automation Testing',
            11: 'Electrical Engineering',
            18: 'Operations Manager',
            20: 'Python Developer',
            8: 'DevOps Engineer',
            17: 'Network Security Engineer',
            19: 'PMO',
            7: 'Database',
            13: 'Hadoop',
            10: 'ETL Developer',
            9: 'DotNet Developer',
            3: 'Blockchain',
            23: 'Testing'
        }

        prediction_int = int(prediction)  

        category = category_map.get(prediction_int, "Unknown"), 
        st.write("The predicted category is: ", category)
        st.write('---'*50)

# python main

if __name__ == '__main__':
    main()
