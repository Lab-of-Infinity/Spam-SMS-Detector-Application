import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from PIL import Image

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Title Block with Image
col3, col4, col5 = st.columns(3)
with col3:
    image = Image.open('sms - Copy.jpeg')
    st.image(image, width=200)
with col4:
    st.title('**Spam SMS \tDetector**')
    st.markdown("---")

with col5:
    image = Image.open('email spam.png')
    st.image(image, width=250)

# Input SMS Block
input_sms = st.text_area('Enter the SMS')
if st.button('Predict SMS'):

    # 1. PREPROCESS
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Display Result
    # If block for spam sms
    if result == 1:
        st.success('This is a Spam message!')
        # st.header('It is Spam SMS')
        col1, col2 = st.columns(2)

        with col1:
            image = Image.open('spam2.jpeg')
            st.image(image, width=300)

        with col2:
            st.subheader("\n\n")
            st.subheader("This SMS looks Fraudulent...")
            st.write("\n")
            st.write("Beware & Stay Away From Sharing Any Personal or Financial Information !!!")
            st.snow()
    # Else block for NOT Spam Mail
    else:
        # st.header('Not Spam SMS')
        st.success('This is a Not Spam message!')
        col1, col2, col6 = st.columns(3)

        with col1:
            image = Image.open('its sms.webp')
            st.image(image, width=200)

        with col2:
            st.subheader("\n\n")
            st.subheader("This SMS is Safe ...")
            # st.subheader("No Need To Worry !!!")
            st.balloons()
        with col6:
            st.subheader("\n\n\n")
            st.subheader("***No Need to Worry !***")
            pass

# with st.container():
# streamlit config show > ~/.streamlit/config.toml

# Sidebar Block
with st.sidebar:
    st.header("About")
    image = Image.open('spam11.jpg')
    st.image(image, use_column_width='auto')
    # st.subheader("\n")

    st.write("***Daily we get a tons of sms, while some SMS are helpful and some message are Spam. In day to day "
             "hustle of "
             "life we don't have enough time to recognize whether these SMS is spam or ham , "
             "So We Created this application is for your help."
             "Hope this application is helpful to you.***")

    st.header("Contact or Feedback")
    st.write(
        """ - **For any kind of suggestions/ issue in Application Please mail me at lokeshbaviskar4@gmail.com.**""")

# Expander block with technical details
with st.expander("See Technical Details of Application"):
    st.write(""" - **A Natural Language Processing with SMS Data to predict whether the SMS is Spam/Ham using Multinomial-naive-bayes
    ML Algorithm.** """)
    st.write(""" - **Dataset : Trained on SMS Spam Dataset created by UCI Machine Learning.**""")
    st.write(""" - **Vectorization : TFIDF Vectorization performed over dataset with max features of 3000.**""")
    st.write("""- **Evaluation Matrix : Primary Evaluation Criteria is Precision Score (Data is imbalanced).**""")
    st.write(""" - **Mean Cross-validation Precision  Score : 99.3087 %.**""")
    st.write(""" - **Accuracy Score of MultinomialNB Classifier : 98.2591 %.**""")
