import streamlit as st
from helper import load_models,load_vectoriser,inference,progressbar,plot,clean_tweet,get_sentiment,twint_to_pandas,predict_url,convert_df,read_markdown_file
import twint # To scrape through Twitter for data
import streamlit as st
from wordcloud import WordCloud
# from importer import *
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('pros_cons')
nltk.download('reuters')
nltk.download('universal_tagset')
nltk.download('snowball_data')
nltk.download('rslp')
nltk.download('porter_test')
nltk.download('vader_lexicon')
nltk.download('treebank')
nltk.download('dependency_treebank')
nltk.download('maxent_treebank_pos_tagger')
nltk.download('omw-1.4')


def main():
    # page info setup
    menu_items = {
        'Get help':'https://www.linkedin.com/in/pass/' ,
        'Report a bug': 'https://www.linkedin.com/in/pass/',
        'About': '''
        ## My Custom App
        Some markdown to show in the About dialog.
        '''}

    st.set_page_config(page_title="MonkeyPox Sentiment analyzer", page_icon="./images/icons8-twitter-avantgarde-120.png", layout='centered',menu_items=menu_items)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    
    #Loading models
    model= load_models()
    vectoriser=load_vectoriser()
    
    # Removing streamlit water mark
    hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # Site Title
    st.markdown("<h1 style = 'Text-align:Center; color:black; font-size: 30px;'>MonkeyPox Sentiment Analyzer App</h1>", unsafe_allow_html=True)
    
    # creating tabs for navigation
    tab1, tab2, tab3 = st.tabs(["Text Prediction ", "URL Prediction", "About Project"])
    with tab1:
        st.subheader("Enter single/multiple tweets separated by semicolon; ")
        tweets = st.text_area("Samples below:", value="Monkeypox vaccine supply now sufficient;Monkeypox ramping up during mosquito bite season is really just nature is truly gonna fuck with us while it gets us on up outta here;I read the book you talked about,its so bad", height=300, max_chars=None, key=None)
        cols = ["tweet"]
     
        if (st.button('Predict Sentiment')):   
            progressbar()
            
            result_df = inference(vectoriser,model, tweets, cols)
            st.table(result_df)
            st.text("")
            st.text("")
            st.text("")
            plot(result_df)
            
            
    with tab2:
        with st.form(key='my_form'):
            st.subheader('Enter Search Term')
            search_term = st.text_input('Type search term',value="MonkeyPox")
            limit= st.slider('Number of Tweets', 10, 50, 100)
            submit_button = st.form_submit_button(label='Predict Sentiment')
        
        if submit_button is False:
            st.markdown("<h5 style='text-align: left; color: black;'>Please A Enter Search Term for Processing ⏳</h3>",unsafe_allow_html=True)
            
        else:
            progressbar()
            # Configure
            config = twint.Config()  # Set up TWINT config
            config.Search = str(search_term)
            config.Limit = int(limit)
            config.Lang = 'en'
            config.Pandas = True

            twint.run.Search(config)
           
            st.text("")

            df_pd = twint_to_pandas(["date", "username", "tweet"])   
            df_pd['sentiment'] = df_pd.tweet.apply(lambda twt: get_sentiment(twt))

            st.write("Number of Tweet gotten from the API is",len(df_pd))
            
            # Storing text and sentiment data in lists for further processing
            text = list(df_pd['tweet'])
    
            df = predict_url(vectoriser,model,text)
            df.sort_values(by=['Probability(Confidence Level)'],ascending=False,inplace=True)
            st.write("Dataframe containing tweet, sentiment and the probability confidence")
            st.dataframe(df)
            csv = convert_df(df)
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='sentiment_dataset.csv',
                mime='text/csv')         
            st.text("")
            st.text("")
            st.text("")
            plot(df)
            st.snow()


            st.subheader('Wordcloud')
            cloud_tweets = df_pd.tweet.apply(lambda twt: clean_tweet(twt))
            wordcloud = WordCloud(background_color = 'white', width = 800, height = 350,).generate(str(cloud_tweets))
            st.image(wordcloud.to_array())

        
    with tab3:
        markdown = read_markdown_file("about.md")
        st.markdown(markdown, unsafe_allow_html=True)   

if __name__ == '__main__':
    main()