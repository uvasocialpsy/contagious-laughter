# Packages
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np 
import time
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.io.wavfile as wavfile
import plotly.express as px
import seaborn as sns
from joypy import joyplot
import streamlit as st
from streamlit_plotly_events import plotly_events
import streamlit.components.v1 as components
import plotly.io as pio
pio.templates.default = "plotly"

# Dashboard setup
st.set_page_config(layout="wide")

# Data
@st.cache(allow_output_mutation=True)
def import_data():
    df = pd.read_excel('df_project2_nodubs.xlsx')
    return df

df = import_data()
df['SpeakerSex'] = df.SpeakerSex.replace(['m', 'f'], ['Male', 'Female'])
df['Avg_rating'] = df['Clip_ID'].map(df.groupby('Clip_ID')['Rating'].mean().to_dict())
df['Std_rating'] = df['Clip_ID'].map(df.groupby('Clip_ID')['Rating'].std().to_dict())


# App
# Intro
st.title("Welcome to Contagious Laughter Analytics App")
st.markdown("***")

# Clip average rating - std
st.markdown("## Clip Contagiousness Analysis")

# Divide for comparison
sorting = st.radio(
    "Select sorting type:",
    ('From most contagious 😂 to least contagious 😐', 'From least contagious 😐 to most contagious 😂'))
cm = sns.color_palette("YlOrRd", as_cmap=True)

if sorting == 'From most contagious 😂 to least contagious 😐':
    st.dataframe(df[['Clip_ID', 'Avg_rating', 'Std_rating']].drop_duplicates().sort_values(by = 'Avg_rating', ascending = False).reset_index(drop = True).rename(columns = {'Avg_rating':'Average Rating', 'Std_rating':'Rating Standard Deviation'}).style.background_gradient(cmap=cm), use_container_width = True)
else:
    st.dataframe(df[['Clip_ID', 'Avg_rating', 'Std_rating']].drop_duplicates().sort_values(by = 'Avg_rating', ascending = True).reset_index(drop = True).rename(columns = {'Avg_rating':'Average Rating', 'Std_rating':'Rating Standard Deviation'}).style.background_gradient(cmap=cm), use_container_width = True)



#st.success("""
#As color scale goes from **red** to **yellow** "Average Rating" of the clip increases. Default 15 clips are showed on plot and table. If you want to select more or remove clips please check filters section above 👆🏽
#""", icon="⚠️")
# Predictive Analytics
st.markdown("***")
st.markdown("## Predictive Features Analysis")

# Violin - Boxplot

predictive_1, predictive_2 = st.columns([1,3])
with predictive_1:
    with st.form('Predictive Filterer'):
        predictive_features = st.multiselect('Select feature(s)👇🏻', options = ['entropy_mean' , 'entropy_sd', 'roughness_mean', 'HNR_mean', 'HNRVoiced_mean', 'CPP', 'pitch_mean', 'pitch_sd', 'scog_mean', 'scog_sd', 'Duration'], default = ['entropy_mean' , 'entropy_sd'])
        submitted = st.form_submit_button("Submit Filters")

with predictive_2:
    df_high_rating = df[df['Clip_ID'].isin((df.groupby('Clip_ID')['Rating'].mean()[df.groupby('Clip_ID')['Rating'].mean() > 2.5]).index)].reset_index(drop = True).drop(['ListenerID', 'Rating', 'CulturalGroup', 'SpeakerSex'], axis = 1).drop_duplicates().reset_index(drop = True)
    df_high_rating = df_high_rating.loc[:, ['Clip_ID','entropy_mean' , 'entropy_sd', 'roughness_mean', 'HNR_mean', 'HNRVoiced_mean', 'CPP', 'pitch_mean', 'pitch_sd', 'scog_mean', 'scog_sd', 'Duration']].assign(rating = 'Higer Than 2.5')
    df_low_rating = df[df['Clip_ID'].isin((df.groupby('Clip_ID')['Rating'].mean()[df.groupby('Clip_ID')['Rating'].mean() < 2.5]).index)].reset_index(drop = True).drop(['ListenerID', 'Rating', 'CulturalGroup', 'SpeakerSex'], axis = 1).drop_duplicates().reset_index(drop = True)
    df_low_rating = df_low_rating.loc[:, ['Clip_ID','entropy_mean' , 'entropy_sd', 'roughness_mean', 'HNR_mean', 'HNRVoiced_mean', 'CPP', 'pitch_mean', 'pitch_sd', 'scog_mean', 'scog_sd', 'Duration']].assign(rating = 'Lower Than 2.5')
    df_predictive = pd.concat([df_high_rating, df_low_rating], axis = 0, ignore_index = True)
    df_predictive = df_predictive.melt(id_vars = ['Clip_ID', 'rating'], var_name = 'Features', value_name = 'Values')
    fig = px.violin(df_predictive.loc[df_predictive['Features'].isin(predictive_features),:], 
        y="Values", 
        x="Features", 
        color="rating", 
        box=True, 
        points="all", 
        title = 'Distribution of Predictive Features Among Contagious and Non-Contagious Clips',
        labels={
                     "rating": "Rating",
                     "Features": ""
                 })
    fig.update_layout(margin = {'t':40, 'l':0, 'r':0, 'b':0})
    st.plotly_chart(fig, use_container_width = True)


# Spectrogram Analysis
st.markdown("***")
st.markdown("## Spectrogram Analysis")

spectrogram_1, spectrogram_2 = st.columns(2)

with spectrogram_1:
    with st.form('Audio Filterer'):
        audio = st.selectbox('Select audio 👇🏻', 
        options = list(df.Clip_ID.str.split('c', expand = True)[0].drop_duplicates().astype(int).sort_values().apply(lambda x: str(x) + 'c.wav').unique()), 
        index = 17)
        submitted = st.form_submit_button("Submit Filters")
    Fs, aud = wavfile.read('audio/'+audio)
    fig, ax = plt.subplots()
    powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(aud, Fs=Fs)
    plt.title('Spectrogram')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.colorbar().set_label("Intensity (dB)")
    #plt.gcf().set_size_inches(4, 2)
    st.pyplot(fig)

with spectrogram_2:
    with st.form('Audio Filterer1'):
        audio = st.selectbox('Select audio 👇🏻', 
        options = list(df.Clip_ID.str.split('c', expand = True)[0].drop_duplicates().astype(int).sort_values().apply(lambda x: str(x) + 'c.wav').unique()), 
        index = 2)
        submitted = st.form_submit_button("Submit Filters")
    Fs, aud = wavfile.read('audio/'+audio)
    fig1, ax = plt.subplots()
    powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(aud, Fs=Fs)
    plt.title('Spectrogram')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.colorbar().set_label("Intensity (dB)")
    #plt.gcf().set_size_inches(4, 2)
    st.pyplot(fig1)


# TSNE Mapping
st.markdown("***")
st.markdown('## 2-D Representation of Clips (T-distributed Stochastic Neighbor Embedding (t-SNE))')
with st.expander("See detailed explanation"):
    exp1, exp2 = st.columns(2)
    with exp1:
        st.markdown("""
        Both t-SNE and K-Means unsupervised learning models are trained with below features. These features are obtained as predictive features during the study while creating machine learning models to predict contagiousness 👇🏻
        - Entropy mean
        - Entropy standard deviation
        - Roughness mean
        - HNR mean
        - HNR Voiced mean
        - CPP
        - Pitch mean
        - Pitch standard deviation
        - Scog mean
        - Scog standard deviation
        - Duration

        Detailed explanation of dimension reduction with t-SNE 👉 https://sonraianalytics.com/what-is-tsne/

        Detailed explanation of K-Means clustering algorithm 👉 https://www.javatpoint.com/k-means-clustering-algorithm-in-machine-learning
        """)
    with exp2:
        fig, ax = plt.subplots()
        # Elbow-method
        intra_cluster_var = [1386.0,
                            1157.338965368262,
                            1016.364522532645,
                            930.481534012986,
                            851.3500283465423,
                            792.328824588673,
                            756.7694003264337,
                            711.184831841256,
                            687.2356759308236,
                            652.5466600989791]
        plt.plot(list(range(1, 11)), intra_cluster_var)
        plt.xticks(ticks = list(range(1, 11)))
        plt.axvline(x = 5, color = 'red')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Total Intra-Cluster Variance')
        plt.title('Elbow Method for Selecting Number of Clusters')
        st.pyplot(fig)
        st.markdown("Number of cluster selection with elbow method can be conducted by visualizing total intra-cluster variation in different number of clusters. As total intra-cluster variation decreases, the obtained clusters are more homogen. In significant breakdowns (elbow-like shape) there can be potential clusters. So in above graph potential number of clusters are 2, 3, 5, 6, 7 and 8. In 5 clusters there were almost equal distribution of clips per clusters which is the reason of chosen number of clusters.")
df_tsne = pd.read_csv('tsne_df.csv')
df_tsne['cluster'] = df_tsne['cluster'].replace(['Cluster 3', 'Cluster 1', 'Cluster 4', 'Cluster 2', 'Cluster 5'], ['Highest Contagiousness', 'High Contagiousness', 'Medium Contagiousness', 'Low Contagiousness', 'Lowest Contagiousness'])
df_tsne.rename(columns = {'Rating':'Contagiousness Rating'}, inplace = True)
tsne_1, tsne_2 = st.columns(2)

with tsne_1:
    fig_tsne = px.scatter(df_tsne, 
        x = 'Embed 1', 
        y = 'Embed 2', 
        color = 'cluster', 
        category_orders = {'cluster':['Highest Contagiousness', 'High Contagiousness', 'Medium Contagiousness', 'Low Contagiousness', 'Lowest Contagiousness']}, 
        hover_name = 'Clip_ID',
        title = '2-D Representation of Predictive Features')

    fig_tsne.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        })
    fig_tsne.update_xaxes(showgrid=True, gridwidth = 1, gridcolor = '#ECECEC')
    fig_tsne.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#ECECEC')
    selected_points = plotly_events(fig_tsne, click_event = True, hover_event = False)
    st.markdown("Please select a point (clip) above 👆🏽 to observe the acoustic details and press play button below to listen 👇🏻")

with tsne_2:
    kmeans_filter = st.selectbox('Please select a feature to make sense of clusters 👇🏻', options = ['Contagiousness Rating', 'entropy_mean', 'entropy_sd', 'roughness_mean', 'HNR_mean',
                                                                                                    'HNRVoiced_mean', 'CPP', 'pitch_mean', 'pitch_sd', 'scog_mean',
                                                                                                    'scog_sd', 'Duration'])
    fig_kmeans = px.bar(df_tsne.groupby('cluster')[kmeans_filter].mean().reset_index().rename(columns = {kmeans_filter:f'Average {kmeans_filter}'}).round(3), y = 'cluster', x = f'Average {kmeans_filter}', color = 'cluster', text = f'Average {kmeans_filter}',
                        title = f'Average {kmeans_filter} per Cluster <br><sup>Clusters are Obtained from K-means Clustering Algorithm</sup>',
                        category_orders = {'cluster':['Highest Contagiousness', 'High Contagiousness', 'Medium Contagiousness', 'Low Contagiousness', 'Lowest Contagiousness']}, )
    fig_kmeans.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    fig_kmeans.update_xaxes(showgrid=True, gridwidth = 1, gridcolor = '#ECECEC')
    fig_kmeans.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#ECECEC')
    plotly_events(fig_kmeans)



try:
    st.write(df_tsne[(df_tsne['Embed 1'] == selected_points[0]["x"]) & (df_tsne['Embed 2'] == selected_points[0]["y"])])
except:
    st.write(df_tsne[df_tsne['Clip_ID'] == '18c.wav'])

try:
    st.audio(f"audio/{df_tsne[(df_tsne['Embed 1'] == selected_points[0]['x']) & (df_tsne['Embed 2'] == selected_points[0]['y'])]['Clip_ID'].values[0]}")
    #@st.cache
    #def play_audio():
    #    html_string = f"""
    #        <audio controls autoplay>
    #            <source src="https://audio-contagious.s3.eu-central-1.amazonaws.com/{df_tsne[(df_tsne['Embed 1'] == selected_points[0]['x']) & (df_tsne['Embed 2'] == selected_points[0]['y'])]['Clip_ID'].values[0]}" type="audio/wav">
    #        </audio>
    #        """
    #    return html_string
    #components.html(play_audio())
    
except:
    st.audio('audio/18c.wav')
    #html_string = f"""
    #    <audio controls autoplay>
    #        <source src="https://audio-contagious.s3.eu-central-1.amazonaws.com/18c.wav" type="audio/wav">
    #    </audio>
    #    """
    #components.html(html_string)
    #pass