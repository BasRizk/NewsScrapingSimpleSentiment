# -*- coding: utf-8 -*-

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default='svg'

def visualize_flair_sentiment(sentiments_df, per_s_sentiments_df):
    df = (sentiments_df['f_t_score']*sentiments_df['f_t_value']).\
            values.reshape((1, len(sentiments_df['f_t_score'])))
            
    fig = make_subplots(rows=6, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02,
                        specs=[
                            [{"rowspan": 1}],
                            [{"rowspan": 5}],
                            [None], [None], [None], [None]
                            ])    
    fig.append_trace(
        go.Heatmap(
            x=per_s_sentiments_df.columns,
            z=df,
            text=df.round(3),
            texttemplate="%{text}",
            textfont={"size":10},
            colorscale='rdylgn',
            showlegend=False,
            showscale=False),
        row=1, col=1
    )

    fig.append_trace(
        go.Heatmap(
            x=per_s_sentiments_df.columns,
            z=per_s_sentiments_df,
            text=per_s_sentiments_df.round(3),
            texttemplate="%{text}",
            textfont={"size":10},
            colorscale='rdylgn',
            showlegend=False,
            showscale=True),
        row=2, col=1
    )
    # fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(height=1000, width=800,
                      title_text=\
                      "Flair | Total Text (Top) vs. Per " +
                      "Sentence in order (Bottom) Sentiment")
    fig.show()
    

def visualize_vader_sentiment(sentiments_df):
    fig = make_subplots(rows=3, cols=1,
                        vertical_spacing=0.02,
                        shared_xaxes=True,
                        specs=[
                            [{"rowspan": 1}],
                            [{"rowspan": 2}],
                            [None],
                            ])
    
    pos = sentiments_df[sentiments_df['v_t_compound'] >= 0]
    fig.append_trace(
        go.Bar(
            name="Postive",
            y=pos['v_t_compound'],
            text=pos['v_t_compound'],
            offsetgroup=0,
            marker_color='Green',
            x=pos.index),
        row=1, col=1
    )
    
    neg = sentiments_df[sentiments_df['v_t_compound'] < 0]
    fig.append_trace(
        go.Bar(
            name="Negative",
            y=neg['v_t_compound'],
            text=neg['v_t_compound'],
            offsetgroup=0,
            marker_color='red',
            x=neg.index),
        row=1, col=1
    )
    
    
    keys = ['v_t_neg', 'v_t_neu', 'v_t_pos']
    colors = ['red', 'blue', 'green']
    for i, (key, color) in enumerate(zip(keys, colors)):        
        offsetY=None
        if i > 0:
            offsetY = sentiments_df[keys[i-1]].copy()
            for previous_i in range(i-2, -1, -1):
                offsetY += sentiments_df[keys[previous_i]]
    
        fig.append_trace(
            go.Bar(
                name=key,
                y=sentiments_df[key],
                text=sentiments_df[key],
                x=sentiments_df.index,
                offsetgroup=0,
                base=offsetY,
                marker_color=color
                ),
            row=2, col=1
        )
        
    fig.update_layout(height=1000, width=800,
                      title_text=\
                      "VaderSentiment | Compound (Top) vs." +
                      " Stacked Proportions Based Polarity (Bottom)")        
    fig.show() 
    
        
        
        
        
        
