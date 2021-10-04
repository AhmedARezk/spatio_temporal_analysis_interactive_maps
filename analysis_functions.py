#!/usr/bin/env python
# coding: utf-8

# In[1]:


def hotspots(variable, perms):
    
    
    '''
    The first section Imports all the libraries and dependencies needed for the analysis
    '''    
    # base libraries
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import shapely as shp
    get_ipython().magic('matplotlib inline')
    # plotly and cufflinks
    import chart_studio.plotly as py
    import plotly.express as px
    import cufflinks as cf
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
    init_notebook_mode(connected = True)
    cf.go_offline()
    import plotly.graph_objects as go
    from ipywidgets import HBox, VBox,Layout
    from ipywidgets.embed import embed_minimal_html, dependency_state
    # Pysal and spatial analysis
    import pysal
    import libpysal.weights as weights
    import esda


    '''
    This section prepares the datasets
    '''  
    
    y1 = pd.read_csv('./datasets_2/1437.csv')
    y2 = pd.read_csv('./datasets_2/1438.csv')
    y3 = pd.read_csv('./datasets_2/1439.csv')
    y1.dropna(inplace=True)
    y2.dropna(inplace=True)
    y3.dropna(inplace=True)
    
    if variable == 'injuries':
        df1 = y1.pivot_table(index = 'region', columns = 'month', values = 'injuries')[['Moharram', 'Safar', 'Rabeea Awal', 'Rabeea Thany', 'Jamad Awal',
           'Jamad Thany', 'Ragab', ' Shaaban', ' Ramadan', ' Shawwal',
           'Zy Aqeaada', 'Zy Alhegga']].reset_index()
        df2 = y2.pivot_table(index = 'region', columns = 'month', values = 'injuries')[['Moharram', 'Safar', 'Rabeea Awal', 'Rabeea Thany', 'Jamad Awal',
           'Jamad Thany', 'Ragab', ' Shaaban', ' Ramadan', ' Shawwal',
           'Zy Aqeaada', 'Zy Alhegga']].reset_index()
        df3 = y3.pivot_table(index = 'region', columns = 'month', values = 'injuries')[['Moharram', 'Safar', 'Rabeea Awal', 'Rabeea Thany', 'Jamad Awal',
           'Jamad Thany', 'Ragab', ' Shaaban', ' Ramadan', ' Shawwal',
           'Zy Aqeaada', 'Zy Alhegga']].reset_index()
        overview = gpd.read_file('./overview.geojson')
        df1 = pd.merge(overview[['NAME','TOTPOP_CY','geometry']],df1,
             left_on= 'NAME', right_on='region', right_index=False).drop(columns = 'region')
        df2 = pd.merge(overview[['NAME','TOTPOP_CY','geometry']],df2,
                 left_on= 'NAME', right_on='region', right_index=False).drop(columns = 'region')
        df3 = pd.merge(overview[['NAME','TOTPOP_CY','geometry']],df3,
                 left_on= 'NAME', right_on='region', right_index=False).drop(columns = 'region')
    elif variable == 'mortalities':
        df1 = y1.pivot_table(index = 'region', columns = 'month', values = 'mortalities')[['Moharram', 'Safar', 'Rabeea Awal', 'Rabeea Thany', 'Jamad Awal',
               'Jamad Thany', 'Ragab', ' Shaaban', ' Ramadan', ' Shawwal',
               'Zy Aqeaada', 'Zy Alhegga']].reset_index()
        df2 = y2.pivot_table(index = 'region', columns = 'month', values = 'mortalities')[['Moharram', 'Safar', 'Rabeea Awal', 'Rabeea Thany', 'Jamad Awal',
               'Jamad Thany', 'Ragab', ' Shaaban', ' Ramadan', ' Shawwal',
               'Zy Aqeaada', 'Zy Alhegga']].reset_index()
        df3 = y3.pivot_table(index = 'region', columns = 'month', values = 'mortalities')[['Moharram', 'Safar', 'Rabeea Awal', 'Rabeea Thany', 'Jamad Awal',
               'Jamad Thany', 'Ragab', ' Shaaban', ' Ramadan', ' Shawwal',
               'Zy Aqeaada', 'Zy Alhegga']].reset_index()
        overview = gpd.read_file('./overview.geojson')
        df1 = pd.merge(overview[['NAME','TOTPOP_CY','geometry']],df1,
                 left_on= 'NAME', right_on='region', right_index=False).drop(columns = 'region')
        df2 = pd.merge(overview[['NAME','TOTPOP_CY','geometry']],df2,
                 left_on= 'NAME', right_on='region', right_index=False).drop(columns = 'region')
        df3 = pd.merge(overview[['NAME','TOTPOP_CY','geometry']],df3,
                 left_on= 'NAME', right_on='region', right_index=False).drop(columns = 'region')
        
    elif variable == 'involvement':
        df1 = y1.pivot_table(index = 'region', columns = 'month', values = 'total number of people involved in RTA')[['Moharram', 'Safar', 'Rabeea Awal', 'Rabeea Thany', 'Jamad Awal',
               'Jamad Thany', 'Ragab', ' Shaaban', ' Ramadan', ' Shawwal',
               'Zy Aqeaada', 'Zy Alhegga']].reset_index()
        df2 = y2.pivot_table(index = 'region', columns = 'month', values = 'total number of people involved in RTA')[['Moharram', 'Safar', 'Rabeea Awal', 'Rabeea Thany', 'Jamad Awal',
               'Jamad Thany', 'Ragab', ' Shaaban', ' Ramadan', ' Shawwal',
               'Zy Aqeaada', 'Zy Alhegga']].reset_index()
        df3 = y3.pivot_table(index = 'region', columns = 'month', values = 'total number of people involved in RTA')[['Moharram', 'Safar', 'Rabeea Awal', 'Rabeea Thany', 'Jamad Awal',
               'Jamad Thany', 'Ragab', ' Shaaban', ' Ramadan', ' Shawwal',
               'Zy Aqeaada', 'Zy Alhegga']].reset_index()
        overview = gpd.read_file('./overview.geojson')
        df1 = pd.merge(overview[['NAME','TOTPOP_CY','geometry']],df1,
                 left_on= 'NAME', right_on='region', right_index=False).drop(columns = 'region')
        df2 = pd.merge(overview[['NAME','TOTPOP_CY','geometry']],df2,
                 left_on= 'NAME', right_on='region', right_index=False).drop(columns = 'region')
        df3 = pd.merge(overview[['NAME','TOTPOP_CY','geometry']],df3,
                 left_on= 'NAME', right_on='region', right_index=False).drop(columns = 'region')
        
    months = ['Moharram', 'Safar', 'Rabeea Awal',
               'Rabeea Thany', 'Jamad Awal', 'Jamad Thany', 'Ragab', ' Shaaban',
               ' Ramadan', ' Shawwal', 'Zy Aqeaada', 'Zy Alhegga']
    
    
    '''
    The following function conducts the analysis, and exports the plots as separate html files
    '''    
        
    lok = locals()

    colors = ["#2389b9", "#4fc8a7", "#e4f79d" , "#ffffc4" , "#fee09a" , "#fead6e" , "#ff644b" , "#d93e4e"]
    colors_m = ["#2389b9", "#fee09a" , "#ff644b" , "#d93e4e"]

    np.random.seed(1992)
    w = weights.Kernel.from_dataframe(df1, k=5, fixed = False, function = 'gaussian')
    w.transform = 'B'


    df1_pc = df1.copy()
    df2_pc = df2.copy()
    df3_pc = df3.copy()
    for i in np.arange(3,15):
        df1_pc.iloc[:,i] = (df1_pc.iloc[:,i]/ (df1_pc.TOTPOP_CY))*100000
        df2_pc.iloc[:,i] = (df2_pc.iloc[:,i]/ (df2_pc.TOTPOP_CY))*100000
        df3_pc.iloc[:,i] = (df3_pc.iloc[:,i]/ (df3_pc.TOTPOP_CY))*100000


    lok[variable+'_all_gi'] = df1[['NAME','geometry']].copy()
    lok[variable+'_all_mi'] = df1[['NAME','geometry']].copy()
    lok[variable+'_all_pc_gi'] = df1[['NAME','geometry']].copy()
    lok[variable+'_all_pc_mi'] = df1[['NAME','geometry']].copy()


    # the hotspot analysis

    for n,year in zip([1437,1438,1439],[df1,df2,df3]):
        for month in months:
            y = year[month].values
            #the analysis gi and moran
            getisord = esda.G_Local(y, w, permutations = perms, star = True)
            moran_loc = esda.moran.Moran_Local(y, w, permutations = perms)
            # setting column names
            column = month + ' ' + str(n)
            #getting the values for gi
            Zs_gi = getisord.Zs
            Zs_gi[getisord.p_sim>0.05]= np.NaN
            lok[variable+'_all_gi'][column]= Zs_gi
            #getting the values for moran i
            lok[variable+'_all_mi']['clustering'] = moran_loc.q
            lok[variable+'_all_mi']['p_sim'] = moran_loc.p_sim
            lok[variable+'_all_mi'][column]= lok[variable+'_all_mi']['clustering'].map({1: 1, 2: 0.5, 3: 0.25, 4: 0.75})
            lok[variable+'_all_mi'].loc[ (lok[variable+'_all_mi'].p_sim > 0.05) , column]= np.NaN
            lok[variable+'_all_mi'].drop(columns = ['p_sim','clustering'], inplace = True)

    for n,year in zip([1437,1438,1439],[df1_pc,df2_pc,df3_pc]):
        for month in months:
            y = year[month].values
            #the analysis gi and moran
            getisord = esda.G_Local(y, w, permutations = perms, star = True)
            moran_loc = esda.moran.Moran_Local(y, w, permutations = perms)
            # setting column names
            column = month + ' ' + str(n)
            #getting the values for gi
            Zs_gi = getisord.Zs
            Zs_gi[getisord.p_sim>0.05]= np.NaN
            lok[variable+'_all_pc_gi'][column]= Zs_gi
            #getting the values for moran i
            lok[variable+'_all_pc_mi']['clustering'] = moran_loc.q
            lok[variable+'_all_pc_mi']['p_sim'] = moran_loc.p_sim
            lok[variable+'_all_pc_mi'][column]= lok[variable+'_all_pc_mi']['clustering'].map({1: 1, 2: 0.5, 3: 0.25, 4: 0.75})
            lok[variable+'_all_pc_mi'].loc[ (lok[variable+'_all_pc_mi'].p_sim > 0.05) , column]= np.NaN
            lok[variable+'_all_pc_mi'].drop(columns = ['p_sim','clustering'], inplace = True)




    #prepare the heatmaps dataframe
    for i in [lok[variable+'_all_gi'], lok[variable+'_all_mi'],lok[variable+'_all_pc_gi'], lok[variable+'_all_pc_mi']]:
        i.drop(columns = ['geometry'], inplace = True)
        i.set_index('NAME', inplace = True)
        i = pd.DataFrame(i)
        i = i.loc[['Madinah','Makkah','Jiddah','At-Taif', 'Al-Baha','Aseer', 'Jazan', 'Najran','Tabouk', 
             'Al-Jouf', 'Al-Qurayyat', 'Northern Region','Hail','Qaseem','Ash Sharqiyah','Ar Riyad'],:]

    # plotting the graphs

    # first prepare the color scheme for local moran
    def discrete_colorscale(bvals, colors):
        if len(bvals) != len(colors)+1:
            raise ValueError('len(boundary values) should be equal to  len(colors)+1')
        bvals = sorted(bvals)     
        nvals = [(v-bvals[0])/(bvals[-1]-bvals[0]) for v in bvals]  #normalized values

        dcolorscale = [] #discrete colorscale
        for k in range(len(colors)):
            dcolorscale.extend([[nvals[k], colors[k]], [nvals[k+1], colors[k]]])
        return dcolorscale

    bvals = [0, 0.25, 0.5, 0.75, 1]
    dcolorsc = discrete_colorscale(bvals, colors_m)



    dataframes = [lok[variable+'_all_gi'], lok[variable+'_all_pc_gi'], lok[variable+'_all_mi'], lok[variable+'_all_pc_mi']]
    counter = 1
    
    for dataframe in dataframes:
        fig = go.Figure()
        if counter < 3:
            fig.add_trace(go.Heatmap(y = dataframe.index.values,
                                    x = dataframe.columns.values,
                                    z = dataframe.values,
                                     colorscale = colors,
                                     colorbar = dict(title="Gi*",
                                                     thicknessmode="pixels", thickness=15,
                                                     lenmode="pixels", len=500,
                                                     #yanchor="bottom",dtick=10
                                                    ),
                                    hovertemplate='Time: %{x}<br>Province: %{y}<br>Getis Gi* value: %{z}<extra></extra>',
                                    showscale = True))

            if counter%2 ==0:
                fig.update_layout(title = dict(text = 'Getis ord GI* values of TA ' + variable+ ' per capita',
                                       x = 0.5,
                                       y = 0.92,
                                       xanchor = 'center',
                                       font = {'family':'Calibri', 'size':30}, ))
            else:
                fig.update_layout(title = dict(text = 'Getis ord GI* values of TA ' + variable,
                                       x = 0.5,
                                       y = 0.92,
                                       xanchor = 'center',
                                       font = {'family':'Calibri', 'size':30}, ))

        else:
            fig.add_trace(go.Heatmap(y = dataframe.index.values,
                                    x = dataframe.columns.values,
                                    z = dataframe.values,
                                     colorscale = dcolorsc,
                                     colorbar = dict(title="Cluster",
                                                     thicknessmode="pixels", thickness=15,
                                                     lenmode="pixels", len=500,
                                                     tickvals = np.linspace(0.35,0.9,num = 4),
                                                     ticktext = ['LL','LH','HL','HH']),
                                    hovertemplate='Time: %{x}<br>Province: %{y}<br>Moran cluster: %{z}<extra></extra>',
                                    showscale = True))
            if counter%2 ==0:
                fig.update_layout(title = dict(text = 'Local moran clustering of TA ' + variable+ ' per capita',
                                       x = 0.5,
                                       y = 0.92,
                                       xanchor = 'center',
                                       font = {'family':'Calibri', 'size':30}, ))
            else:
                fig.update_layout(title = dict(text = 'Local moran clustering of TA ' + variable,
                                       x = 0.5,
                                       y = 0.92,
                                       xanchor = 'center',
                                       font = {'family':'Calibri', 'size':30}, ))

        xal= 6
        for year in ['1437','1438','1439']:
            fig.add_annotation(x = xal, y = 16,
                               text = '<b>Year 1: %s</b>'%year,
                               align = 'center',
                               showarrow=False,
                              font=dict(family='Calibri', size = 18))
            xal += 12

        fig.update_layout(xaxis_tickangle = -90)

        fig.update_layout(height = 700, width = 700)
        fig.update_layout(xaxis = {'showgrid':False,
                                   'tickvals': [i for i in range(36)],
                                   'ticktext': months*3},
                         yaxis = {'showgrid':False})



        x = 0.5
        # add the regions dividing lines
        fig.add_hline(y = 4.5,line_width=x, line_dash="dash", opacity = 1)
        fig.add_hline(y = 7.5,line_width=x, line_dash="dash", opacity = 1)
        fig.add_hline(y = 12.5,line_width=x, line_dash="dash", opacity = 1)


        for i in np.arange(12,25,12):
            fig.add_vline(x = i-0.5,line_width=x, line_dash="dash", opacity = 1)

        globals()['f' + str(counter)] = go.FigureWidget(fig)
        fig.write_html('f'+str(counter)+'_'+variable+'.html')
        
        counter = counter + 1


    fig_subplots =  VBox([f1, f2, f3, f4], layout=Layout(width='100%',display='inline-flex',
                                                flex_flow='column', align_items='stretch'))
    return fig_subplots


# In[ ]:




