import pandas as pd
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import seaborn as sb

with open('kmeans_model.pkl', 'rb') as file:
    kmeans = pickle.load(file)

with open('scaler_model.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('encoder_model.pkl', 'rb') as file:
    encoder = pickle.load(file)

with open('pca_model.pkl', 'rb') as file:
    pca = pickle.load(file)

#

st.set_page_config(layout="wide")
data_og = pd.read_excel('og.xlsx')
pred_cluster = None
list_feat = ['Income','Age','Education', 'Marital_Status', 'Kidhome', 'Teenhome', 'Total_Spent',
             'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases','NumStorePurchases', 'NumWebVisitsMonth','Campaign']
custom_palette = ['#FFB74D', '#81C784', '#64B5F6']

tab1, tab2, tab3 = st.tabs(["Predict", "Insights", 'Conclusion'])

with tab1:
    st.title('Model Deployment: Customer Segmentation')
    
    Income = st.number_input("Income",min_value = 0.00)
    Age = st.number_input("Age", min_value=0, step=1, format="%d")
    #Education = st.selectbox("Education Level",["Basic", "Graduation", "Master", "PhD"])
    #Marital_Status = st.selectbox("Marital Status",["Married", "Single", "Together", "Divorced",'Widow'])
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        Education = st.radio("Education Level",["Basic", "Graduation", "Master", "PhD"])
        
    with col2:
        Marital_Status = st.radio("Marital Status",["Married", "Single", "Together", "Divorced",'Widow'])

    with col3:
        Kidhome = st.radio("Kidhome", [0, 1, 2])
        
    with col4:
        Teenhome = st.radio("Teenhome", [0, 1, 2])


    if st.button("Predict"):
        if Income == 0 or Age == 0:
            st.warning("Please enter valid values :) ")
        else:
            inp = {
                    'Income': Income,
                    'Age' : Age,
                    'Education' : Education,
                    'Marital_Status' : Marital_Status,
                    'Kidhome' : Kidhome,
                    'Teenhome' : Teenhome,
                }

            data_in = pd.DataFrame( inp, index = [0] )
        
            num_cols = ['Income', 'Age', 'Kidhome', 'Teenhome']
            cat_cols = ['Education','Marital_Status']

            data_num_cols = data_in[num_cols].copy()
            data_cat_cols = data_in[cat_cols].copy()

            scaled_arr1 = scaler.transform(data_num_cols[num_cols])
            data_num1 = pd.DataFrame(scaled_arr1 , columns = num_cols)

            encoded_arr1 = encoder.transform(data_cat_cols)
            encoded_arr1 = encoded_arr1.astype(int)
            data_cat1 = pd.DataFrame(encoded_arr1, columns=encoder.get_feature_names_out(data_cat_cols.columns))


            data_n = pd.concat( [ data_num1 , data_cat1 ], axis=1 )

            cluster = kmeans.predict(pca.transform(data_n))

            pred_cluster = cluster[0]

            if pred_cluster == 0:
                st.header(f'This Customer belongs to Cluster : [{pred_cluster}] High Purchase Ability')
            elif pred_cluster == 1:
                st.header(f'This Customer belongs to Cluster : [{pred_cluster}] Moderate Purchase Ability')
            else:
                st.header(f'This Customer belongs to Cluster : [{pred_cluster}] Low Purchase Ability')
            col1, col2 = st.columns(2)
            with col1:
                if pred_cluster == 0:  
                    st.markdown("""
                        ### Cluster [0] Characteristics:
                        - **High Average Income**
                        - **99% Certain to have an Education higher than Basic**
                        - **Almost certain to not have any Kids or Teens**
                        - **Highest Total_Spent**
                        - **Super Low Discount Purchases**
                        - **Super High Store and Catalogue Purchases**
                        - **High Web Purchases**
                        - **Super Unlikely to Visit Web Page suggesting only visits to buy**
                        - **Only Customers on Average to participate in Campaign**
                    """)
                elif pred_cluster == 1:
                    st.markdown("""
                        ### Cluster [1] Characteristics:
                        - **Moderate Average Income**
                        - **90% Certain to have an Education higher than Basic**
                        - **Almost certain to have atleast one Teen**
                        - **Moderate Total_Spent**
                        - **Super High Discount Purchases**
                        - **High Store Purchases**
                        - **Moderate Catalogue Purchases**        
                        - **High Web Purchases**
                        - **Moderate Web Visits**
                        - **Customers on Average did not participate in Campaign**
                    """)
                else:
                    st.markdown("""
                        ### Cluster [2] Characteristics:
                        - **Low Average Income**
                        - **95% Certainly not a Widow**
                        - **Almost certain to have atleast one Kid**
                        - **Lowest Total_Spent**
                        - **Super Low Discount Purchases**
                        - **Moderate Store Purchases**
                        - **Super Low Catalogue Purchases**        
                        - **Super Low Web Purchases**
                        - **Super High Web Visits suggesting frequent Web Visits but no purchase**
                        - **Customers on Average did not participate in Campaign**
                    """)

            with col2:
                st.subheader('[0] : High Purchase Ability')
                st.subheader('[1] : Moderate Purchase Ability')
                st.subheader('[2] : Low Purchase Ability')
        
            cat_cols = ['Education', 'Marital_Status', 'Campaign']
            num_cols = [col for col in data_og.columns if col not in cat_cols]

            a = data_og.groupby('Cluster')[num_cols].mean()
            a = a.round(1)
        
            b = data_og.groupby('Cluster')[cat_cols].apply(lambda x: x.mode())
            b.reset_index( inplace=True)
            b.drop('level_1',axis = 1 , inplace = True)
            b.set_index('Cluster',inplace=True)

            ab = pd.concat([a,b], axis = 1)
            ab.set_index('Cluster',inplace = True)

            c = ab.loc[pred_cluster]

            st.subheader(f'Average Features of Cluster : [{pred_cluster}] ---> ')
            cols = st.columns(5) 
            counter = 0
            for feature in list_feat:
                col = cols[counter]  
                col.markdown(f"""
                    <div style="border:2px solid #4CAF50; padding:10px; margin:5px; text-align:center; border-radius:5px;">
                        <strong>{feature}</strong><br>
                        <hr style="width:50%; border-top:2px solid #4CAF50; margin: 5px auto;"> 
                        {c[feature]}
                    </div>
                """, unsafe_allow_html=True)
                counter += 1
                if counter == 5:
                    counter = 0
   
with tab2:
    col1, col2 = st.columns([2.25, 2.75]) 
    with col1:
        plt.figure(figsize=(12,7))
        st.subheader('Cluster Distribution')
        sb.countplot(x='Cluster', data=data_og , palette = custom_palette)
        plt.xlabel('')
        plt.xticks(fontsize = 21)
        plt.ylabel('')
        plt.yticks(fontsize = 15)
        st.pyplot(plt)

    with col2:
        a1 = data_og['Cluster'].value_counts()
        a2 = [a1.T]
        a3 = pd.DataFrame(a2)
        html_table = a3.to_html(classes='table table-striped', border=0)
        
        st.markdown(
        f"""
        <div style="text-align: left;">
            <div style="display: inline-block; margin: 10px;">
                {html_table}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
        )

        st.markdown("""
            ### Total Customers : 2054           
            - **Cluster 0 ---> 512 / 2054 = 24.9%**
            - **Cluster 1 ---> 942 / 2054 = 45.8%**
            - **Cluster 2 ---> 600 / 2054 = 29.2%**        
             """ )

    col1,col2 = st.columns(2)
    with col1:
        st.subheader('Cluster Average Features')
        cat_cols = ['Education', 'Marital_Status', 'Campaign']
        num_cols = [col for col in data_og.columns if col not in cat_cols]
        a = data_og.groupby('Cluster')[num_cols].mean()
        a = a.round(1)

        a1 = a[['Income','Age', 'Kidhome', 'Teenhome', 'Total_Spent']]
        html_table = a1.to_html(classes='table table-striped')

        st.markdown(
        f"""
        <div style="text-align: center;">
            <div style="display: inline-block; margin: 10px;">
                {html_table}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
        )

        
        a1 = a[['NumWebPurchases','NumCatalogPurchases','NumStorePurchases']]
        html_table = a1.to_html(classes='table table-striped')

        st.markdown(
        f"""
        <div style="text-align: center;">
            <div style="display: inline-block; margin: 10px;">
                {html_table}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
        )

        a1 = a[['NumDealsPurchases', 'NumWebVisitsMonth']]
        html_table = a1.to_html(classes='table table-striped')

        st.markdown(
        f"""
        <div style="text-align: center;">
            <div style="display: inline-block; margin: 10px;">
                {html_table}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
        )

        
    with col2:
        st.markdown("<h3 style='text-align: right;'>Overall Data Statistics</h3>", unsafe_allow_html=True)

        z = data_og.describe().loc[['mean', '25%', '50%', '75%']][['Income','Age', 'Kidhome', 'Teenhome', 'Total_Spent']]
        z = z.round(1)

        html_table = z.to_html(classes='table table-striped')

        st.markdown(
        f"""
        <div style="text-align: center;">
            <div style="display: inline-block; margin: 10px;">
                {html_table}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
        )

        z = data_og.describe().loc[['mean', '25%', '50%', '75%']][['NumWebPurchases','NumCatalogPurchases','NumStorePurchases']]
        z = z.round(1)

        html_table = z.to_html(classes='table table-striped')

        st.markdown(
        f"""
        <div style="text-align: center;">
            <div style="display: inline-block; margin: 10px;">
                {html_table}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
        )

        z = data_og.describe().loc[['mean', '25%', '50%', '75%']][['NumDealsPurchases', 'NumWebVisitsMonth']]
        z = z.round(1)

        html_table = z.to_html(classes='table table-striped')

        st.markdown(
        f"""
        <div style="text-align: center;">
            <div style="display: inline-block; margin: 10px;">
                {html_table}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
        )
    
        cat_cols = ['Education', 'Marital_Status', 'Campaign']
    
    
    st.subheader('Cluster Categorical Statistics')
    c_dict = {}

    group = data_og[data_og['Cluster'] == 0]
    for col in cat_cols:
        mode = group[col].mode()[0]  
        freq = group[col].value_counts().max()  
        total = len(group[col])  
        percent = round((freq / total) * 100,1)  
        c_dict[col] = {
        'Mode': mode,
        'Frequency': freq,
        'Percentage': percent
        }
    a = pd.DataFrame(c_dict)

    c_dict = {}

    group = data_og[data_og['Cluster'] == 1]
    for col in cat_cols:
        mode = group[col].mode()[0]  
        freq = group[col].value_counts().max()  
        total = len(group[col])  
        percent = round((freq / total) * 100, 1 ) 
        c_dict[col] = {
        'Mode': mode,
        'Frequency': freq,
        'Percentage': percent
       }
    b = pd.DataFrame(c_dict)


    c_dict = {}

    group = data_og[data_og['Cluster'] == 2]
    for col in cat_cols:
        mode = group[col].mode()[0] 
        freq = group[col].value_counts().max()
        total = len(group[col])  
        percent = round((freq / total) * 100 , 1) 
        c_dict[col] = {
            'Mode': mode,
            'Frequency': freq,
            'Percentage': percent
        }
    c = pd.DataFrame(c_dict)

    edu = 'Education'
    m = 'Mode'
    f = 'Frequency'
    p = 'Percentage'
    mar = 'Marital_Status'
    cmg = 'Campaign'
    e1 = 'Mode : ' + a[edu].loc[m] + ' // Frequency : ' + str(a[edu].loc[f]) + ' // Percentage : ' + str(a[edu].loc[p]) + '%'
    m1 = 'Mode : ' + a[mar].loc[m] + ' // Frequency : ' + str(a[mar].loc[f]) + ' // Percentage : ' + str(a[mar].loc[p]) + '%'
    c1 = 'Mode : ' + str(round(a[cmg].loc[m])) + ' // Frequency : ' + str(round(a[cmg].loc[f])) + ' // Percentage : ' + str((a[cmg].loc[p])) + '%'

    e2 = 'Mode : ' + b[edu].loc[m] + ' // Frequency : ' + str(b[edu].loc[f]) + ' // Percentage : ' + str(b[edu].loc[p]) + '%'
    m2 = 'Mode : ' + b[mar].loc[m] + ' // Frequency : ' + str(b[mar].loc[f]) + ' // Percentage : ' + str(b[mar].loc[p]) + '%'
    c2 = 'Mode : ' + str(round(b[cmg].loc[m])) + ' // Frequency : ' + str(round(b[cmg].loc[f])) + ' // Percentage : ' + str(b[cmg].loc[p]) + '%'

    e3 = 'Mode : ' + c[edu].loc[m] + ' // Frequency : ' + str(c[edu].loc[f]) + ' // Percentage : ' + str(c[edu].loc[p]) + '%'
    m3 = 'Mode : ' + c[mar].loc[m] + ' // Frequency : ' + str(c[mar].loc[f]) + ' // Percentage : ' + str(c[mar].loc[p]) + '%'
    c3 = 'Mode : ' + str(round(c[cmg].loc[m])) + ' // Frequency : ' + str(round(c[cmg].loc[f])) + ' // Percentage : ' + str(c[cmg].loc[p]) + '%'

    z = pd.DataFrame({ edu : [ e1, e2, e3 ],
                   mar : [ m1, m2, m3 ],
                   cmg : [ c1, c2, c3 ] }, index = [ 'Cluster 0', 'Cluster 1', 'Cluster 2' ] )
    
    html_table = z.to_html(classes='table table-striped')

    st.markdown(
    f"""
    <div style="text-align: left;">
        <div style="display: inline-block; margin: 10px;">
            {html_table}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
    )
    
    col1, col2 = st.columns(2)
    with col2:
        st.markdown("<h3 style='text-align: right;'>Overall Data Categorical Statistics</h3>", unsafe_allow_html=True)

        for col in cat_cols:
            mode = data_og[col].mode()[0]
            freq = data_og[col].value_counts().max()
            total = len(data_og[col])
            percent = str(round((freq / total) * 100 , 1)) + '%'

            c_dict[col] = {
                'Mode': mode,
                'Frequency': freq,
                'Percentage': percent
            }
        a = pd.DataFrame(c_dict)
    
        html_table = a.to_html(classes='table table-striped')

        st.markdown(
        f"""
        <div style="text-align: right;">
            <div style="display: inline-block; margin: 10px;">
                {html_table}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Feature Visual Insights</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1: 
        feature = 'Education'
        fig, ax = plt.subplots(figsize=(7,5))
        ax = sb.countplot(data=data_og, x = feature, hue='Cluster', ax=ax, palette=custom_palette)
        ax.set_title(f'{feature} Count Plot')  
        for x in ax.containers:
            ax.bar_label(x)
        st.pyplot(fig)
    with col2:  
        feature = 'Marital_Status'
        fig, ax = plt.subplots(figsize=(7,5))
        ax = sb.countplot(data=data_og, x = feature, hue='Cluster', ax=ax, palette=custom_palette)
        ax.set_title(f'{feature} Count Plot')  
        for x in ax.containers:
            ax.bar_label(x)
        st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
            feature = 'Income' 
            plt.figure(figsize=(7,5))
            sb.kdeplot(data=data_og, x = feature, hue='Cluster', fill=True, palette=custom_palette)
            plt.title(f"{feature} KDE Plot")
            st.pyplot(plt) 
    with col2:  
            feature = 'Total_Spent' 
            plt.figure(figsize=(7,5))
            sb.kdeplot(data=data_og, x = feature, hue='Cluster', fill=True, palette=custom_palette)
            plt.title(f"{feature} KDE Plot")
            st.pyplot(plt)

    list_feat1 = [ 'NumWebVisitsMonth','NumWebPurchases', 'NumDealsPurchases', 'NumCatalogPurchases','NumStorePurchases', 'Age', 'Kidhome', 'Teenhome', 'Campaign' ]
    
    feature = st.selectbox('Need more Insights? Select a Feature :)',list_feat1)

    if st.button("Show"):
        col1, col2 ,col3 = st.columns([0.25,0.5,0.25])
        with col2: 
            if feature in cat_cols:
                fig, ax = plt.subplots(figsize=(7,5))
                ax = sb.countplot(data=data_og, x = feature, hue='Cluster', ax=ax, palette=custom_palette)
                ax.set_title(f'{feature} Count Plot')  
                for x in ax.containers:
                    ax.bar_label(x)
                st.pyplot(fig)
            else:
                plt.figure(figsize=(7,5))
                sb.kdeplot(data=data_og, x = feature, hue='Cluster', fill=True, palette=custom_palette)
                plt.title(f"{feature} KDE Plot")
                st.pyplot(plt)        

with tab3:
    st.title("Conclusion")
    st.subheader('Campaign :') 
    st.markdown("""- **Considering the huge difference in Campaign participation...our Campaign methodologies and strategies need a huge Revision.**""")
    st.subheader('Cluster [0] Discount Purchases :') 
    st.markdown("""- **Figure out the reason behind why our Cluster [0] customers with high purchase ability are very low on discount purchases...** """)
    st.subheader('Cluster [2] Web Visits :') 
    st.markdown("""
        - **Cluster [2] customers seem to be the highest in Web Visits but lowest in Web Purchases...**
        - **Introduce Time-Limited Online Discounts and First-Time Purchase Discounts etc rectifying their Very Low Web Purchases and their Very Low Discount Purchases.**
    """)
    st.subheader('Kid and Teen Theme Packages :') 
    st.markdown("""- **Create Kid themed and Teen Themed Package deals like BournVita + Cricket Bat = Offer.** """)
    st.subheader('Membership Program :') 
    st.markdown("""    
        - **Develop a Membership program where points are added for each purchase via any form.** 
        - **And during Campaign, provide more exciting ways to redeem points, effectively increasing Purchase frequency and Campaign participation.**
    """)
