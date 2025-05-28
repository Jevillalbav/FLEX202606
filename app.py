import streamlit as st
import pandas as pd 
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk


st.set_page_config(layout="wide", page_title="Real Estate Report", page_icon="🏠", 
                    initial_sidebar_state="expanded")

st.title('Real Estate Report 🏠'
         )

summary = pd.read_csv('data/summaries.csv', index_col=0, parse_dates=True).sort_values(by=['state', 'city', 'date_calculated'])
summary['npv/equity'] = summary['npv'] / summary['equity']

cashflow = pd.read_csv('data/cashflows.csv', index_col=0, parse_dates=True)
cashflow['state'] = cashflow['market'].str.split(',').str[1]
cashflow['city'] = cashflow['market'].str.split(',').str[0]

# st.header('Cashflow Analysis and Market Summary', divider='blue')
# st.dataframe(cashflow.head(3), use_container_width=True)

table = pd.read_csv('data/tables.csv', index_col=0, parse_dates=True)
table['state'] = table['market'].str.split(',').str[1]
table['city'] = table['market'].str.split(',').str[0]

costar = pd.read_csv('data/costar.csv', index_col=0, parse_dates=True)

# Market Filters
states = summary['state'].unique()
horizons = summary['horizon'].unique()
dates = summary['date_calculated'].unique() 

filtro_columnas_mapa = ['current_price', 'loan', 'equity', 'market_cagr',
                        'noi_cap_rate_compounded','operation_cashflow',
                        'market_cap_appreciation_bp', 'irr', 'npv',
                        'stabilized_ncf', 'max_loan_dcr_test','max_loan_debt_yield_test',
                        'headline_loan_request', 'valuation', 'cap_rate','equity_multiple', 'npv/equity',
                        ]

mapa_columns = pd.DataFrame(filtro_columnas_mapa, columns=['columnas'])
mapa_columns.index = mapa_columns['columnas'].str.replace('_',' ').replace('bp','basis point').str.title().str.replace('Yoy','YoY%').str.replace('Npv','Net Present Value').str.replace('Irr', 'IRR')
mapa_columns['unit'] = ['USD','USD','USD','%',
                        '%','%',
                        'bp','%','USD',
                        'USD','USD','USD',
                        'USD','USD','%', 'x', 'x'
                        ]

with st.expander('States Filter'):
    st.session_state.selected_states = states
    selected_states = st.multiselect('**Select States**', states, default=st.session_state.selected_states)

with st.sidebar:
    # Filters for the horizon
    st.header('Market Filters', divider= 'blue')
    horizon = st.selectbox('Select Horizon', options=horizons, index=0)
    st.session_state.horizon = horizon

    # Population filter
    box_selector_pop = ['All','+2M' , '+3M' , '+5M', '+7M', '+10M']
    population = st.selectbox('Select Population', options=box_selector_pop, index=0)
    result = population.replace('+', '').replace('.', '').replace('M', '000000').replace('K', '000').replace('All', '0')
    st.session_state.population = result

    # Filter by date
    date_filtering = st.selectbox('Select Date', options=dates, index=len(dates)-1)
    st.session_state.date_filtering = date_filtering

    # Aspect to classify
    filtro_columnas_mapa_mostrar = st.selectbox('Aspect to classify', options=mapa_columns.index, index=0)
    st.session_state.filtro_columnas_mapa = mapa_columns.loc[filtro_columnas_mapa_mostrar].values[0]
    if st.button('Reset Filters'):
        st.rerun()



    st.header('Individual Market Filters', divider= 'blue')

    # Select state and city
    selected_state = st.selectbox('Select State', selected_states, index=0)
    st.session_state.selected_state = selected_state

    selected_cities = summary[summary['state'] == selected_state]['city'].unique()
    selected_city = st.selectbox('Select City', selected_cities, index=0)
    st.session_state.selected_city = selected_city

unidad_columna = mapa_columns.loc[filtro_columnas_mapa_mostrar].values[1]

def transformar_value(column, unidad):
    if unidad in ['%', 'bp']:
        return column.round(4 if unidad == '%' else 0)
    return column.round(2 if unidad == 'x' else 0)
def valor_a_mostrar(column, unidad):
    if unidad == '%':
        return f'{column:.2%}'
    elif unidad == 'x':
        return f'{column:.2f}x'
    elif unidad == 'USD':
        return f'USD {column:,.0f}'
    elif unidad == 'bp':
        return f'{column:.0f} bp'
# Función para asignar colores basados en IRR
def get_color(value, column_name):
    if value > 90:
        return [0, 128, 0, 200]  # Verde más oscuro
    elif value > 75:
        return [144, 238, 144, 200]  # Verde super claro
    elif value > 50:
        return [173, 200, 47, 200]
    elif value > 30:
        return [255, 255, 0, 200]
    elif value > 5:
        return [255, 165, 0, 200] 
    elif value <= 5:
        return [255, 0, 0, 200]

summary_filtered = summary[
                           (summary['state'].isin(selected_states)) & 
                           (summary['population'] >= int(st.session_state.population)) &
                           (summary['horizon'] == st.session_state.horizon)
                           ].copy()
summary_filtered_map = summary_filtered[
    (summary_filtered['date_calculated'] == st.session_state.date_filtering)
    ].copy()

summary_filtered_map['value'] = transformar_value(summary_filtered_map[st.session_state.filtro_columnas_mapa], unidad_columna)
summary_filtered_map['value_show'] = summary_filtered_map['value'].apply(lambda x: valor_a_mostrar(x, unidad_columna))
##3 para el alto de la barra hago un rank y cada uno le asigno su puesto siendo 100 el mayor valor y 1 el menor
summary_filtered_map['bar_height'] =  summary_filtered_map['value'].rank(ascending=True, method='max', pct=True) * 100
summary_filtered_map['color'] = summary_filtered_map['bar_height'].apply( lambda x: get_color(x, summary_filtered_map['bar_height']))
summary_filtered_map['color_no_list'] = summary_filtered_map['color'].apply(lambda x: f'rgba({x[0]},{x[1]},{x[2]},{x[3]})')
#summary_filtered_map['bar_height'] = summary_filtered_map['bar_height'] ** 1.8
summary_filtered_map['market'] = summary_filtered_map['market'].str.replace(',', ' - ')
summary_filtered_map['log_population'] = (summary_filtered_map['population']) ** 0.5
summary_filtered_map['population_millions'] = (summary_filtered_map['population'] / 1_000_000).round(2).astype(str) + 'M'

##########
col1 , col2  = st.columns([1.8, 1.1])

with col1:
    st.subheader( f'US Market Flex Properties classified by {filtro_columnas_mapa_mostrar} (bar height) and population (area)')
    # Definimos la capa ColumnLayer
    ### reset view for map 
    st.button('Reset View')
    irr_layer = pdk.Layer(
        "ColumnLayer",
        data=summary_filtered_map,
        get_position=["longitude", "latitude"],
        get_elevation="bar_height",
        elevation_scale=2500,  # Ajusta según sea necesario para la visibilidad
        radius=20000,  # Ajusta el radio de las columnas
        get_fill_color="color",  # Asignar color basado en la columna calculada
        pickable=True, # Permite seleccionar las barras
        extruded=True,
        auto_highlight=True, 
    )

    population_layer = pdk.Layer(
        "ScatterplotLayer",
        data=summary_filtered_map,
        get_position=["longitude", "latitude"],
        get_radius="log_population",  # Radio proporcional a la población
        radius_scale=90,  # Ajustar el factor de escala según sea necesario
        get_fill_color= ## Verde con transparencia
        [55, 8, 94, 80],
        pickable=True
    )

    # Configura la vista inicial del mapa
    view_state = pdk.ViewState(
        longitude=-99,
        latitude=38.83,
        zoom=3.4,
        min_zoom=2,
        max_zoom=7,
        pitch=75,  # Reducido para hacer más distinguibles las barras altas
        bearing=23
    )

    lights = pdk.LightSettings(
        number_of_lights= 3)


    # Renderiza el mapa
    st.pydeck_chart(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/outdoors-v11",

            initial_view_state=view_state,
            layers=[irr_layer, population_layer],
            tooltip={
                "html": """
                    <b>State:</b> {state}<br/>
                    <b>City:</b> {city}<br/>
                    <b>Value:</b> {value_show}<br/>
                    <b>Population:</b> {population_millions}
                    """,
                "style": {
                    "backgroundColor": "steelblue",
                    "color": "white",
                    "fontSize": "14px",
                    "padding": "10px",
                    "borderRadius": "15px"
                }
            }
            
        ),
        use_container_width=True
        )
with col2:
    subcol1, subcol2 = st.columns([0.2, 1])
    with subcol2:
        st.subheader('Distribution')
        fig = px.histogram(
            summary_filtered_map, 
            x='value', 
            color='color_no_list', 
            color_discrete_map='identity',
            pattern_shape='city',
            orientation='v', 
            nbins=100,  
            ###3 avoid showing number of observations in the hover
            template='plotly_dark',
        )

        # Ajustar la apariencia del gráfico
        fig.update_layout(
            showlegend=False, 
            yaxis_visible=False, 
            xaxis_title=f'{filtro_columnas_mapa_mostrar}', 
            xaxis_tickformat=',.2f' if unidad_columna == 'USD' else '.2%' if unidad_columna == '%' else '.2f',
            yaxis_title=None,
            bargap=0.1  # Controlar el espacio entre las barras del histograma
        )

        # Mostrar el gráfico en Streamlit
        st.plotly_chart(fig, use_container_width=True)
        # Notas del mapa
        st.markdown('''**Map Notes:** The height of the bars is proportional to the value you're filtering, and the color is based on percentiles.  
                The size of the circles is proportional to the population of the city.''')

st.subheader('US Markets Summary', divider= 'blue')
st.write('The table below shows the summary of the selected markets,  you can sort them pressing the column name')
data_show = summary_filtered_map[['market','current_price', 'loan', 'equity', 'market_cagr',
'noi_cap_rate_compounded', 'fixed_interest_rate', 'operation_cashflow',
'market_cap_appreciation_bp', 'irr', 'npv',
'stabilized_ncf', 'dcr', 'mortgage_constant', 'max_loan_dcr_test',
'min_debt_yield', 'max_loan_debt_yield_test',
'headline_loan_request', 'valuation', 'cap_rate',
'equity_multiple', 'npv/equity'
]]

if st.session_state.filtro_columnas_mapa in data_show.columns:
     data_show = data_show.sort_values( st.session_state.filtro_columnas_mapa, ascending=False)
else:
    pass

data_show.columns = ['Market', 'Current Price', 'Loan', 'Equity', 'Market CAGR',
                    'NOI Cap Rate Compounded', 'Fixed Interest Rate', 'Operation Cashflow',
                    'Market Cap Appreciation BP', 'IRR', 'NPV',
                    'Stabilized NCF', 'DCR', 'Mortgage Constant', 'Max Loan DCR Test',
                    'Min Debt Yield', 'Max Loan Debt Yield Test',
                    'Headline Loan Request', 'Valuation', 'Cap Rate',
                    'Equity Multiple', 'NPV/Equity']

formats = {
    'Current Price': '${:,.0f}',
    'Loan': '${:,.0f}',
    'Equity': '${:,.0f}',
    'Market CAGR': '{:.2%}',
    'NOI Cap Rate Compounded': '{:.2%}',
    'Fixed Interest Rate': '{:.2%}',
    'Operation Cashflow': '{:,.2%}',
    'Market Cap Appreciation BP': '{:,.0f} bp',
    'IRR': '{:.2%}',
    'NPV': '${:,.0f}',
    'Stabilized NCF': '${:,.0f}',
    'DCR': '{:.2f}',
    'Mortgage Constant': '{:.2%}',
    'Max Loan DCR Test': '{:.2f}',
    'Min Debt Yield': '{:.2%}',
    'Max Loan Debt Yield Test': '{:.2f}',
    'Headline Loan Request': '${:,.0f}',
    'Valuation': '${:,.0f}',
    'Cap Rate': '{:.2%}',
    'Equity Multiple': '{:.2f}x',
    'NPV/Equity': '{:.2f}x'
}
# Aplicar los formatos a las columnas correspondientes  
for col, fmt in formats.items():
    if col in data_show.columns:
        data_show[col] = data_show[col].apply(lambda x: fmt.format(x) if pd.notnull(x) else '')

event =  st.dataframe(data_show.set_index('Market'), 
                      selection_mode=['single-row'], 
                      height=290, use_container_width=True)

###3 filtros por estado y mercado singulares para examinar por separado
st.header('Individual Market Analysis 🏠' + selected_state + '  - ' + selected_city
          , divider= 'blue')
st.write('''Tables below show individual market (and slice) analysis, starting from latest IRR from 10 Yrs, 
         5 Yrs and expected IRR for the next 5 Yrs. Then the current cashflow analysis,  and later other perfromance metrics for the selected specific horizon''')


summary_filtered_ts = summary_filtered[
    (summary_filtered['state'] == selected_state) &
    (summary_filtered['city'] == selected_city)
].copy().rename(columns={
    'date_calculated': 'date',
}).set_index('date')
summary_filtered_ts.index = pd.to_datetime(summary_filtered_ts.index)
summary_filtered_ts = summary_filtered_ts.sort_index()

years_horizon = int(horizon.split(' ')[0])
st.write(f'**Horizon:** {years_horizon*4} Q')
last_date_irr = summary_filtered_ts.index[-years_horizon*4]
last_date_ashok_5 = summary_filtered_ts.index[-5*4]
last_date_ashok_10 = summary_filtered_ts.index[-10*4]
last_date_prices = summary_filtered_ts.index[-1]

table_filter_ts = table[
    (table['state'] == selected_state) &
    (table['city'] == selected_city) &
    (table['horizon'] == horizon) &
    (table['noi'] > 0)  
].groupby('date')[['noi']].first()
table_filter_ts.index = pd.to_datetime(table_filter_ts.index)
table_filter_ts = table_filter_ts.sort_index()


costar_filter = costar[
    (costar['state'] == selected_state) &
    (costar['city'] == selected_city)
].copy()
costar_filter.index = pd.to_datetime(costar_filter.index)
costar_filter = costar_filter.sort_index()
costar_filter['net_interest_rate'] = summary_filtered_ts['net_interest_rate']
costar_filter['irr'] = summary_filtered_ts['irr']
costar_filter['noi'] = table_filter_ts['noi']
costar_filter['noi_percent'] = costar_filter['noi'] / costar_filter['price_sf']
costar_filter.loc[last_date_prices:, 'type_numbers'] = 'forecast'
costar_filter.loc[:last_date_prices, 'type_numbers'] = 'current'
costar_filter.loc[last_date_irr:, 'type_irr'] = 'forecast'
costar_filter.loc[:last_date_irr, 'type_irr'] = 'current'


prices_current = costar_filter[costar_filter['type_numbers'] == 'current'].copy().drop(columns=['type_numbers', 'type_irr'])
prices_forecast = costar_filter[costar_filter['type_numbers'] == 'forecast'].copy().drop(columns=['type_numbers', 'type_irr'])

irr_current = costar_filter[costar_filter['type_irr'] == 'current'].copy().drop(columns=['type_numbers', 'type_irr'])
irr_forecast = costar_filter[costar_filter['type_irr'] == 'forecast'].copy().drop(columns=['type_numbers', 'type_irr'])

fig = go.Figure()
fig.update_layout(template='simple_white', title='Market Metrics: ' + selected_state + ' - ' + selected_city, 
                width = 1000, height=700,
                xaxis = dict(title='Date', showgrid=False, zeroline=False, tickformat='%Y-%m', range=[costar_filter.index[0], '2030-01-01'],
                            tickfont=dict(size=15, color='black'), domain=[0, 0.9]),
                yaxis = dict(title='Price (USD)', tickformat='$,.0f', range=[costar_filter['price_sf'].min()*0.05 , costar_filter.loc[:'2030-01-01']['price_sf'].max() * 1.01],
                                showgrid=False, zeroline=False, tickfont=dict(size=15, color='darkblue')),
                yaxis2=dict(title='Rate (%)', overlaying='y', side='right',  showgrid=False, zeroline=True, range=[-0.15, 0.55],
                            tickfont=dict(size=15, color='darkgreen'), position=0.95, visible=False),
                yaxis3=dict(title='IRR (%)', overlaying='y', side='right', showgrid=False, zeroline=True,  visible=False,
                            range=[0, 2.5]),
                showlegend = False,  
                hovermode='x'
                )
#Plot Price
fig.add_trace(go.Scatter(x=prices_current.index, y=prices_current['price_sf'], mode='lines', name='Price / SF (Current)', 
                         line=dict(color='darkblue', width=1.5), hovertemplate='$%{y:,.0f}'))
fig.add_trace(go.Scatter(x=prices_forecast.index, y=prices_forecast['price_sf'], mode='lines', name='Price / SF (Forecast)',
                         line=dict(color='blue', width=1, dash='dot'), hovertemplate='$%{y:,.0f}'))


#Plot Interest Rate
fig.add_trace(go.Scatter(x=costar_filter.index, y=costar_filter['net_interest_rate'], mode='lines', name='Interest Rate + Spread',
                            line=dict(color='darkred', width=1), yaxis='y2', hovertemplate='%{y:.2%}', fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.2)'))
# Plot NOI Cap Rate
fig.add_trace(go.Scatter(x=prices_current.index, y=prices_current['noi_percent'], mode='lines', name='NOI / Price (Current)',
                         line=dict(color='darkgreen', width=1), yaxis='y2', hovertemplate='%{y:.2%}', fill='tonexty', fillcolor='rgba(0, 255, 0, 0.2)'))
#Plot NOI Cap Rate Forecast
fig.add_trace(go.Scatter(x=prices_forecast.index, y=prices_forecast['noi_percent'], mode='lines', name='NOI / Price (Forecast)',
                         line=dict(color='green', width=1, dash='dot'), yaxis='y2', hovertemplate='%{y:.2%}', fill='tozeroy', fillcolor='rgba(0, 255, 0, 0.1)'))
#Plot IRR in Bars
fig.add_trace(go.Bar(x=irr_current.index, y=irr_current['irr'], name='IRR (Current)',
                     marker=dict(color='purple', opacity=0.7),
                     yaxis='y3', hovertemplate='%{y:.2%}'))
fig.add_trace(go.Bar(x=irr_forecast.index, y=irr_forecast['irr'], name='IRR (Forecast)',
                     marker=dict(color='violet', opacity=0.7),
                     yaxis='y3', hovertemplate='%{y:.2%}'))
fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
st.plotly_chart(fig, use_container_width=True, theme=None)

#################################################################################################################################
st.subheader('Cashflow Analysis 🏦' + selected_state + '  - ' + selected_city + ' - ' + horizon + ' Since ' + date_filtering
             , divider= 'blue')
st.write('The table below shows the cashflow analysis for the selected market, city, horizon, and date. ')

cashflow_filtered = cashflow[
    (cashflow['state'] == selected_state) &
    (cashflow['city'] == selected_city) &
    (cashflow['horizon'] == horizon) & 
    (cashflow['date_calculated'] == date_filtering)

].copy()
cashflow_filtered = cashflow_filtered[['equity', 'noi', 'debt_payment','loan_payoff', 'valuation', 'cashflow']]
cashflow_filtered.index = pd.to_datetime(cashflow_filtered.index).strftime('%Y-%m')

cf_formats = {
    'equity': '${:,.2f}',
    'noi': '${:,.2f}',
    'debt_payment': '${:,.2f}',
    'loan_payoff': '${:,.2f}',
    'valuation': '${:,.2f}',
    'cashflow': '${:,.2f}'
}
# Aplicar los formatos a las columnas correspondientes
for col, fmt in cf_formats.items():
    if col in cashflow_filtered.columns:
        cashflow_filtered[col] = cashflow_filtered[col].apply(lambda x: fmt.format(x))

st.dataframe(cashflow_filtered, use_container_width=True)

#################################################################################################################################

st.subheader('IRR Analysis: 10 Yrs, 5 Yrs and Expected IRR for the next 5 Yrs', 
                divider= 'blue')


s_f = summary.loc[
    (summary['state'] == selected_state) &
    (summary['city'] == selected_city)].copy()
s_f['date_calculated'] = pd.to_datetime(s_f['date_calculated'])
s_f['date_end'] = pd.to_datetime(s_f['date_end'])

s_f = s_f.loc[
    ((s_f['horizon'] == '10 Yrs') & (s_f['date_end'] == last_date_prices)) |
    ((s_f['horizon'] == '5 Yrs') & (s_f['date_calculated'] == last_date_ashok_5)) |
    ((s_f['horizon'] == '5 Yrs') & (s_f['date_calculated'] == last_date_prices))
    ].copy()


s_f['ltv'] = 0.7

s_f = s_f[['current_price','ltv', 'loan', 'equity', 'interest_rate', 'spread',
           'net_interest_rate', 'horizon', 'irr', 'equity_multiple', 'npv/equity', 'market_cagr',
              'noi_cap_rate_compounded', 'operation_cashflow', 'market_cap_appreciation_bp',
              'npv', 'stabilized_ncf', 'dcr', 'mortgage_constant', 'max_loan_dcr_test',
              'min_debt_yield', 'max_loan_debt_yield_test', 'headline_loan_request',
              'valuation', 'cap_rate', ]]
formats_sf = {
    'current_price': '${:,.0f}',
    'ltv': '{:.2%}',
    'loan': '${:,.0f}',
    'equity': '${:,.0f}',
    'interest_rate': '{:.2%}',
    'spread': '{:.2%}',
    'net_interest_rate': '{:.2%}',
    'irr': '{:.2%}',
    'equity_multiple': '{:.2f}x',
    'market_cagr': '{:.2%}',
    'noi_cap_rate_compounded': '{:.2%}',
    'operation_cashflow': '{:,.2%}',
    'market_cap_appreciation_bp': '{:,.0f} bp',
    'npv': '${:,.0f}',
    'stabilized_ncf': '${:,.0f}',
    'dcr': '{:.2f}',
    'mortgage_constant': '{:.2%}',
    'max_loan_dcr_test': '{:.2f}',
    'min_debt_yield': '{:.2%}',
    'max_loan_debt_yield_test': '{:.2f}',
    'headline_loan_request': '${:,.0f}',
    'valuation': '${:,.0f}',
    'cap_rate': '{:.2%}',
    'equity_multiple': '{:.2f}x',
    'npv/equity': '{:.2f}x'
}

# Aplicar los formatos a las columnas correspondientes
for col, fmt in formats_sf.items():
    if col in s_f.columns:
        s_f[col] = s_f[col].apply(lambda x: fmt.format(x))

s_f.columns = ['Current Price', 'LTV', 'Loan', 'Equity', 'Interest Rate',
              'Spread', 'Net Interest Rate', 'Horizon', 'IRR', 'Equity Multiple',
              'NPV/Equity', 'Market CAGR', 'NOI Cap Rate Compounded',
              'Operation Cashflow', 'Market Cap Appreciation BP', 'NPV',
              'Stabilized NCF', 'DCR', 'Mortgage Constant', 'Max Loan DCR Test',
              'Min Debt Yield', 'Max Loan Debt Yield Test',
              'Headline Loan Request', 'Valuation', 'Cap Rate']
s_f = s_f.T
s_f.columns = ['10 Yrs (2015-2025)', '5 Yrs (2025-2025)', '5 Yrs (Forecast)']

st.dataframe(s_f, use_container_width=True,height=920,)

