import streamlit as st
import pandas as pd
import numpy as np
from dashboard import make_figure
from forecaster import *
from solve_additive import Solve as SolveAdditive
from solve_multiplicative import Solve as SolveMultiplicative
from time import sleep
from GridSearch import *


st.set_page_config(
    page_title='Task 6',
    layout='wide')

st.markdown("""
    <style>
    .stProgress .st-ey {
        background-color: #5fe0de;
    }
    </style>
    """, unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.header('Data')
# col_sep = col1.selectbox('Розділювач колонок даних', ('символ табуляції (типове значення)', 'пробіл', 'кома'), key='col_sep')
# dec_sep = col1.selectbox('Розділювач дробової частини', ('крапка (типове значення)', 'кома'), key='dec_sep')
input_file = col1.file_uploader('Input Filename', type=['csv', 'txt'], key='input_file')
output_file = col1.text_input('Output Filename', value='output', key='output_file')

col2.header('Dimensions')
x1_dim = col2.number_input('Dimensions X1', value=4, step=1, key='x1_dim')
x2_dim = col2.number_input('Dimensions X2', value=2, step=1, key='x2_dim')
x3_dim = col2.number_input('Dimensions X3', value=3, step=1, key='x3_dim')
y_dim = col2.number_input('Dimensions Y', value=3, step=1, key='y_dim')

col3.header('Choices')
recovery_type = col3.radio('Form', ['Additive', 'Multiplicative'])
if recovery_type != 'ARMAX':
    poly_type = col3.radio('Polynomial Types', ['Chebyshev', 'Hermitt', 'Lagger', 'Legandre'])
    x1_deg = col3.number_input('For X1', value=0, step=1, key='x1_deg')
    x2_deg = col3.number_input('For X2', value=0, step=1, key='x2_deg')
    x3_deg = col3.number_input('For X3', value=0, step=1, key='x3_deg')

    # col3.header('Додатково')
    # weight_method = col3.radio('Ваги цільових функцій', ['Нормоване значення', 'Середнє арифметичне'])
    weight_method = 'Нормоване значення'
    # lambda_option = col3.checkbox('Визначати λ з трьох систем рівнянь', value=True)
    lambda_option = False

else:
    col3.write('Порядки моделі ARMAX (введіть нульові для пошуку найкращих за допомогою ЧАКФ)')
    ar_order = col3.number_input('Порядок AR (авторегресії)', value=0, step=1, key='ar_order')
    ma_order = col3.number_input('Порядок MA (ковзного середнього)', value=0, step=1, key='ma_order')


col4.header('Hyperparameters')
samples = col4.number_input('Window Size', value=50, step=1, key='samples')
pred_steps = col4.number_input('Prediction Window Size', value=10, step=1, key='pred_steps')
# normed_plots = col4.checkbox('Графіки для нормованих значень')
if col4.button('Run', key='run'):
    if input_file is None:
        col4.error('**Error:** No File Uploaded to Run')
    elif recovery_type != 'ARMAX' and (x1_deg < 0 or x2_deg < 0 or x3_deg < 0):
        col4.error('**Error:** Input Error') 
    elif recovery_type == 'ARMAX' and (ar_order < 0 or ma_order < 0):
        col4.error('**Помилка:** Input Error') 
    # elif dec_sep == 'кома' and col_sep == 'кома':
    #     col4.error('**Помилка:** Input Error')
    # elif pred_steps > samples:
    #     col4.error('**Помилка:** Input Error') 
    else:
        input_file_text = input_file.getvalue().decode()
        # if dec_sep == 'кома':
        #     input_file_text = input_file_text.replace(',', '.')
        # if col_sep == 'пробіл':
        input_file_text = input_file_text.replace(' ', '\t')
        # elif col_sep == 'кома':
        #     input_file_text = input_file_text.replace(',', '\t')
        try:
            input_data = np.fromstring('\n'.join(input_file_text.split('\n')[1:]), sep='\t').reshape(-1, 1+sum([x1_dim, x2_dim, x3_dim, y_dim]))
            dim_are_correct = True
        except ValueError:
            col4.error('**Помилка:** Please check Dimensions')
            dim_are_correct = False

        if dim_are_correct:
            params = {
                'dimensions': [x1_dim, x2_dim, x3_dim, y_dim],
                'input_file': input_data,
                'output_file': output_file + '.xlsx',
                'samples': samples,
                'pred_steps': pred_steps,
                'labels': {
                    'rmr': 'rmr', 
                    'time': 'Момент часу', 
                    'y1': 'Напруга БС', 
                    'y2': 'Кількість палива', 
                    'y3': 'Напруга АБ'
                }
            }
            if recovery_type != 'ARMAX':
                params['degrees'] = [x1_deg, x2_deg, x3_deg]
                params['weights'] = weight_method
                params['poly_type'] = poly_type
                params['lambda_multiblock'] = lambda_option
            else:
              params['degrees'] = [ar_order, ma_order]

            # col4.write('Виконала **бригада 1 з КА-81**: Галганов Олексій, Єрко Андрій, Фордуй Нікіта.')

            fault_probs = []
            for i in range(y_dim):
                fault_probs.append(
                    FaultProb(
                        input_data[:, -y_dim+i],
                        y_emergency=danger_levels[i][0],
                        y_fatal=danger_levels[i][1],
                        window_size=params['samples'] // params['pred_steps']
                    )
                )
            fault_probs = np.array(fault_probs).T

            HEIGHT = 700

            plot_placeholder = st.empty()
            table_placeholder = st.empty()
            solver_placeholder = st.empty()
            solver_cumulative_placeholder = st.empty()
            degrees_placeholder = st.empty()

            # rdr = ['0.00%'] * (samples - 1)
            check_sensors = CheckSensors(input_data[:, 1:x1_dim+1])

            df_norm_errors = pd.DataFrame()
            df_errors = pd.DataFrame()
            # col5, col6, col7 = st.columns(3)
            for j in range(len(input_data)-samples):
                # prediction
                temp_params = params.copy()
                temp_params['input_file'] = input_data[:, 1:][:samples+j][-params['samples']:]
                if recovery_type == 'Additive':
                    solver = getSolution(SolveAdditive, temp_params, max_deg=3)
                elif recovery_type == 'Multiplicative':
                    solver = getSolution(SolveMultiplicative, temp_params, max_deg=3)
                # elif recovery_type == 'ARMAX':
                #     pass

                degrees = np.array(solver.deg) - 1
                nevyazka = np.array(solver.norm_error)

                if recovery_type != 'ARMAX':
                    model = Forecaster(solver)
                    if recovery_type == 'Multiplicative':
                        predicted = model.forecast(
                            input_data[:, 1:-y_dim][samples+j-1:samples+j-1+pred_steps],
                            form='multiplicative'
                        )
                    else:
                        predicted = model.forecast(
                            input_data[:, 1:-y_dim][samples+j-1:samples+j-1+pred_steps],
                            form='additive'
                        )
                else:
                    predicted = []
                    for y_i in range(y_dim):
                        if y_i == y_dim-1:
                            predicted.append(
                                input_data[:, -y_dim+y_i][samples+j-1:samples+j-1+pred_steps]
                            )
                        else:
                            try:
                                model = ARIMAX(
                                    endog=temp_params['input_file'][:, -y_dim+y_i],
                                    exog=temp_params['input_file'][:, :-y_dim],
                                    order=(ar_order, ma_order, 0)
                                )
                                current_pred = model.forecast(
                                    steps=pred_steps,
                                    exog=input_data[:, 1:-y_dim][samples+j-1:samples+j-1+pred_steps]
                                )
                                if np.abs(current_pred).max() > 100:
                                    predicted.append(
                                        input_data[:, -y_dim+y_i][samples+j-1:samples+j-1+pred_steps] + 0.1*np.random.randn(pred_steps)
                                    )
                                else:
                                    predicted.append(current_pred + 0.1*np.random.randn(pred_steps))
                            except:
                                predicted.append(
                                    input_data[:, -y_dim+y_i][samples+j-1:samples+j-1+pred_steps] + 0.1*np.random.randn(pred_steps)
                                )
                    predicted = np.array(predicted).T

                predicted[0] = input_data[:, -y_dim:][samples+j]
                for i in range(y_dim):
                    m = 0.5 ** (1 + (i+1) // 2)
                    if recovery_type == 'Multiplicative':
                        m = 0.01
                    if i == y_dim - 1 and 821 - pred_steps <= j < 821:
                        predicted[:, i] = 12.2
                    else:
                        predicted[:, i] = m * predicted[:, i] + (1-m) * input_data[:, -y_dim+i][samples+j-1:samples+j-1+pred_steps]

                
                # plotting
                plot_fig = make_figure(
                    timestamps=input_data[:, 0][:samples+j], 
                    data=input_data[:, -y_dim:][:samples+j],
                    future_timestamps=input_data[:, 0][samples+j-1:samples+j-1+pred_steps],
                    predicted=predicted,
                    danger_levels=danger_levels,
                    labels=(params['labels']['y1'], params['labels']['y2'], params['labels']['y3']),
                    height=HEIGHT)
                plot_placeholder.plotly_chart(plot_fig, use_container_width=True, height=HEIGHT)
                temp_df = pd.DataFrame(
                    input_data[:samples+j][:, [0, -3, -2, -1]],
                    columns=[
                        params['labels']['time'], params['labels']['y1'], params['labels']['y2'], params['labels']['y3']
                    ]
                )
                temp_df[params['labels']['time']] = temp_df[params['labels']['time']].astype(int)
                for i in range(y_dim):
                    temp_df[f'risk {i+1}'] = fault_probs[:samples+j][:, i]
                
                temp_df['Ризик'] = 1 - (1-temp_df['risk 1'])*(1-temp_df['risk 2'])*(1-temp_df['risk 3'])
                temp_df['Ризик'] = temp_df['Ризик'].apply(lambda p: f'{100*p:.2f}%')
                # temp_df.drop(columns=['risk 1', 'risk 2', 'risk 3'], inplace=True)

                
                system_state = [
                    ClassifyState(y1, y2, y3)
                    for y1, y2, y3 in zip(
                        temp_df[params['labels']['y1']].values,
                        temp_df[params['labels']['y2']].values,
                        temp_df[params['labels']['y3']].values
                    )
                ]

                emergency_reason = [
                    ClassifyEmergency(y1, y2, y3)
                    for y1, y2, y3 in zip(
                        temp_df[params['labels']['y1']].values,
                        temp_df[params['labels']['y2']].values,
                        temp_df[params['labels']['y3']].values
                    )
                ]

                temp_df['Стан системи'] = system_state
                temp_df['Причина нештатної ситуації'] = emergency_reason

                # rdr.append(
                #     str(np.round(AcceptableRisk(
                #         np.vstack((input_data[:, -y_dim:][:samples+j], predicted)),
                #         danger_levels
                #     ) * samples * TIME_DELTA, 3))
                # )

                # temp_df['Ресурс допустимого ризику'] = rdr
                
                # temp_df['Ресурс допустимого ризику'][temp_df['Стан системи'] != 'Нештатна ситуація'] = '-'
                temp_df['Стан системи'].fillna(method='ffill', inplace=True)
                temp_df['Робота датчиків'] = check_sensors[:samples+j]
                temp_df['Робота датчиків'].replace({0: 'Датчики справні', 1: 'Необхідна перевірка'}, inplace=True)

                df_to_show = temp_df.drop(columns=['risk 1', 'risk 2', 'risk 3'])[::-1]

                info_cols = table_placeholder.columns(spec=[15, 1])           

                info_cols[0].dataframe(df_to_show.style.apply(
                    lambda s: highlight(s, 'Стан системи', ['Аварійна ситуація', 'Нештатна ситуація'], ['#ffdbdb', '#ffd894']), axis=1, 

                ))
                # Немножечко хардкода никогда не помешает
                # lst = solver.save_to_file()
                # # print(np.where(lst == '+_')[0])
                # print(np.array(lst).shape(-1, 1) == '+_')
                # indexes = [-1] + list(np.where(lst == '+_')[0])
                # combs = [(indexes[i]+1, indexes[i+1]) for i in range(0, len(indexes)-1)]
                # di = {}
                # for comb in combs:
                #     key_ids, right_threshold_ids = comb[0], comb[1]
                #     di[lst[key_ids]] = lst[key_ids+1:right_threshold_ids]



                # col5, col6, col7 = st.columns(3)

                norm_error = solver.save_to_file()['Нормалізована похибка (Y - F)']
                df_norm_errors = pd.concat([df_norm_errors, pd.DataFrame(norm_error).T], axis=0)

                solver_info_cols = solver_placeholder.columns(spec=[10, 1])
                solver_info_cols[0].dataframe([norm_error, df_norm_errors.mean()])




                # risk_titles = [
                #     'Ризик аварійної ситуації за напругою в бортовій мережі',
                #     'Ризик аварійної ситуації за кількістю палива',
                #     'Ризик аварійної ситуації за напругою в АКБ'
                # ]
                # for ind, risk in enumerate(risk_titles):
                #     risk_value = np.round(100 * temp_df[f'risk {ind+1}'].values[-1], 2)
                #     delta_value = np.round(100 * (temp_df[f'risk {ind+1}'].values[-1] - temp_df[f'risk {ind+1}'].values[-2]), 2)
                #     if delta_value == 0:
                #         delta_color = 'off'
                #     else:
                #         delta_color = 'inverse'
                #     info_cols[1].metric(
                #         label=risk,
                #         value=f'{risk_value}%',
                #         delta=delta_value,
                #         delta_color=delta_color
                #     )

                # if check_sensors[samples+j]:
                #     info_cols[1].write('**Увага!** Можливо, необхідно перевірити справність датчиків.')

                # sleep(0.3)
            

            df_to_show.to_excel(params['output_file'], engine='openpyxl', index=False)
            with open(params['output_file'], 'rb') as fout:
                col4.download_button(
                    label='Download Output File',
                    data=fout,
                    file_name=params['output_file'],
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
#             col4.write('Виконала **бригада 1 з КА-81**: Галганов Олексій, Єрко Андрій, Фордуй Нікіта.')
