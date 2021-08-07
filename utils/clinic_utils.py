from datetime import datetime


def str_to_dt(date_str, time_str):
    return datetime.strptime(date_str + ' ' + time_str, '%Y-%m-%d %H:%M')


def cal_duration(dt1, dt2, div=60):
    return (dt1-dt2).total_seconds() / div


def extract_handcraft_features(data):
    features = dict()
    features['age'] = float(data['age'])

    arrival_dt = str_to_dt(data['arrival_date'], data['arrival_time'])

    if data['onset'] == 'known_onset':
        clear_onset_dt = str_to_dt(
            data['clear_onset_date'], data['clear_onset_time'])
        features['duration lsn (minute)'] = cal_duration(arrival_dt, clear_onset_dt)
        features['duration fsa (minute)'] = features['duration lsn (minute)']

    else:
        lsn_dt = str_to_dt(data['lsn_date'], data['lsn_time'])
        fsa_dt = str_to_dt(data['fsa_date'], data['fsa_time'])
        features['duration lsn (minute)'] = cal_duration(arrival_dt, lsn_dt)
        features['duration fsa (minute)'] = cal_duration(arrival_dt, fsa_dt)

    features['gradual onset'] = 1 if data['time_course'] == 'gradual' else 0
    features['peak clear onset'] = 1 if data['time_course'] == 'peak_at_onset' else 0

    features['cortical lobe sign'] = 1 if (
        (int(data['cc_aphasia']) == 1) or
        (int(data['nihss_1b']) == 2) or
        (int(data['nihss_2']) == 2) or
        (int(data['nihss_3']) > 0) or
        (int(data['nihss_9']) > 0) or
        (int(data['nihss_11']) > 0)
    ) else 0

    features['metabolic syndrome'] = 1 if (
        (int(data['ud_dm']) == 1) or
        (int(data['ud_ht']) == 1) or
        (int(data['ud_obesity']) == 1) or
        (int(data['ud_dlp']) == 1)
    ) else 0

    features['vascular disease'] = 1 if (
        (int(data['ud_peripheral_ad']) == 1) or
        (int(data['ud_prev_tia']) == 1) or
        (int(data['ud_prev_stroke']) == 1) or
        (int(data['ud_coronary_hd']) == 1)
    ) else 0

    features['valvular heart disease'] = int(data['ud_valvular_hd'])
    features['hx tia (same, in 2 wks)'] = int(data['ud_hx_tia'])
    features['smoking'] = int(data['ud_smoking'])

    return features
