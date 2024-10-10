import numpy as np

def get_age_from_idade(idade):
    age_str = str(idade)

    unit = int(age_str[0])
    value = int(age_str[1:])

    hours_in_day = 24
    hours_in_month = hours_in_day * 30
    hours_in_year = hours_in_day * 365

    if unit == 4:
        return value * hours_in_year
    elif unit == 3:
        return value * hours_in_month
    elif unit == 2:
        return value * hours_in_day
    elif unit == 1:
        return value
    else:
        return np.nan