import pandas as pd

df = pd.read_csv('data/car.data')

for index, row in df.iterrows():
  if row.buying == 'vhigh':
    row.buying = 1
  if row.buying == 'high':
    row.buying = 0.67
  if row.buying == 'med':
    row.buying = 0.33
  if row.buying == 'low':
    row.buying = 0

  if row.maint == 'vhigh':
    row.maint = 1
  if row.maint == 'high':
    row.maint = 0.67
  if row.maint == 'med':
    row.maint = 0.33
  if row.maint == 'low':
    row.maint = 0

  if row.doors == '5more':
    row.doors = 5

  if row.persons == 'more':
    row.persons = 6
  
  if row.lug_boot == 'small':
    row.lug_boot = 0
  if row.lug_boot == 'med':
    row.lug_boot = 0.5
  if row.lug_boot == 'big':
    row.lug_boot = 1

  if row.safety == 'low':
    row.safety = 0
  if row.safety == 'med':
    row.safety = 0.5
  if row.safety == 'high':
    row.safety = 1

  if row['class'] == 'unacc':
    row['class'] = 0
  if row['class'] == 'acc':
    row['class'] = 1
  if row['class'] == 'good':
    row['class'] = 2
  if row['class'] == 'vgood':
    row['class'] = 3

df = df[['class', 'buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']]

print(df)

df.to_csv('data/carNumeric.data', index=False)