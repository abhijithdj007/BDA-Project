# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Overview
# MAGIC 
# MAGIC This notebook will show you how to create and query a table or DataFrame that you uploaded to DBFS. [DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html) is a Databricks File System that allows you to store data for querying inside of Databricks. This notebook assumes that you have a file already inside of DBFS that you would like to read from.
# MAGIC 
# MAGIC This notebook is written in **Python** so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` syntax. Python, Scala, SQL, and R are all supported.

# COMMAND ----------

# File location and type
fi_file_location = "/FileStore/tables/Food_Inspection_data.csv"
crime_file_location = "/FileStore/tables/Crimes_Data.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
fi_df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(fi_file_location)

display(fi_df)

# COMMAND ----------

# The applied options are for CSV files. For other file types, these will be ignored.
crime_df = spark.read.format(file_type) \
  .option("inferSchema", 'true') \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(crime_file_location)

display(crime_df)
# crime_df.printSchema()

# COMMAND ----------

# Create a view or table

temp_table_name = "Food_Inspections_data_csv"

fi_df.createOrReplaceTempView(temp_table_name)

# Create a view or table

temp_table_name = "Crimes_Data_csv"

crime_df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC /* Query the created temp table in a SQL cell */
# MAGIC 
# MAGIC select * from `Food_Inspections_data_csv`

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC /* Query the created temp table in a SQL cell */
# MAGIC 
# MAGIC select * from `Crimes_Data_csv`

# COMMAND ----------

# With this registered as a temp view, it will only be available to this particular notebook. If you'd like other users to be able to query this table, you can also create a table from the DataFrame.
# Once saved, this table will persist across cluster restarts as well as allow various users across different notebooks to query this data.
# To do so, choose your table name and uncomment the bottom line.

permanent_fi_table_name = "Food_Inspections_data_csv"
permanent_crime_table_name = "Crimes_Data_csv"


# df.write.format("parquet").saveAsTable(permanent_table_name)

# COMMAND ----------

import pandas as pd

final_frame = pd.DataFrame()

pd_crime_df = spark.sql("select * from `Crimes_Data_csv`").toPandas()

pd_crime_df.columns = [c.replace(' ', '_') for c in pd_crime_df.columns]
pd_crime_df = pd_crime_df[['Block','Primary_Type','Arrest','Year','Latitude','Longitude']]
pd_crime_df['Block'] = pd_crime_df["Block"].apply(lambda x: x.lower())
pd_crime_df = pd.DataFrame({'ArrestCount': pd_crime_df.groupby(
  ['Block','Primary_Type','Year','Arrest','Latitude','Longitude']).size()}).reset_index()
#df.loc[df['Arrest'] == False, 'ArrestCount'] = 0

# COMMAND ----------

import sys

def progress(progress, total, status=''):
    length = 40  # modify this to change the length
    block = int((progress/total)*40)
    percent = (block/length)*100
    msg = "\r[{0}] {1}%({2}/{3}) - {4}".format("#" * block + "-" * (length - block), round(percent, 2), progress, total, status)
    if progress >= total: msg += " DONE\r\n"
    sys.stdout.write(msg)
    sys.stdout.flush()
    
def refine_type(x):
    out = "Restaurant"
    if "grocery" in x.lower():
        out = "Grocery"
    elif "school" in x.lower():
        out = "School"
    elif "restaurant" in x.lower():
        out = "Restaurant"
    return out

pd_fi_df = spark.sql("select * from `Food_Inspections_data_csv`").toPandas()

rest_names = pd_fi_df['name'].unique()
rest_names = rest_names.tolist()

prog=0

for restaurant in rest_names:
    prog = prog+1

    temp_rest_data = pd_fi_df[pd_fi_df['name'] == restaurant]
    check_license = temp_rest_data[temp_rest_data['licence_description'].str.contains("Liquor")].size > 0
    check_license1 = temp_rest_data[temp_rest_data['licence_description'].str.contains("Tobacco")].size > 0
    temp_rest_data = temp_rest_data.iloc[0]
    temp_rest_data['has_liqour_license'] = check_license
    temp_rest_data['has_Tobacco_license'] = check_license1
    temp_crime_frame = pd_crime_df[pd_crime_df['Block'] == temp_rest_data['Block']]

    if temp_crime_frame.size > 0:
        lat = temp_crime_frame.iloc[0]['Latitude']
        long = temp_crime_frame.iloc[0]['Longitude']
        maxLat = lat + 0.003
        minLat = lat - 0.003
        maxLong = long + 0.003
        minLong = long - 0.003
        blocks_3_crimes = pd_crime_df[(pd_crime_df['Latitude'] >= minLat) & (pd_crime_df['Latitude'] <= maxLat) 
                                      & (pd_crime_df['Longitude'] <= minLong) & (pd_crime_df['Longitude'] <= maxLong)]
        if blocks_3_crimes.size > 0:

            blocks_3_crimes1 = blocks_3_crimes.groupby(['Primary_Type', 'Year'], as_index=False)[
                ["ArrestCount"]].sum()
            blocks_3_crimes1 = blocks_3_crimes1.rename(columns={'ArrestCount':'total'})
            blocks_3_crimes.loc[blocks_3_crimes['Arrest'] == False, 'ArrestCount'] = 0
            blocks_3_crimes = blocks_3_crimes.groupby(['Primary_Type', 'Year'], as_index=False)[
                ["ArrestCount"]].sum()
            blocks_3_crimes = pd.merge(blocks_3_crimes, blocks_3_crimes1, on=['Primary_Type', 'Year'])
            on_prem_crimes = pd_crime_df[pd_crime_df['Block'] == temp_rest_data['Block']]

            if on_prem_crimes.size > 0:

                on_prem_crimes = pd.DataFrame(
                    {'onPrem': on_prem_crimes.groupby(["Block", "Primary_Type", "Year"]).size()}).reset_index()
                merged2 = pd.merge(blocks_3_crimes, on_prem_crimes, on=['Primary_Type', 'Year'])
                rest_main_df = pd.DataFrame(temp_rest_data).transpose()
                final_loop_merge = pd.merge(rest_main_df, merged2, on=['Block'])
                final_frame = final_frame.append(final_loop_merge)
    progress(prog, len(rest_names) - 1, status='Generating crime report')

# COMMAND ----------

final_frame.dtypes

# COMMAND ----------

final_frame = final_frame[['Year','categories','name','address','has_Tobacco_license','has_liqour_license','Primary_Type','total','ArrestCount','onPrem']]
final_frame = final_frame.sort_values(["Year", 'total'], ascending=[True, False])

final_frame = final_frame.rename(columns={'categories':'Business Type','name':'Business Name','has_Tobacco_license':'Has Tobacco License','has_liqour_license':'Has Liquor License','Primary_Type':'Crime Type','total':'#Crimes','ArrestCount':'#Arrests','onPrem':'#On Premises'})


final_frame['Business Type'] = final_frame['Business Type'].apply(lambda x: refine_type(x))


# COMMAND ----------

display(final_frame)
