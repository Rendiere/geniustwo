from genuistwo.utils import xml_to_dataframe

itunes_file = "Library.xml"

 # Load iTunes file
df = xml_to_dataframe(itunes_file)

df.to_csv('Library.csv', index=False)
