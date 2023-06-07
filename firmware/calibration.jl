using Dates, DataFrames, CSV, ScikitLearn, PyPlot

filepath = "C:/Users/sethl/OneDrive/Desktop/Calibration-of-LoRa-Nodes-using-Machine-Learning-main/calibrate.csv"
df = DataFrames.DataFrame(CSV.File(filepath))


y_grimm = []
y_loRa = []
y_Palas = []
x = []

#df.Date = Date.(df.dateTime, "dd/mm/yyyy")
col_name = names(df)

append!(x, "dateTime")

for i in col_name
    if "_grimm" in i
        append!(y_grimm, i)
    if "_loRa" in i
        append!(x, i)
    if "Palas" in i
        append!(y_Palas, i)
end
# microgram/meter cubed for pm

