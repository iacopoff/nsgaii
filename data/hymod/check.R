library(ggplot2)
library(data.table)

file = "~/research/dev/nsgaii/data/hymod/hymod_input.csv"

df = fread(file)



df[,Date:=as.POSIXct(Date,format = "%d.%m.%Y")]

df = na.omit(df)

colSums(df[,-1])

df.melt = melt(df,id.vars = "Date")


ggplot(df.melt) + 
     geom_line(aes(x=Date,y=value,colour=variable)) #+ 
    # scale_y_log10()
