library(xts)
library(pdfetch)
library(mFilter) 
library(tidyverse)
library(fredr)

fredr_set_key("18c2830f79155831d5c485d84472811f")

data_pc <- pdfetch_FRED(c("GDPC1", "UNRATE", "CPIAUCSL", "PPIACO", "CPILFESL"))
unemp <- rename(fredr(series_id = "UNRATE"), "unemp" = value)
gdp <- rename(fredr(series_id = "GDPC1"), "gdp" = value)
cpi <- rename(fredr(series_id = "CPIAUCSL"), "cpi" = value)
ppi <- rename(fredr(series_id = "PPIACO"), "ppi" = value)
cpil <- rename(fredr(series_id = "CPILFESL"), "cpil" = value)

# Convert data to quarterly frequency
data_pc <- to.period(data_pc, period = "quarter", OHLC = FALSE)
#View(data_pc)

#Transformations
data_pc$l_cpi <- log(data_pc$CPIAUCSL)
data_pc$l_cpi_core <- log(data_pc$CPILFESL)
data_pc$l_ppiaco <- log(data_pc$PPIACO)
data_pc$unrate <- (data_pc$UNRATE)

#Series for plots of the Phillips curve
data_pc$inflation <- 100*diff(data_pc$l_cpi, 4)
#plot.xts(data_pc$inflation)

#Quarterly inflation, annualized
data_pc$inflation_q = 100*diff(data_pc$l_cpi)

#Inflation expectations as an average of 4 past y-o-y inflation rates
data_pc$infexp <- 1/4*(lag(data_pc$inflation,1) + lag(data_pc$inflation, 2) + lag(data_pc$inflation, 3) + lag(data_pc$inflation,4))

#Creating inflation gap
data_pc$infgap <- data_pc$inflation_q-data_pc$infexp

#Supply shocks
data_pc$s_shock <- 4*diff(data_pc$l_cpi)*100 - 4*diff(data_pc$l_cpi_core)*100

as.data.frame(data_pc) %>%
  rownames_to_column(var = "date") %>%
  select(date,"unemp" = unrate, infgap,ppi, infexp, "cpi" = inflation_q,s_shock) %>%
  write.csv("C:/Users/HP/Downloads/philips_new.csv") 
