library(xts)
library(pdfetch) #Library for loading FRED data
library(ggplot2) #Library for plotting
library(mFilter) #Library for HP filter

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
data_pc$lgdp <- log(data_pc$GDPC1) # Take logs
hp_gdp <- hpfilter(data_pc$lgdp, freq = 1600, type="lambda")
data_pc$gdpgap <- 100*hp_gdp$cycle
data_pc$l_cpi <- log(data_pc$CPIAUCSL)
data_pc$l_cpi_core <- log(data_pc$CPILFESL)
data_pc$l_ppiaco <- log(data_pc$PPIACO)
data_pc$unrate <- (data_pc$UNRATE)

#Series for plots of the Phillips curve
data_pc$inflation <- 100*diff(data_pc$l_cpi, 4)
#plot.xts(data_pc$inflation)

#Add recession bars
recessions.df = read.table(textConnection(
  "Peak, Trough
  1957-08-01, 1958-04-01
  1960-04-01, 1961-02-01
  1969-12-01, 1970-11-01
  1973-11-01, 1975-03-01
  1980-01-01, 1980-07-01
  1981-07-01, 1982-11-01
  1990-07-01, 1991-03-01
  2001-03-01, 2001-11-01
  2007-12-01, 2009-06-01
  2020-02-01, 2020-05-01"), sep=',',
  colClasses=c('Date', 'Date'), header=TRUE)

ggplot() +
  geom_line(data = data_pc$unrate, aes(x = Index, y = data_pc$unrate, color = "Unemployment"), lwd = 1) +
  geom_line(data = data_pc$inflation, aes(x = Index, y = data_pc$inflation, color = "Inflation"), lwd = 1) +
  geom_rect(data=recessions.df, inherit.aes=F, aes(xmin=Peak, xmax=Trough, ymin=-Inf, ymax=+Inf), fill='darkgray', alpha=0.5) +
  theme_classic() +
  labs(title = "US Unemployment rate and Inflation", x = "Quarter", y = "") +
  labs(color="Legend") +
  theme(legend.position="bottom")

#Quarterly inflation, annualized
data_pc$inflation_q = 4*100*diff(data_pc$l_cpi)

#Inflation expectations as an average of 4 past y-o-y inflation rates
data_pc$infexp <- 1/4*(lag(data_pc$inflation,1) + lag(data_pc$inflation, 2) + lag(data_pc$inflation, 3) + lag(data_pc$inflation,4))

plot.xts(data_pc$inflation, col = "black", lwd = 2)
addSeries(data_pc$infexp, on = 1, col = "red", lwd = 2 )

#Creating inflation gap
data_pc$infgap <- data_pc$inflation_q-data_pc$infexp
plot.xts(data_pc$inflation_q)
addSeries(data_pc$infgap, on = 1, col = "red", lwd = 2 )

#Supply shocks
data_pc$ss1 <- 4*diff(data_pc$l_cpi)*100 - 4*diff(data_pc$l_cpi_core)*100
data_pc$ss2 <- 100*diff(data_pc$l_ppiaco)

data_pc$ss1 <- 4*diff(data_pc$l_cpi)*100 - 4*diff(data_pc$l_cpi_core)*100
data_pc$ss2 <- 100*diff(data_pc$l_ppiaco)

library(rollRegres)
data1 <- na.omit(data_pc)
pc_rolling <- roll_regres(data1$infgap ~ data1$unrate + data1$ss1, width = 40, do_downdates = TRUE)
data1$slope <- pc_rolling$coefs[1:255, 2:2]
plot.xts(data1$slope)

as.data.frame(data_pc) %>%
  rownames_to_column(var = "date") %>%
  #filter(inflation_q > -10) %>%
ggplot() +
  geom_point(aes(x = unrate, y = infgap, color = date > "2000-01-01")) +
  geom_smooth(aes(x = unrate, y = infgap, color = date > "2000-01-01"),
              method = "lm")

as.data.frame(data_pc) %>%
  rownames_to_column(var = "date") %>%
  filter(date > "2000-01-01") %>%
  lm(inflation_q ~ unrate, data = .) %>%
  summary()

as.data.frame(data_pc) %>%
  rownames_to_column(var = "date") %>%
  select(date, unrate, infgap, infexp, inflation_q) %>%
  write.csv("C:/Users/HP/Downloads/philips.csv") 
