# Zachary Roman Ph.D. <ZacharyJoseph.Roman@uzh.ch>
# Run file for unsupervised bot classification model in Roman, Brandt, 
# and Miller (2022).

library(mvtnorm)
library(R2jags)
library(lavaan)
library(psych)
library(ggplot2)
library(kableExtra)
source("functions/functions.R")


dat <- readRDS("data/study_3_RWA_SDO_NAT.RDS")

# Survey Items are col 24-48
# Rest are meta-data and demographics
# Isolate survey items
ydat <- dat[,c(24:48)]

syntax <-  "
RWA =~  rwa1 + rwa2 + rwa3 + rwa4 + rwa5 + rwa6 + rwa7 + rwa8 + rwa9 + rwa10
SDO =~  sdo1 + sdo2 + sdo3 + sdo4 + sdo5 + sdo6 + sdo7 + sdo8
NAT =~ nationalism1 + nationalism2 + nationalism3 + nationalism4 +
nationalism5 + nationalism6 + nationalism7"


# Person-fit index function 
# Section 2.2, page 3, Roman, Brandt, and Miller (2022)
# ydat: the matrix of observed variables
# mod: the lavaan syntax specifying the factor structure
# See functions/functions.R for the code
cd <- cdfun(ydat = , mod = syntax)


# Participant level variances
# Section 2.3, page 4, Roman, Brandt, and Miller (2022)
vy <- rep(0,nrow(ydat))
for(i in 1:nrow(ydat)){
  vy[i]  <- var(as.numeric(ydat[i,]))
}

# # # # # # #
# Lavaan Model
# Used for comparison, not included in the paper
# Serves as a sanity check for the Bayesian CFA
# Estimates are nearly identical 
# (i.e., verifies the factor loading priors aren't influential)



syntax <-  "
RWA =~  rwa1 + rwa2 + rwa3 + rwa4 + rwa5 + rwa6 + rwa7 + rwa8 + rwa9 + rwa10
SDO =~  sdo1 + sdo2 + sdo3 + sdo4 + sdo5 + sdo6 + sdo7 + sdo8
NAT =~ nationalism1 + nationalism2 + nationalism3 + nationalism4 +
nationalism5 + nationalism6 + nationalism7"

summary(cfa(model = syntax,
            data = ydat, std.lv = TRUE),std = TRUE)

# # # # # # # # #
# Jags Model 
# Includes hidden-markov classification sub-model
# Priors are hard-coded to match the manuscript
# See models/cfa_mix_jags_static.txt for specific priors translated to Jags
# Formal specifications and justifications can be found on 
# section 2.4, page 4 of Roman, Brandt, and Miller (2022)


inputObj <- list(N=nrow(ydat), # Sample size
              y=ydat, # Observed survey items
              R0=diag(3), # Number of latent factors
              vy=vy, # Person level variances
              cd=cd) # Person-fit index

params <- c("ty1", # Item level intercepts
            "b0", # Classification sub-model parameters
            "psixi", # Latent precision matrix
            "ly", # Factor loadings
            "Cend", # Classifications
            "sigmaxi", # Residual variances of factors
            "sigmay") # residual variances of observed items

# Optionally you could extract factor scores by adding "xi" to the params list

# This mirrors the iteration counts from the paper
# Testing shows that the full 12,000 is not necessary and
# was just used for confidence in the density of posterior tails
# For weaker computational resources cut this down by a factor of 10

fit1 <- jags.parallel(data=inputObj,
                     parameters.to.save=params,
                     n.iter=12000,
                     n.chains=4,
                     n.thin=2,
                     n.burnin=6000,
                     model.file="models/cfa_mix_jags_static.txt")

summ1 <- fit1$BUGSoutput$summary

summ1

# # # # # # # # #
# Traditional Bayesian CFA 
# For comparison
# This model ignores the bots in the data

inputObj2 <- list(N=nrow(ydat), # Sample size
                  y=ydat, # Observed survey items
                  R0=diag(3)) # Number of latent factors

params <- c("ty1", 
            "psixi",
            "ly",
            "sigmaxi",
            "sigmay")

fit2 <- jags.parallel(data=inputObj2,
                      parameters.to.save=params,
                      n.iter=12000,
                      n.chains=4,
                      n.thin=2,
                      n.burnin=6000,
                      model.file="models/cfa_nomix_jags.txt")

summ2 <- fit2$BUGSoutput$summary

summ2


# # # # # # # # # # #
# Post processing and plots

# Extract classifications - mean posterior classification
# Class 2 is "bots" class 1 is "nonbots"

dat$est_class <- ifelse(summ1[paste0("Cend[", 1:395, "]"), "50%"] == 2,
                         "EstBot", "EstNoBot")

# # # # # # # # # # # # # #
# Existing bots in the data
# Duplicate Locations

dat$loc <- interaction(dat$LocationLatitude,dat$LocationLongitude)
dat$dupes <- ifelse(duplicated(dat$loc),"Bot","Nobot")

# Descriptives table
bot_dat <- dat[dat$dupes == "Bot",]
non_dat <- dat[dat$dupes == "Nobot",]

tabx <- data.frame(describe(non_dat)[,c("mean","median","sd")],
                   describe(bot_dat)[,c("mean","median","sd")])

# Temp colnames
colnames(tabx) <- c("NonBotMean","NonBotMed","NonBotSD",
                     "BotMean","BotMed","BotSD")

# LaTeX table 
kable(tabx, row.names = TRUE,format = "latex", booktabs = TRUE, digits = 2)

# Reformatting estimated classes and scores into a tidy (long) DF for plots
# Raw data plots

# Number of survey items
k = 25
holder <-  list()
for(i in 1:(nrow(ydat))){
  dati <- dat[i,24:48]
  temp <- data.frame("Score"=as.numeric(matrix(dati, ncol = 1)),
                     "ID"=rep(i,k),
                     "Index"=1:k, 
                     "Status"=rep(dat$dupes[i],k),
                     "Est" = rep(dat$est_class[i],k))
  holder[[i]]  <- temp
}
pdat <- do.call("rbind",holder)



# # # # # # # # # #
# ROC
# Diagnostic accuracy

# NOTE: Due to the stochastic nature of the analysis, these results could
# vary from the analysis presented in Roman, Brandt, and Miller, (2022)
# In particular if you changed MCMC inputs dramatically. 
# That said, it should be pretty close, as our sensitivity analysis
# indicated homogeneous and robust model results.

tab2 <- table(pdat$Status,pdat$Est)

TP <- tab2[1,1]
FP <- tab2[2,1]
FN <- tab2[1,2]
TN <- tab2[2,2]

Sensitivity <- (TP/(TP + FN)) *100

Sensitivity

Specificity <- (TN/(TN+FP))*100

Specificity

PPV <- TP/(TP + FP) *100

NPV <- TN/(TN + FN) *100

PPV
NPV

# # # # # # # # # #
# Standardized Loadings
# Used for comparisons across models

# Extract covariance matrix
# Factor 1 = RWA (items 1:10)
# Factor 2 = SDO (items 11:18)
# Factor 3 = NAT (items 19:25)

# Standardizing by factor for transparency

# RWA
ly <- c(1, summ1[paste0("ly[", 1:9, "]"), "mean"]) # Isolate loading estimates
# First "1" is LV scaling cons.
sigxi2 <- summ1["sigmaxi[1,1]", "mean"] # Isolate factor variances estimates
sigy2 <- summ1[paste0("sigmay[", 1:10, "]"), "mean"] # Isolate Item variances
ly_rwa <- vx <- ly
for (j in 1:10) {
  #model-implied variance of items
  vx[j] <- ly[j] ^ 2 * sigxi2 + sigy2[j]
  # standardized factor loadings
  ly_rwa[j] <- ly[j] * sqrt(sigxi2 / vx[j])
}

#SDO
ly <- c(1, summ1[paste0("ly[", 10:16, "]"), "mean"])
sigxi2 <- summ1["sigmaxi[2,2]", "mean"]
sigy2 <- summ1[paste0("sigmay[", 11:18, "]"), "mean"]
ly_sdo <- vx <- ly
for (j in 1:8) {
  vx[j] <- ly[j] ^ 2 * sigxi2 + sigy2[j]
  ly_sdo[j] <- ly[j] * sqrt(sigxi2 / vx[j])
}

#NAT
ly <- c(1, summ1[paste0("ly[", 17:22, "]"), "mean"])
sigxi2 <- summ1["sigmaxi[3,3]", "mean"]
sigy2 <- summ1[paste0("sigmay[", 19:25, "]"), "mean"]
ly_nat <- vx <- ly
for (j in 1:7) {
  vx[j] <- ly[j] ^ 2 * sigxi2 + sigy2[j]
  ly_nat[j] <- ly[j] * sqrt(sigxi2 / vx[j])
}

StdClass <- data.frame("StdLoadLC" = c(ly_rwa, ly_sdo, ly_nat))

#RWA
ly <- c(1, summ2[paste0("ly[", 1:9, "]"), "mean"])
sigxi2 <- summ2["sigmaxi[1,1]", "mean"]
sigy2 <- summ2[paste0("sigmay[", 1:10, "]"), "mean"]

ly_rwa <- vx <- ly
for (j in 1:10) {
  vx[j] <- ly[j] ^ 2 * sigxi2 + sigy2[j]

  ly_rwa[j] <- ly[j] * sqrt(sigxi2 / vx[j])
}

#SDO
ly <- c(1, summ2[paste0("ly[", 10:16, "]"), "mean"])
sigxi2 <- summ2["sigmaxi[2,2]", "mean"]
sigy2 <- summ2[paste0("sigmay[", 11:18, "]"), "mean"]

ly_sdo <- vx <- ly
for (j in 1:8) {
  vx[j] <- ly[j] ^ 2 * sigxi2 + sigy2[j]
  ly_sdo[j] <- ly[j] * sqrt(sigxi2 / vx[j])
}

#NAT
ly <- c(1, summ2[paste0("ly[", 17:22, "]"), "mean"])
sigxi2 <- summ2["sigmaxi[3,3]", "mean"]
sigy2 <- summ2[paste0("sigmay[", 19:25, "]"), "mean"]

ly_nat <- vx <- ly
for (j in 1:7) {
  vx[j] <- ly[j] ^ 2 * sigxi2 + sigy2[j]
  ly_nat[j] <- ly[j] * sqrt(sigxi2 / vx[j])
}

StdNclass <- data.frame("StdLoadCFA" = c(ly_rwa, ly_sdo, ly_nat))

# Item names for plots
Lnam <- c("RWA 1","RWA 2","RWA 3","RWA 4","RWA 5","RWA 6","RWA 7",
          "RWA 8","RWA 9","RWA 10","SDO 1","SDO 2","SDO 3","SDO 4",
          "SDO 5","SDO 6","SDO 7","SDO 8", "NAT 1","NAT 2","NAT 3",
          "NAT 4","NAT 5","NAT 6","NAT 7")

Pload <- data.frame(StdNclass,StdClass,"Item" = rep(Lnam,2))

# # # # # # # # # # 
# Plot

ggplot(data = Pload) +
  geom_segment(x = 0, xend = 1.0, y = 0, yend = 1.0, lty = 2) +
  geom_label(aes(x = StdLoadCFA,y = StdLoadLC, label = Item)) +
  xlim(0.6,1.0) + ylim(0.6,1.0) +
  theme_minimal() +
  ylab("Standardized LC-CFA Loadings") +
  xlab("Standardized CFA Loadings")


ggplot(data = Pload) +
  geom_point(aes(x = StdLoadCFA,y = StdLoadLC)) +
  geom_segment(x = 0, xend = 1.0, y = 0, yend = 1.0, lty = 2) +
  xlim(0.6,1.0) + ylim(0.6,1.0) +
  theme_minimal() +
  ylab("Standardized LC-CFA Loadings") +
  xlab("Standardized CFA Loadings")

# # # # # # # # # # # #
# Factor correlations 
# Derived from covariances

##LC
# Factor variances
xi11 <- summ1[c("sigmaxi[1,1]"),"mean"]
xi22 <- summ1[c("sigmaxi[2,2]"),"mean"]
xi33 <- summ1[c("sigmaxi[3,3]"),"mean"]
# Factor covariances
xi21 <- summ1[c("sigmaxi[2,1]"),"mean"]
xi31 <- summ1[c("sigmaxi[3,1]"),"mean"]
xi32 <- summ1[c("sigmaxi[3,2]"),"mean"]

LCcorxi21 <- xi21/(xi22*xi11)
LCcorxi31 <- xi31/(xi33*xi11)
LCcorxi32 <- xi32/(xi33*xi22)

##CFA
# Factor variances
xi11 <- summ2[c("sigmaxi[1,1]"),"mean"]
xi22 <- summ2[c("sigmaxi[2,2]"),"mean"]
xi33 <- summ2[c("sigmaxi[3,3]"),"mean"]
# Factor covariances
xi21 <- summ2[c("sigmaxi[2,1]"),"mean"]
xi31 <- summ2[c("sigmaxi[3,1]"),"mean"]
xi32 <- summ2[c("sigmaxi[3,2]"),"mean"]

CFAcorxi21 <- xi21/(xi22*xi11)
CFAcorxi31 <- xi31/(xi33*xi11)
CFAcorxi32 <- xi32/(xi33*xi22)


CFAcor <- c(CFAcorxi21,CFAcorxi31,CFAcorxi32)
LCcor <- c(LCcorxi21,LCcorxi31,LCcorxi32)

ggplot() +
  geom_point(aes(x = CFAcor,y = LCcor)) +
  geom_segment(x = 0.01, xend = 0.19, y = 0.01, yend = 0.19, lty = 2) +
  xlim(0,0.2) + ylim(0,0.2) +
  theme_minimal() 





