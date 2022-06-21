# Zachary Roman Ph.D. <ZacharyJoseph.Roman@uzh.ch>
# Supporting functions file for unsupervised bot classification model in Roman, Brandt, 
# and Miller (2022).

# Person-fit index function 
# Section 2.2, page 3, Roman, Brandt, and Miller (2022)
# ydat: the matrix of observed variables
# mod: the lavaan syntax specifying the factor structure
cdfun <- function(ydat,mod){
  upsilon <- ydat
  p <- dim(ydat)[2]
  N  <- dim(ydat)[1]
  sem0 <- sem(mod,ydat,meanstructure =T)
  sigma0 <- fitted(sem0)
  mu0 <- apply(ydat,2,mean)
  sig0 <- cov(ydat)
  md1 <- mahalanobis(ydat,center=mu0,cov=sig0)
  md2 <- mahalanobis(ydat,center=sigma0$mean,cov=sigma0$cov)
  cd1 <- -0.5*(p*log(2*pi)+log(det(sig0))+md1)
  cd2 <- -0.5*(p*log(2*pi)+log(det(sigma0$cov))+md2)
  upsilon <- -2*(cd2-cd1)
  return(upsilon)
}