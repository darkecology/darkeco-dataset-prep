library(rgdal)
library(tmap)
library(gstat) # Use gstat's idw routine
library(sp)    # Used for the spsample function
library(raster)
library(maps)
library(mapdata)
library(data.table)
library(geosphere)
library(lubridate)
library(mgcv)
library(maptools)
library(viridis)
library(circular)
library(igraph)

setwd("~/Desktop/data_for_visual")
output_directory="map_output/"

#Read radar location data
meta = read.csv("meta/radar_locations.csv")

#Read gis layers for maps
usa_map = readOGR("map_layers/usa_merge.shp")
usa_map_state = readOGR("map_layers/ne_50m_admin_1_states_provinces_lakes_shp.shp")
country<- readOGR("map_layers/ne_10m_admin_0_countries.shp")
greatlakes<- readOGR("map_layers/greatlakes.shp")

#################################################################################
#Read the integrated radar data
daily_sum=fread("data/whole_year.csv")

#Format dates
daily_sum$samplingperiod=as.Date(daily_sum$samplingperiod)
daily_sum$ordinal=yday(daily_sum$samplingperiod)
daily_sum$year=substr(daily_sum$samplingperiod,1,4)

#This is the merge KJAN and KDGX, merge at 2004. They're basically at the same location
kjan04=subset(daily_sum, radar_id=="KJAN" & year==2004 & samplingperiod<"2004-04-01")
kdgx04=subset(daily_sum, radar_id=="KDGX" & year==2004 & samplingperiod>="2004-04-01")
kdgx04=rbind(kjan04, kdgx04)

daily_sum=subset(daily_sum, radar_id!="KDGX" | year!=2004)
daily_sum=subset(daily_sum, radar_id!="KJAN" | year!=2004)
daily_sum=rbind(daily_sum, kdgx04)
remove(kdgx04, kjan04)

#Switch all KJAN to KDGX
daily_sum$radar_id=ifelse(daily_sum$radar_id=="KJAN", "KDGX", daily_sum$radar_id)

#Very high clutter on this night
daily_sum=subset(daily_sum, samplingperiod!="2012-09-11" | radar_id!="KBBX")

#There's a lot of clutter in 1998
daily_sum=subset(daily_sum, year!=1998 | radar_id!="KEYX")

#This defines the seasons. This is important for the direction of the arrows. 
daily_sum$season=ifelse(daily_sum$ordinal>=(60) & daily_sum$ordinal<=(165),"spring", 
                        ifelse(daily_sum$ordinal>=(213) & daily_sum$ordinal<=(318),"fall", "nonmig")) 
daily_sum$percent_sampled=ifelse(daily_sum$season=="spring" | daily_sum$season=="fall", daily_sum$percent_sampled*2, daily_sum$percent_sampled)
daily_sum$percent_sampled=ifelse(daily_sum$percent_sampled>1.5, daily_sum$percent_sampled/2,daily_sum$percent_sampled)
daily_sum=subset(daily_sum, percent_sampled>=.75)

#Caclulate track and groundspeed from U and V directions
daily_sum$track = (180/pi)	*(atan2(daily_sum$u,daily_sum$v))
daily_sum$track=ifelse (daily_sum$track <0, (daily_sum$track+360), daily_sum$track)
daily_sum$groundspeed= sqrt(daily_sum$u^2+daily_sum$v^2)

#Subset to remove extreme groundspeeds
daily_sum=subset(daily_sum, groundspeed<30)


#Smooth and average all years by site using a GAM. This will take a while to run. Maybe 20 mintues. 
#Everything will be collected in "cm2_km". 
cm2_km_smooth=data.frame()
for (r in unique(daily_sum$radar_id)){
  print(r)
  radar_smooth=subset(daily_sum, radar_id==r)
  #this faimily presents predictions below 0
  mod1 <- gam(abs(cm2_km)~s(ordinal, k=10),data = radar_smooth,family=quasipoisson)
  pdat <- data.frame(min(radar_smooth$ordinal):max(radar_smooth$ordinal))
  colnames(pdat)<-c("ordinal")
  pred2 <- predict(mod1, pdat, type = "response", se.fit = TRUE)
  pdat2 <- transform(pdat, cm2_km_smooth = pred2$fit)
  pdat2$radar_id=r
  
  #turn this on if you want to see the smooth plots with each iteration of the loop
  #print(plot(pdat2$ordinal, (pdat2$cm2_km_smooth), main=r))
  cm2_km_smooth= rbind(pdat2, cm2_km_smooth)

}
remove(pdat, pdat2,pred2,r,mod1)

#Merge the smooth values with the entire dataset. 
daily_sum=merge(daily_sum, cm2_km_smooth, by=c("ordinal", "radar_id"))


#Merge the integrated data with the meta data. This is important to have the Latitude and Longitude 
#associated with the sites.
daily_sum=merge(daily_sum,meta, by="radar_id")

#This is to remove leap year days. 
daily_sum=subset(daily_sum, ordinal<=365)

#This is going to result in an averaged dataset for the entire year for each site 
daily_sum=aggregate(daily_sum[,c(4,9,10,16:18)], by=list(daily_sum$ordinal,daily_sum$radar_id), FUN=mean)
colnames(daily_sum)[1:2]=c("ordinal", "radar_id")

daily_sum$track = (180/pi)*(atan2(daily_sum$u,daily_sum$v))
daily_sum$track=ifelse (daily_sum$track <0, (daily_sum$track+360), daily_sum$track)
daily_sum$groundspeed= sqrt(daily_sum$u^2+daily_sum$v^2)
daily_sum=subset(daily_sum, groundspeed<30)

########################################################################
#This is to set the range of the scale. This was done manually. 
max_eta=631316.4
min_eta=0

#Create scale for the plot. Other opitions include inferno and viridis
color_bar=plasma(5)
color_bar[1]="#0D088715" #plasma
color_bar= colorRampPalette(color_bar,alpha=TRUE)
plot(rep(1,200),col=color_bar(200),pch=19,cex=3)

#This loop will make a national rasters that are smoothed using a GAM. One map per ordinal day. 
for (w in 1:365){
  print(w)
  df_week=subset(daily_sum, ordinal==w)
  
  r  = raster(ncol=200, nrow=200, res=.5)
  extent(r) = usa_map@bbox
  lat = lon <- r
  xy = coordinates(r)
  lon[] = xy[, 1]
  lat[] = xy[, 2]
  
  cr = stack(lat, lon)
  names(cr) = c("LATITUDE_N", "LONGITUDE_W")
  gam1 = gam(cm2_km_smooth~s(LATITUDE_N,k= 10)+s(LONGITUDE_W, k=9),data=df_week)
  
  raster_base=predict(cr, gam1,  progress='text')
  r.m =  mask(raster_base, usa_map)
  #print(raster_base@data@max)
  print(raster_base@data@max)
  
  r.m[is.na(r.m)] <- 0
  r.m[r.m< 0] <- 0
  
  r.m     <- mask(raster_base, usa_map)
  r.m[r.m< 0] <- 0
  ############################################
  #This creates the data for the arrows
  
  R = 6378.1 #Radius of the Earth
  brng=rad(df_week$track) #Bearing converted to radians.
  d = (df_week$groundspeed-min(df_week$groundspeed))*10 #Distance in km weighted by groundspeed
  d=ifelse(d<45,45, d)
  
  lat1 = rad(df_week$LATITUDE_N) #Current lat point converted to radians
  lon1 = rad(df_week$LONGITUDE_W) #Current long point converted to radians
  
  lat2 = asin( sin(lat1)*cos(d/R) +
                 cos(lat1)*sin(d/R)*cos(brng))
  lat2=deg(lat2)
  lon2 = lon1 + atan2(sin(brng)*sin(d/R)*cos(lat1),
                      cos(d/R)-sin(lat1)*sin(lat2))
  lon2=deg(lon2)
  df_week$LatEnd = lat2
  df_week$LonEnd = lon2
  
  ############################################
  #Image Output 
  
  png(paste(output_directory, "heat_",w, ".png", sep=""), width=18.5, height=10.5, bg="black", units="in", res= 300)
  map(usa_map_state,col = 'transparent', fill =T,lty = 1,lwd = .35 ,border="gray", mar = c(0,0,0,0),
      ylim=c(20,51),xlim=c(-125, -65))
  plot(r.m, add=T, col=color_bar(200),zlim=c(min_eta,(max_eta)), legend = F)
  map(country,col = 'transparent',fill = T,lty = 1,lwd = 1.75 ,border="darkgray", add=T)
  map(greatlakes, fill = T,lty = 1,lwd = 1.75 ,border="darkgray", add=T)
  map(usa_map_state,col = 'transparent', fill = T,lty = 1,lwd = .35 ,border="gray", add=T)
  map(usa_map_state,col = 'transparent', fill = T,lty = 1,lwd = .25 ,border="black", add=T)
  map(usa_map,col = 'transparent', fill = T,lty = 1,lwd = 2 ,border="black", add=T)
  map(usa_map,col = 'transparent', fill = T,lty = 1,lwd = 1.75 ,border="gray", add=T)
  points(df_week$LONGITUDE_W, df_week$LATITUDE_N, col = "white",  
         cex =abs(df_week$cm2_km)/max(abs(daily_sum$cm2_km))*16)
  igraph:::igraph.Arrows(df_week$LONGITUDE_W, df_week$LATITUDE_N, df_week$LonEnd, 
                         df_week$LatEnd,size=.75, width=1, sh.lwd=.75, sh.col="white")
  title_head<-format(as.Date(w-1, origin=as.Date("2015-01-01")),"%B %d")
  title(main=title_head, col.main= "white", cex.main = 3.5, font.main = 1,line = -50, adj=0.115)
  dev.off()
  
}


