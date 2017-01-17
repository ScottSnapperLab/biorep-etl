library(RSQLite)

#create dataframe from csv
file<-"./data/raw/SnapperGIBioreposito_DATA_2017-01-10_1643.csv"
df<-read.csv(file, header = TRUE, sep = ",")

#generate compound key for visit entity
df$vis.id<-paste(df$biorepidnumber,df$sample_date)
vis.id<-df$vis.id

#generate family and member id for subject entity
fam<-as.character(format(round(df$biorepidnumber, digits =2)))
fam<-as.data.frame(strsplit(fam, '[.]'))
fam<-t(fam)
df$family_id<-fam[,1]
df$member_id<-fam[,2]

#reference
t<-as.data.frame(t(df))
id<-c(1:nrow(t))
t$V1<-id

#create Database
#database<-dbConnect(SQLite(), dbname="db.sqlite")


#######################################################################################################################
#SUBJECT ENTITY


#create dataframe for Subject
sub<-as.data.frame(df[c(5,9,10,12:34,36:51,215,216)])
sub<-sub[!duplicated(sub),]

#check for uniqueness of primary key
n_occur <- data.frame(table(sub[1]))
n_occur[n_occur$Freq > 1,]
#1414 reoccuring ids

write.csv(sub, file = "./data/processed/subject.csv", row.names = F)

# #dbWriteTable(conn = database, "sub", sub, row.names = F)
# 
# 
# #create Subject table
# #dbSendQuery(conn = database,
# #            "CREATE TABLE subject
# #           (BioRepo_Id TEXT PRIMARY KEY,
# #            Registry_Consent INTEGER,
# #            ETC ETC ETC )");
# 
# dbGetQuery(conn = database, "SELECT * FROM sub")
# dbGetQuery(conn = database, "INSERT INTO subject SELECT * FROM sub")
# dbGetQuery(conn = database, "SELECT * FROM subject")
# dbRemoveTable(database, "sub")

#######################################################################################################################
#SAMPLE ENTITY


#create dataframe for Samples
sam<-as.data.frame(df[c(1:3,5,6:8,11,56:126,214)])
sam<-sam[!duplicated(sam),]

#check uniqueness of sample primary key
n_occur <- data.frame(table(sam[1]))
n_occur[n_occur$Freq > 1,]
#0 reoccuring ids

write.csv(sam, file = "./data/processed/sample.csv", row.names = F)

# dbWriteTable(conn = database, "sam", sam, row.names = T)
# 
# #create sample table
# dbSendQuery(conn = database,
#             "CREATE TABLE sample
#             (Record_Id INTEGER,
#             Sample_Id TEXT PRIMARY KEY,
#             Label_Id TEXT,
#             BioRepo_Id TEXT,
#             ETC...
#             ETC...
#             FOREIGN KEY (BioRepo_Id) REFERENCES subject(BioRepo_Id),
#             FOREIGN KEY (Visit_Id) REFERENCES visit(Visit_Id))");
# 
# dbGetQuery(conn = database, "SELECT * FROM sam")
# dbGetQuery(conn = database, "INSERT INTO sample SELECT * FROM sam")
# dbGetQuery(conn = database, "SELECT * FROM sample")
# dbRemoveTable(database, "sam")

#######################################################################################################################

#VISIT ENTITY


#create dataframe for Visit
vis<-as.data.frame(df[c(214,5,8,2,54,55,127:213)])
vis<-vis[!duplicated(vis),]

#check for uniqeness of primary key
n_occur <- data.frame(table(vis[1]))
n_occur[n_occur$Freq > 1,]
#2350 reoccuring keys

write.csv(vis, file = "./data/processed/visit.csv", row.names = F)

# dbWriteTable(conn = database, "vis", vis, row.names = T)
# 
# #create visit table
# dbSendQuery(conn = database,
#             "CREATE TABLE visit
#             (Visit_Id TEXT PRIMARY KEY,
#             BioRepo_Id TEXT,
#             Sample_Date TEXT,
#             Sample_Id TEXT,
#             Nsaid INTEGER,
#             ETC...
#             ETC...
#             FOREIGN KEY (BioRepo_Id) REFERENCES subject(BioRepo_Id),
#             FOREIGN KEY (Sample_Id) REFERENCES sample(Sample_Id))");
# 
# dbGetQuery(conn = database, "SELECT * FROM vis")
# dbGetQuery(conn = database, "INSERT INTO visit SELECT * FROM vis")
# dbGetQuery(conn = database, "SELECT * FROM visit")
# dbRemoveTable(database, "vis")

#####################################################################################################################

print("Finished.")
######################################################################################################################