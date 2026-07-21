# LTR4QO
leveraging learning-to-rank for query optimization

### Setup
###### 1. Install MSSQL with Docker and Run a container as MSSQL server:

```
sudo docker pull mcr.microsoft.com/mssql/server:2022-latest

docker run -e "ACCEPT_EULA=Y" -e "MSSQL_SA_PASSWORD=YOUR_PASSWORD" \
   -p 1433:1433 --name sql_server_test --hostname sql_server_test -d mcr.microsoft.com/mssql/server:2022-latest
```
######  2. install ODBC driver and pyodbc library for SQL Server connection with Python
```
https://learn.microsoft.com/en-us/sql/connect/odbc/linux-mac/installing-the-microsoft-odbc-driver-for-sql-server?view=sql-server-ver16&tabs=ubuntu18-install%2Calpine17-install%2Cdebian8-install%2Credhat7-13-install%2Crhel7-offline
pip install pyodbc
```
###### 3. open MSSQL terminal and create databases (e.g., TPCH)

```
 docker exec -it sql_server_test "bash" /opt/mssql-tools18/bin/sqlcmd -S localhost -U sa -P "YOUR_PASSWORD" -No

 create database tpch;use tpch;go
```

###### 4. load tables and set index: 
```
 see create_table_and_index.sql
```

###### 4. install all necessary python libraries such as pyodbc, sklearn, torch. etc..


### Reproduce

###### 1. Generate random subqueries and subplans

````
bash LTR_enumerate_plans_VM_Random.sh RD train tpch None tpch1000  ### generate TPCH training subqueries/subplans 

````

###### 2. Generate vector training data

````
bash generate_train_data_VM.sh tpch1000 Q6D10P98Tree53435600%1 1000 linear  &

````

###### 3.a Training the model and directly do enumeration on its testing queries

````angular2html
bash train_and_enum_VM.sh linearR LTRankNet0 Ada tpch-d tpch 1thSamTrain50%FixValid20%SamSize20NoScheduler tpch1000 Q6D10Tree53435600%1 &

````

###### 3.b Assume an trained model already exists, just do ennumeration on testing queries

````
bash LTR_enumerate_plans_VM.sh Ada tpch-d tpch MODEL_LTRankNet0_lambdaloss_sigma_0.5_k_5_mu_1.0_NDCG_Loss2++_presort_False_TWL_tpch1000_Q6D10Tree53435600_linearR_ITER_1thSamTrain50%FixValid20%SamSize20NoScheduler None

````
