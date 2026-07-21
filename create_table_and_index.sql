CREATE TABLE nation (n_nationkey INT NOT NULL PRIMARY KEY, n_name CHAR(25), n_regionkey INT, n_comment VARCHAR(152), n_dummy VARCHAR(10));
docker cp nation.tbl sql_server_test:/
BULK INSERT nation FROM '/nation.tbl' WITH (FIELDTERMINATOR = '|', ROWTERMINATOR = '\n');


CREATE TABLE region (r_regionkey INT NOT NULL PRIMARY KEY, r_name CHAR(25), r_comment VARCHAR(152),r_dummy VARCHAR(10));
docker cp region.tbl sql_server_test:/
BULK INSERT region FROM '/region.tbl' WITH (FIELDTERMINATOR = '|', ROWTERMINATOR = '\n');


CREATE TABLE supplier (s_suppkey INT NOT NULL PRIMARY KEY, s_name CHAR(25), s_address VARCHAR(40), s_nationkey INT, s_phone CHAR(15), s_acctbal DECIMAL(15,2), s_comment VARCHAR(101), s_dummy VARCHAR(10));
docker cp supplier.tbl sql_server_test:/
BULK INSERT supplier FROM '/supplier.tbl' WITH (FIELDTERMINATOR = '|', ROWTERMINATOR = '\n');


CREATE TABLE customer (c_custkey INT NOT NULL PRIMARY KEY, c_name VARCHAR(25), c_address VARCHAR(40), c_nationkey INT, c_phone CHAR(15), c_acctbal DECIMAL(15,2), c_mktsegment CHAR(10), c_comment VARCHAR(117), c_dummy VARCHAR(10));
docker cp customer.tbl sql_server_test:/
BULK INSERT customer FROM '/customer.tbl' WITH (FIELDTERMINATOR = '|', ROWTERMINATOR = '\n');


CREATE TABLE part ( p_partkey INT NOT NULL PRIMARY KEY, p_name VARCHAR(55), p_mfgr CHAR(25), p_brand CHAR(10), p_type VARCHAR(25), p_size INT, p_container CHAR(10), p_retailprice DECIMAL(15,2), p_comment VARCHAR(23), p_dummy VARCHAR(10));
docker cp part.tbl sql_server_test:/
BULK INSERT part FROM '/part.tbl' WITH (FIELDTERMINATOR = '|', ROWTERMINATOR = '\n');



CREATE TABLE partsupp (ps_partkey INT NOT NULL, ps_suppkey INT NOT NULL, ps_availqty INT, ps_supplycost DECIMAL(15,2), ps_comment VARCHAR(199), ps_dummy VARCHAR(10), PRIMARY KEY(ps_partkey, ps_suppkey));
docker cp partsupp.tbl sql_server_test:/
BULK INSERT partsupp FROM '/partsupp.tbl' WITH (FIELDTERMINATOR = '|', ROWTERMINATOR = '\n');


CREATE TABLE orders (o_orderkey INT NOT NULL PRIMARY KEY, o_custkey INT, o_orderstatus CHAR(1), o_totalprice DECIMAL(15,2), o_orderdate DATE, o_orderpriority CHAR(15), o_clerk CHAR(15), o_shippriority INT, o_comment VARCHAR(79), o_dummy VARCHAR(10));
docker cp orders.tbl sql_server_test:/
BULK INSERT orders FROM '/orders.tbl' WITH (FIELDTERMINATOR = '|', ROWTERMINATOR = '\n');



CREATE TABLE lineitem (l_orderkey INT, l_partkey INT, l_suppkey INT, l_linenumber INT, l_quantity DECIMAL(15,2), l_extendedprice DECIMAL(15,2), l_discount DECIMAL(15,2), l_tax DECIMAL(15,2), l_returnflag CHAR(1), l_linestatus CHAR(1), l_shipdate DATE, l_commitdate DATE, l_receiptdate DATE, l_shipinstruct CHAR(25), l_shipmode CHAR(10), l_comment VARCHAR(44), l_dummy VARCHAR(10), PRIMARY KEY(l_orderkey, l_linenumber));
docker cp lineitem.tbl sql_server_test:/
BULK INSERT lineitem FROM '/lineitem.tbl' WITH (FIELDTERMINATOR = '|', ROWTERMINATOR = '\n');


##### alter index name to align with the index name in python:

alter table region drop CONSTRAINT ORIGINAL_NAME;
CREATE CLUSTERED INDEX PK__REGION__F403C3F000643726 ON region(R_REGIONKEY ASC);


alter table nation drop CONSTRAINT ORIGINAL_NAME;
CREATE CLUSTERED INDEX PK__NATION__AF64455CF27143A4 ON nation(n_nationkey ASC);


alter table supplier drop CONSTRAINT ORIGINAL_NAME;
CREATE CLUSTERED INDEX PK__SUPPLIER__3632082B20AB4470 ON supplier(s_suppkey ASC);



alter table customer drop CONSTRAINT ORIGINAL_NAME;
CREATE  INDEX Index_customer_1 ON customer(c_custkey ASC);



alter table part drop CONSTRAINT ORIGINAL_NAME;
CREATE  CLUSTERED INDEX PK__PART__7FC1E95F40C7D867 ON part(p_partkey ASC);



alter table partsupp drop CONSTRAINT PK__partsupp__FB4383D2787838A1;
# drop index PK__PARTSUPP__54937F6995C19898 on partsupp;
CREATE  CLUSTERED INDEX PK__PARTSUPP__54937F6995C19898 ON partsupp(ps_partkey ASC, ps_suppkey ASC);


alter table orders drop CONSTRAINT PK__orders__42185E85AB39C580;
# drop index PK__ORDERS__AAA6619D7C63EC05 on orders;
CREATE  CLUSTERED INDEX PK__ORDERS__AAA6619D7C63EC05 ON orders(o_orderkey ASC);



alter table lineitem drop CONSTRAINT PK__lineitem__64740AD2985C7FB9;
# drop index PK__LINEITEM__DD1C9C94D3616509 on lineitem;
CREATE  CLUSTERED INDEX PK__LINEITEM__DD1C9C94D3616509 ON lineitem(l_orderkey ASC, l_linenumber ASC);
