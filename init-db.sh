#!/bin/sh
set -e

psql -v ON_ERROR_STOP=1 --username "langgraph_user" --dbname "langgraph_db" <<-EOSQL
    CREATE USER langgraph_user WITH PASSWORD 'langgraph_password';
    CREATE DATABASE langgraph_db;
    GRANT ALL PRIVILEGES ON DATABASE langgraph_db TO langgraph_user;
    \c langgraph_db
    GRANT ALL ON SCHEMA public TO langgraph_user;
    GRANT CREATE ON SCHEMA public TO langgraph_user;
EOSQL